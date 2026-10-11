"""Retain observed hours/minutes, backfill the archive and publish count inputs.

No training, interpolation, zero filling or weather downloads. Keep one writer
running per store/output pair. An hourly archive value takes priority over a
minute sum; minute sums require all 60 distinct valid readings.
"""
import argparse
import io
import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .clean_data import clean_counts
    from .load_data import API, COUNTS, MINUTES, ROOT, get, selected_ids
    from .observations import input_status, read_counts, utc
else:
    from clean_data import clean_counts
    from load_data import API, COUNTS, MINUTES, ROOT, get, selected_ids
    from observations import input_status, read_counts, utc


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ObservationStore:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS hours (
                sensor INTEGER, stamp INTEGER, count INTEGER NOT NULL,
                priority INTEGER NOT NULL, PRIMARY KEY(sensor,stamp));
            CREATE TABLE IF NOT EXISTS minutes (
                sensor INTEGER, stamp INTEGER, count INTEGER,
                PRIMARY KEY(sensor,stamp));
            CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT);
        ''')

    def close(self):
        self.db.close()

    def save_hours(self, frame, priority, now):
        end = utc(now).floor('h')
        valid = frame.loc[frame.timestamp.lt(end) & frame['count'].notna()]
        rows = [(int(r.location_id), int(r.timestamp.timestamp()), int(r.count), priority)
                for r in valid.itertuples(index=False)]
        with self.db:
            self.db.executemany('''INSERT INTO hours VALUES (?,?,?,?)
                ON CONFLICT(sensor,stamp) DO UPDATE SET count=excluded.count,
                priority=excluded.priority WHERE excluded.priority >= hours.priority''', rows)
        return len(rows)

    def seed(self, path, now):
        if self.db.execute("SELECT 1 FROM state WHERE key='seeded'").fetchone():
            return
        path = Path(path)
        if path.is_file():
            parts = [y.dropna().rename('count').rename_axis('timestamp').reset_index().assign(location_id=sid)
                     for sid, y in read_counts(path).items()]
            if parts:
                self.save_hours(pd.concat(parts, ignore_index=True), 1, now)
        with self.db:
            self.db.execute("INSERT INTO state VALUES ('seeded','yes')")

    def save_minutes(self, frame, ids, now):
        required = {'location_id', 'sensing_datetime', 'total_of_directions'}
        if not required.issubset(frame):
            raise ValueError('Minute feed is missing required columns.')
        if frame.empty:
            return dict(minute_rows=0, invalid_minutes=0)
        text = frame.sensing_datetime.astype(str)
        if not text.str.contains(r'(?:Z|[+-]\d{2}:\d{2})$').all():
            raise ValueError('Live timestamps require UTC/offset timezone.')
        stamps = pd.to_datetime(text, utc=True, errors='raise')
        sids = pd.to_numeric(frame.location_id, errors='raise')
        values = pd.to_numeric(frame.total_of_directions, errors='coerce')
        if not (sids.gt(0) & sids.mod(1).eq(0)).all() or not stamps.eq(stamps.dt.floor('min')).all():
            raise ValueError('Minute feed has invalid sensor IDs or unaligned timestamps.')
        valid = np.isfinite(values) & values.ge(0) & values.mod(1).eq(0)
        rows = [(int(sid), int(stamp.timestamp()), int(value) if ok else None)
                for sid, stamp, value, ok in zip(sids, stamps, values, valid)
                if int(sid) in ids and stamp < utc(now).floor('min')]
        with self.db:
            # Identical repeated snapshots are idempotent. Conflicting or invalid
            # readings stay unknown until an authoritative hourly total arrives.
            self.db.executemany('''INSERT INTO minutes VALUES (?,?,?)
                ON CONFLICT(sensor,stamp) DO UPDATE SET count=CASE
                    WHEN minutes.count=excluded.count THEN minutes.count ELSE NULL END''', rows)
            self.db.execute('DELETE FROM hours WHERE priority=2')
            self.db.execute('''INSERT INTO hours
                SELECT sensor, (stamp/3600)*3600, sum(count), 2 FROM minutes
                WHERE stamp < ? GROUP BY sensor, (stamp/3600)*3600
                HAVING count(*)=60 AND count(count)=60
                ON CONFLICT(sensor,stamp) DO UPDATE SET count=excluded.count,
                priority=2 WHERE hours.priority <= 2''', (int(utc(now).floor('h').timestamp()),))
        return dict(minute_rows=len(rows), invalid_minutes=int((~valid).sum()))

    def counts(self):
        frame = pd.read_sql_query('SELECT sensor AS location_id, stamp, count FROM hours ORDER BY sensor,stamp', self.db)
        if frame.empty:
            return pd.DataFrame(columns=['timestamp', 'location_id', 'count'])
        frame['timestamp'] = pd.to_datetime(frame.stamp, unit='s', utc=True)
        parts = []
        for sid, group in frame.groupby('location_id'):
            y = group.set_index('timestamp')['count'].sort_index()
            grid = pd.date_range(y.index.min(), y.index.max(), freq='h')
            parts.append(y.reindex(grid).rename('count').rename_axis('timestamp').reset_index().assign(location_id=sid))
        return pd.concat(parts, ignore_index=True)[['timestamp', 'location_id', 'count']]


def refresh(store, data, selection, artifacts, days=35, include_archive=True, now=None, fetch=get):
    now = utc(now) if now is not None else pd.Timestamp.now(tz='UTC')
    ids = selected_ids(selection)
    selected = json.loads(Path(selection).read_text())
    where = f"location_id IN ({','.join(map(str, ids))})"
    data = Path(data)
    store.seed(data/'counts.csv', now)
    sources, errors = {}, []
    if include_archive:
        start = (now.tz_convert('Australia/Melbourne')-pd.Timedelta(days=days)).date()
        try:
            raw = pd.read_csv(io.BytesIO(fetch(API+COUNTS+'/exports/csv',
                                 dict(where=f"{where} AND sensing_date >= date'{start}'", delimiter=','))))
            if raw.empty:
                sources['archive_rows'] = 0
            else:
                cleaned, quality = clean_counts(raw, ids)
                sources['archive_rows'] = store.save_hours(cleaned, 3, now)
                sources['archive_quality'] = quality
        except Exception as exc:
            errors.append(dict(source='hourly_archive', reason=str(exc)))
    try:
        live = pd.read_csv(io.BytesIO(fetch(API+MINUTES+'/exports/csv', dict(where=where, delimiter=','))))
        sources.update(store.save_minutes(live, ids, now))
    except Exception as exc:
        errors.append(dict(source='minute_feed', reason=str(exc)))
    frame = store.counts()
    # Preserve gaps between old archive and newer live observations; never require
    # the live window to touch the archive, and never invent the intervening hours.
    atomic_write(data/'counts.csv', frame.to_csv(index=False))
    counts = {int(sid): g.set_index('timestamp')['count'] for sid, g in frame.groupby('location_id')}
    report = input_status(counts, selected, artifacts, now)
    report.update(sources=sources, download_errors=errors,
                  minute_policy='60 distinct valid minute observations per hour; absent minutes remain unknown.',
                  collection_store='Persistent SQLite; trained artifacts are not modified.')
    atomic_write(data/'counts_status.json', json.dumps(report, indent=2, allow_nan=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data/cleaned')
    parser.add_argument('--selection', type=Path, default=ROOT/'config/cbd_sensor_selection.json')
    parser.add_argument('--artifacts', type=Path, default=ROOT/'artifacts')
    parser.add_argument('--store', type=Path, default=ROOT/'data/live/observations.sqlite')
    parser.add_argument('--days', type=int, default=35, help='Archive lookback for recent backfill.')
    parser.add_argument('--watch', action='store_true', help='Keep collecting; stop with Ctrl+C.')
    parser.add_argument('--interval', type=int, default=600, help='Minute feed polling interval in seconds.')
    parser.add_argument('--archive-interval', type=int, default=3600)
    args = parser.parse_args()
    if args.days < 1 or args.interval < 60 or args.archive_interval < 60:
        parser.error('days must be positive; intervals must be at least 60 seconds.')
    store = ObservationStore(args.store)
    last_archive = float('-inf')
    try:
        while True:
            archive_due = time.monotonic()-last_archive >= args.archive_interval
            report = refresh(store, args.data, args.selection, args.artifacts, args.days, archive_due)
            if archive_due and not any(e['source']=='hourly_archive' for e in report['download_errors']):
                last_archive = time.monotonic()
            print(json.dumps(dict(generated_at=report['generated_at'],
                  counts_ready_sensors=report['counts_ready_sensors'], selected_sensors=report['selected_sensors'],
                  download_errors=report['download_errors'], report=str(args.data/'counts_status.json'))), flush=True)
            if not args.watch:
                return 1 if report['download_errors'] else 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0
    finally:
        store.close()


if __name__ == '__main__':
    raise SystemExit(main())
