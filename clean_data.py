import os
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from psycopg2 import sql
from psycopg2.extras import execute_values
from dotenv import load_dotenv

from database.helper import get_db_connection

if __package__:
    from .load_data import ROOT, WEATHER, selected_ids
else:
    from load_data import ROOT, WEATHER, selected_ids

TZ = 'Australia/Melbourne'
ENV_FILE = ROOT / '.env'
SCHEMA = 'digital_twin_v2'

def clean_counts(frame, ids=None):
    frame = frame.copy()
    required = {'location_id', 'sensing_date', 'hourday', 'pedestriancount'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Counts need columns: {sorted(required)}')
    sid = pd.to_numeric(frame.location_id, errors='coerce')
    hour = pd.to_numeric(frame.hourday, errors='coerce')
    dates = pd.to_datetime(frame.sensing_date, errors='coerce')
    # This source contains local calendar dates, not timezone-bearing instants.
    if dates.dt.tz is not None:
        raise ValueError('sensing_date must be a Melbourne calendar date without an offset')
    valid = sid.gt(0) & sid.mod(1).eq(0) & hour.between(0, 23) & hour.mod(1).eq(0) & dates.notna()
    if ids is not None:
        valid &= sid.isin(ids)
    frame = frame.loc[valid].copy()
    if frame.empty:
        raise ValueError('No valid sensor/hour rows after filtering.')
    frame['location_id'] = sid.loc[valid].astype(int)
    wall = dates.loc[valid].dt.normalize() + pd.to_timedelta(hour.loc[valid], unit='h')
    frame['timestamp'] = wall.dt.tz_localize(TZ, ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC')
    report = {'input_rows': len(sid), 'invalid_or_excluded_rows': int((~valid).sum()),
              'ambiguous_or_nonexistent_local_hours': int(frame.timestamp.isna().sum())}
    frame = frame.dropna(subset=['timestamp'])
    if frame.empty:
        raise ValueError('No unambiguous timestamps remain.')
    count = pd.to_numeric(frame.pedestriancount, errors='coerce')
    bad = ~np.isfinite(count) | count.lt(0) | count.mod(1).ne(0)
    frame['count'] = count.mask(bad)
    report['invalid_counts'] = int(bad.sum())
    frame = frame[['location_id', 'timestamp', 'count']]
    before = len(frame)
    frame = frame.drop_duplicates()
    report['identical_duplicates_removed'] = before - len(frame)
    conflicts = frame.duplicated(['location_id', 'timestamp'], keep=False)
    if conflicts.any():
        example = frame.loc[conflicts, ['location_id', 'timestamp']].iloc[0].to_dict()
        raise ValueError(f'Conflicting sensor-hour records require review: {example}')
    parts = []
    for location_id, group in frame.groupby('location_id'):
        values = group.set_index('timestamp')['count'].sort_index()
        grid = pd.date_range(values.index.min(), values.index.max(), freq='h')
        parts.append(pd.DataFrame({'timestamp': grid, 'location_id': location_id,
                                   'count': values.reindex(grid).to_numpy()}))
    result = pd.concat(parts, ignore_index=True)
    report['sensors'] = int(result.location_id.nunique())
    report['missing_hours'] = int(result['count'].isna().sum())
    return result, report


def extend_complete_minutes(counts, live):
    """Append complete hours after each archive."""
    required = {'location_id', 'sensing_datetime', 'total_of_directions'}
    if not required.issubset(live.columns):
        raise ValueError(f'Live input needs {sorted(required)}')
    
    if live.empty:
        return counts, 0
    
    live = live.copy()
    text = live.sensing_datetime.astype(str)

    # Verify valid timestamp string
    if not text.str.contains(r'(?:Z|[+-]\d{2}:\d{2})$').all():
        raise ValueError('Live timestamps need an explicit UTC/offset timezone.')

    # Data preprocessing
    live['timestamp'] = pd.to_datetime(text, utc=True, errors='coerce')
    live['location_id'] = pd.to_numeric(live.location_id, errors='coerce')
    live['count'] = pd.to_numeric(live.total_of_directions, errors='coerce')
    live = live.dropna(subset=['timestamp', 'location_id']).drop_duplicates(['location_id', 'timestamp', 'count'])

    valid = (live.timestamp.eq(live.timestamp.dt.floor('min')) & np.isfinite(live['count']) &
             live['count'].ge(0) & live['count'].mod(1).eq(0))
    live['count'] = live['count'].where(valid)
    live['hour'] = live['timestamp'].dt.floor('h')
    aggregated = live.groupby(['location_id', 'hour'])['count'].agg(['sum']).reset_index()

    # Add hourly count data from live to historical count data
    complete_hour = live['hour'].max() - pd.Timedelta(hours=1)
    rows = (
        aggregated[aggregated['hour'] <= complete_hour][['hour', 'location_id', 'sum']]
        .rename(columns={'hour': 'timestamp', 'sum': 'count'})
    )
    counts = pd.concat([counts, pd.DataFrame(rows)], ignore_index=True).sort_values(['location_id', 'timestamp'])
    return counts, len(rows)


def clean_weather(frame):
    if not {'timestamp', *WEATHER}.issubset(frame.columns):
        raise ValueError('Weather needs timestamp and ' + ', '.join(WEATHER))
    frame = frame.copy()
    if not frame.timestamp.astype(str).str.contains(r'(?:Z|[+-]\d{2}:\d{2})$').all():
        raise ValueError('Weather timestamps need an explicit UTC/offset timezone.')
    frame['timestamp'] = pd.to_datetime(frame.timestamp, utc=True, errors='coerce')
    frame = frame.dropna(subset=['timestamp'])
    if frame.empty or not frame.timestamp.eq(frame.timestamp.dt.floor('h')).all():
        raise ValueError('Weather must contain hour-aligned timestamps.')
    for col in WEATHER:
        values = pd.to_numeric(frame[col], errors='coerce')
        valid = np.isfinite(values)
        if col != 'temperature_2m':
            valid &= values.ge(0)
        frame[col] = values.where(valid)
    frame = frame[['timestamp', *WEATHER]].drop_duplicates()
    if frame.timestamp.duplicated().any():
        raise ValueError('Conflicting weather timestamps require review.')
    frame = frame.set_index('timestamp').sort_index()
    return frame.reindex(pd.date_range(frame.index.min(), frame.index.max(), freq='h')).rename_axis('timestamp').reset_index()


def _db_value(value):
    """Convert pandas/NumPy missing values to SQL NULL."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def upsert_dataframe(conn, schema, table, frame, columns, conflict_columns):
    """
    Insert rows in batches. Conflicting keys update all non-key columns.
    Table and column names are safely quoted.
    """
    if frame.empty:
        return 0

    rows = [
        tuple(_db_value(value) for value in row)
        for row in frame[columns].itertuples(index=False, name=None)
    ]

    update_columns = [
        col for col in columns if col not in conflict_columns
    ]

    if update_columns:
        action = sql.SQL("DO UPDATE SET {}").format(
            sql.SQL(", ").join(
                sql.SQL("{} = EXCLUDED.{}").format(
                    sql.Identifier(col), sql.Identifier(col)
                )
                for col in update_columns
            )
        )
    else:
        action = sql.SQL("DO NOTHING")

    statement = sql.SQL("""
        INSERT INTO {}.{} ({})
        VALUES %s
        ON CONFLICT ({})
        {}
    """).format(
        sql.Identifier(schema),
        sql.Identifier(table),
        sql.SQL(", ").join(map(sql.Identifier, columns)),
        sql.SQL(", ").join(map(sql.Identifier, conflict_columns)),
        action,
    ).as_string(conn)

    with conn.cursor() as cur:
        execute_values(cur, statement, rows, page_size=1000)

    return len(rows)


def get_latest_count_timestamp(conn, schema):
    """Return the latest timestamp currently stored in counts."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT MAX(timestamp) FROM {schema}.pedestrian_counts")
        return cur.fetchone()[0]


def get_live_counts(live_path):
    """Clean the live snapshot and return complete hourly totals."""
    live = pd.read_csv(live_path)

    # Preserve the existing live-data validation and aggregation logic.
    empty = pd.DataFrame(
        columns=["timestamp", "location_id", "count"]
    )
    result, _ = extend_complete_minutes(empty, live)

    return result


def clean_files(
    db_url,
    counts_path,
    weather_path,
    output=None,
    only_live=False,
    selection=None,
    live_path=None,
):
    """
    Clean source data and upsert it into PostgreSQL.

    output is retained for CLI compatibility, but CSV data files
    are no longer written.
    """
    report = {}
    ids = selected_ids(selection) if selection else None

    conn = get_db_connection(db_url)

    try:
        if only_live:
            if not live_path:
                raise ValueError("--only-live requires --live")

            counts = get_live_counts(live_path)
            latest = get_latest_count_timestamp(conn, SCHEMA)

            if latest is not None:
                # Keep the existing incremental-only behavior:
                # append complete hours after the stored archive.
                begin_hour = (
                    pd.Timestamp(latest).floor("h")
                    + pd.Timedelta(hours=1)
                )
                counts = counts[counts["timestamp"] >= begin_hour]

            inserted = upsert_dataframe(
                conn,
                SCHEMA,
                "pedestrian_counts",
                counts,
                ["location_id", "timestamp", "count"],
                ["location_id", "timestamp"],
            )
            conn.commit()

            report.update({
                "clean_time": str(datetime.now(timezone.utc)),
                "live_hours_processed": inserted,
            })
            return report

        # Clean historical counts.
        counts, report = clean_counts(
            pd.read_csv(counts_path), ids
        )

        # Optionally include complete live hours.
        if live_path:
            live_counts = get_live_counts(live_path)
            counts = pd.concat(
                [counts, live_counts], ignore_index=True
            )
            report["live_hours_processed"] = len(live_counts)

        # Clean weather.
        weather = clean_weather(pd.read_csv(weather_path))

        # Upsert both datasets in the same transaction.
        upsert_dataframe(
            conn,
            SCHEMA,
            "pedestrian_counts",
            counts,
            ["location_id", "timestamp", "count"],
            ["location_id", "timestamp"],
        )

        upsert_dataframe(
            conn,
            SCHEMA,
            "weather",
            weather,
            ["timestamp", *WEATHER],
            ["timestamp"],
        )

        conn.commit()

        report.update({
            "counts_file": str(Path(counts_path).resolve()),
            "weather_file": str(Path(weather_path).resolve()),
            "weather_missing_values": int(
                weather[WEATHER].isna().sum().sum()
            ),
            "unavailable_selected_ids": sorted(
                set(ids or []) - set(counts.location_id.unique())
            ),
            "counts_rows_processed": len(counts),
            "weather_rows_processed": len(weather),
            "database": os.getenv(
                "PGDATABASE", "pedestrian_data"
            ),
        })

        # Keep quality/configuration metadata if desired.
        if output is not None:
            output = Path(output)
            output.mkdir(parents=True, exist_ok=True)
            (output / "quality.json").write_text(
                json.dumps(report, indent=2)
            )

            if selection:
                (output / "sensor_selection.json").write_text(
                    Path(selection).read_text()
                )

            sources = Path(counts_path).parent / "sources.json"
            if sources.exists():
                (output / "sources.json").write_text(
                    sources.read_text()
                )

        return report

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main():
    load_dotenv(ENV_FILE)
    db_url = os.environ.get('DB_URL')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--counts", type=Path,
        default=ROOT / "data/raw/counts.csv",
    )
    parser.add_argument(
        "--weather", type=Path,
        default=ROOT / "data/raw/weather_history.csv",
    )
    parser.add_argument(
        "--selection", type=Path,
        default=ROOT / "config/cbd_sensor_selection.json",
    )
    parser.add_argument("--live", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/cleaned",
        help="Directory for quality/configuration metadata only",
    )
    parser.add_argument("--only-live", action="store_true")
    args = parser.parse_args()

    report = clean_files(
        db_url,
        args.counts,
        args.weather,
        args.output,
        args.only_live,
        args.selection,
        args.live,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()