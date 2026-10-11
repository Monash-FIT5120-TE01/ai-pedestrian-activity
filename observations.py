"""Validate observed counts and explain readiness without loading model weights."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .clean_data import TZ
else:
    from clean_data import TZ


def utc(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError('Timestamps must include a UTC/offset timezone.')
    return stamp.tz_convert('UTC')


def read_counts(path):
    frame = pd.read_csv(path)
    if not {'timestamp', 'location_id', 'count'}.issubset(frame):
        raise ValueError('Use cleaned counts.csv: timestamp, location_id, count.')
    if not frame.timestamp.astype(str).str.contains(r'(?:Z|[+-]\d{2}:\d{2})$').all():
        raise ValueError('Count timestamps must include a UTC/offset timezone.')
    frame['timestamp'] = pd.to_datetime(frame.timestamp, utc=True, errors='raise')
    sid = pd.to_numeric(frame.location_id, errors='raise')
    if not (sid.gt(0) & sid.mod(1).eq(0)).all():
        raise ValueError('Sensor IDs must be positive integers.')
    frame['location_id'] = sid.astype(int)
    values = pd.to_numeric(frame['count'], errors='raise')
    if ((~np.isnan(values)) & (~np.isfinite(values) | values.lt(0) | values.mod(1).ne(0))).any():
        raise ValueError('Counts must be nonnegative integers or missing.')
    frame['count'] = values
    if not frame.timestamp.eq(frame.timestamp.dt.floor('h')).all():
        raise ValueError('Counts must be hour aligned.')
    if frame.duplicated(['location_id', 'timestamp']).any():
        raise ValueError('Duplicate sensor-hour counts; run cleaning first.')
    return {int(sid): group.set_index('timestamp')['count'].sort_index()
            for sid, group in frame.groupby('location_id')}


def count_readiness(y, now=None, history_hours=168, max_age_hours=2):
    now = utc(now) if now is not None else pd.Timestamp.now(tz='UTC')
    result = dict(counts_ready=False, required_history_hours=int(history_hours),
                  max_sensor_age_hours=max_age_hours, last_observed_at=None,
                  last_observation_end=None, source_age_hours=None,
                  missing_history_hours=int(history_hours), missing_hours=[],
                  consecutive_history_hours=0)
    if y is None:
        return dict(result, code='counts_missing')
    known = y.loc[y.index < now.floor('h')].sort_index()
    complete = known.dropna()
    if complete.empty:
        return dict(result, code='counts_missing')
    end = complete.index[-1] + pd.Timedelta(hours=1)
    index = pd.date_range(end=end-pd.Timedelta(hours=1), periods=history_hours, freq='h')
    valid = np.isfinite(known.reindex(index).to_numpy(dtype=float))
    missing = index[~valid]
    breaks = np.flatnonzero(~valid)
    result.update(last_observed_at=complete.index[-1].tz_convert(TZ).isoformat(),
                  last_observation_end=end.tz_convert(TZ).isoformat(),
                  source_age_hours=round(float((now-end)/pd.Timedelta(hours=1)), 3),
                  missing_history_hours=int((~valid).sum()),
                  missing_hours=[t.tz_convert(TZ).isoformat() for t in missing[:24]],
                  consecutive_history_hours=int(len(valid)-breaks[-1]-1) if len(breaks) else len(valid))
    if now-end > pd.Timedelta(hours=max_age_hours):
        return dict(result, code='counts_stale')
    if missing.size:
        return dict(result, code='history_gap')
    return dict(result, counts_ready=True, code='ready')


def input_status(counts, selection, artifacts, now=None, max_age_hours=2):
    now = utc(now) if now is not None else pd.Timestamp.now(tz='UTC')
    sensors = []
    for sensor in selection['sensors']:
        sid = int(sensor['id'])
        if sid not in selection['selectedIds']:
            continue
        path = Path(artifacts)/f'sensor_{sid}'/'model.json'
        history, model_metadata = 168, 'missing'
        if path.is_file():
            try:
                history = int(json.loads(path.read_text())['history_hours'])
                if history < 1:
                    raise ValueError('Invalid history window.')
                model_metadata = 'present'
            except (ValueError, KeyError, TypeError):
                history, model_metadata = 168, 'invalid'
        sensors.append(dict(sensor=sensor, model_metadata=model_metadata,
                            **count_readiness(counts.get(sid), now, history, max_age_hours)))
    return dict(generated_at=now.isoformat(), timezone=TZ, selected_sensors=len(sensors),
                counts_ready_sensors=sum(s['counts_ready'] for s in sensors), sensors=sensors,
                scope='Count readiness only; forecast weather, seasonal references and model compatibility are checked by /api/forecast.')
