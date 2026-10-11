"""Inference from saved training artifacts. No fitting or network requests (database reads only)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import psycopg2
import torch

if __package__:
    from .clean_data import TZ, clean_weather
    from .features import FEATURE_NAMES, features, seasonal_reference, counts_from_scaled
    from .database.helper import get_db_connection
    from .load_data import ROOT, WEATHER
    from .models import ActivityNetwork
    from .observations import count_readiness, input_status, utc
else:
    from clean_data import TZ, clean_weather
    from features import FEATURE_NAMES, features, seasonal_reference, counts_from_scaled
    from database.helper import get_db_connection
    from load_data import ROOT, WEATHER
    from models import ActivityNetwork
    from observations import count_readiness, input_status, utc

SCHEMA = 'digital_twin_v2'
# seasonal_reference() looks back at most 4 weeks before the forecast origin
# (up to 3 whole weeks of horizon offset is already absorbed by the target's own
# week shift, leaving 4 prior weeks). One extra day covers DST wall-clock shifts.
# The history window can be longer than that, so the query uses the larger of the two.
SEASONAL_LOOKBACK_HOURS = 29 * 24
ENV_FILE = ROOT / '.env'

class Unavailable(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def read_weather(path):
    """Forecast weather still comes from the refreshed CSV snapshot."""
    return clean_weather(pd.read_csv(path)).set_index('timestamp')[WEATHER]


def fetch_counts(conn, sensor_id, now, history_hours):
    """Read only the completed hourly counts one sensor needs, filtered in SQL.

    Returns None if the sensor has no completed non-null observation.
    The window is anchored on the sensor's latest observation (not on `now`), so a stale
    sensor is still reported as counts_stale rather than looking like it has no data.
    The lower bound reaches back far enough for both the model's history window and the
    seasonal reference. The upper bound excludes the current, incomplete hour.
    """
    end = now.floor('h')
    with conn.cursor() as cur:
        cur.execute(f'SELECT max("timestamp") FROM {SCHEMA}.pedestrian_counts '
                    'WHERE location_id = %s AND "count" IS NOT NULL AND "timestamp" < %s',
                    (sensor_id, end.to_pydatetime()))
        latest = cur.fetchone()[0]
        if latest is None:
            return None
        # utc() rejects naive timestamps, so "timestamp" must be timestamptz.
        start = utc(latest) - pd.Timedelta(hours=max(int(history_hours), SEASONAL_LOOKBACK_HOURS))
        cur.execute(f'SELECT "timestamp", "count" FROM {SCHEMA}.pedestrian_counts '
                    'WHERE location_id = %s AND "timestamp" >= %s AND "timestamp" < %s '
                    'ORDER BY "timestamp"',
                    (sensor_id, start.to_pydatetime(), end.to_pydatetime()))
        rows = cur.fetchall()
    frame = pd.DataFrame(rows, columns=['timestamp', 'count'])
    frame['timestamp'] = pd.to_datetime(frame.timestamp, utc=True, errors='raise')
    values = pd.to_numeric(frame['count'], errors='raise').astype(float)  # NULL -> NaN
    # Same checks the CSV reader applied.
    if ((~np.isnan(values)) & (~np.isfinite(values) | values.lt(0) | values.mod(1).ne(0))).any():
        raise ValueError('Counts must be nonnegative integers or missing.')
    if not frame.timestamp.eq(frame.timestamp.dt.floor('h')).all():
        raise ValueError('Counts must be hour aligned.')
    if frame.timestamp.duplicated().any():
        raise ValueError('Duplicate sensor-hour counts in pedestrian_counts.')
    return pd.Series(values.to_numpy(), index=pd.DatetimeIndex(frame.timestamp), name='count').sort_index()


def fetch_history_weather(conn, now, history_hours, max_sensor_age_hours):
    """Read only the historical weather hours the neural model can use, filtered in SQL.

    The neural model only runs when the last count is at most `max_sensor_age_hours` old, so
    its forecast origin is >= now - max_sensor_age_hours and its history window starts no
    earlier than origin - history_hours (+1h margin). Hours at/after the current hour come
    from forecast weather, so the upper bound is the current hour.
    Assumes the weather table already holds cleaned values.
    """
    end = now.floor('h')
    start = end - pd.Timedelta(hours=int(history_hours) + int(max_sensor_age_hours) + 1)
    columns = ', '.join(f'"{c}"' for c in WEATHER)
    with conn.cursor() as cur:
        cur.execute(f'SELECT "timestamp", {columns} FROM {SCHEMA}.weather '
                    'WHERE "timestamp" >= %s AND "timestamp" < %s ORDER BY "timestamp"',
                    (start.to_pydatetime(), end.to_pydatetime()))
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype=float) for c in WEATHER},
                            index=pd.DatetimeIndex([], tz='UTC', name='timestamp'))
    frame = pd.DataFrame(rows, columns=['timestamp', *WEATHER])
    frame['timestamp'] = pd.to_datetime(frame.timestamp, utc=True, errors='raise')
    if frame.timestamp.duplicated().any():
        raise ValueError('Duplicate hours in weather table.')
    # Decimal/NULL -> float/NaN; NaNs become explicit *_missing features downstream.
    frame[WEATHER] = frame[WEATHER].apply(pd.to_numeric, errors='raise').astype(float)
    return frame.set_index('timestamp')[WEATHER]


def model_history_hours(artifacts, sensor_id, default=168):
    """History window from model.json, used only to size the counts query for /input-status."""
    try:
        value = int(json.loads((Path(artifacts)/f'sensor_{sensor_id}'/'model.json').read_text())['history_hours'])
        return value if value >= 1 else default
    except (OSError, ValueError, KeyError, TypeError):
        return default


def load_artifact(directory, sensor_id):
    path = Path(directory) / f'sensor_{sensor_id}'
    if not (path / 'model.json').is_file():
        raise Unavailable('model_missing', 'No trained artifact for this sensor.')
    stats = json.loads((path / 'model.json').read_text())
    if (stats.get('format_version') != 1 or stats.get('sensor', {}).get('id') != sensor_id
            or stats.get('feature_names') != FEATURE_NAMES or stats.get('timezone') != TZ):
        raise Unavailable('model_incompatible', 'Sensor, feature order, timezone or artifact version differs.')
    mean, scale = np.asarray(stats['feature_mean']), np.asarray(stats['feature_scale'])
    numeric = np.r_[mean, scale, stats['count_log_mean'], stats['count_log_scale'], stats['quiet'], stats['busy']]
    if (mean.shape != (len(FEATURE_NAMES),) or scale.shape != mean.shape
            or not np.isfinite(numeric).all() or (scale <= 0).any()
            or stats['count_log_scale'] <= 0 or not 0 <= stats['quiet'] <= stats['busy']
            or min(stats['history_hours'], stats['horizon_hours']) < 1):
        raise Unavailable('model_incompatible', 'Invalid saved scaling, dimensions or activity thresholds.')
    kind = stats.get('selected_model')
    if kind == 'seasonal':
        return stats, None
    if kind not in {'gru', 'transformer'}:
        raise Unavailable('model_incompatible', 'Unknown selected model.')
    if not (path / f'{kind}.pt').is_file():
        raise Unavailable('model_missing', f'Missing selected {kind} checkpoint.')
    saved = torch.load(path / f'{kind}.pt', map_location='cpu', weights_only=True)
    config = saved['model_config']
    if (saved.get('format_version') != 1 or saved.get('sensor_id') != sensor_id
            or config['kind'] != kind or config['nfeatures'] != len(FEATURE_NAMES)
            or config['history_hours'] != stats['history_hours']
            or config['horizon_hours'] != stats['horizon_hours']):
        raise Unavailable('model_incompatible', 'Checkpoint does not match model.json.')
    model = ActivityNetwork(**config)
    model.load_state_dict(saved['state_dict'], strict=True)
    model.eval()
    return stats, model


def predict_sensor(y, historical_weather, forecast_weather, weather_metadata, stats, model,
                   now=None, hours=None, max_sensor_age_hours=2, max_weather_age_hours=12,
                   fallback_seasonal=False):
    """Output stops at the first missing weather hour, within the trained horizon."""
    now = utc(now) if now is not None else pd.Timestamp.now(tz='UTC')
    issued = utc(weather_metadata['fetched_at'])
    if issued > now or now - issued > pd.Timedelta(hours=max_weather_age_hours):
        raise Unavailable('weather_stale', 'Refresh forecast weather before requesting predictions.')
    # The current hour is incomplete. Never use its count or a future count as input.
    known = y.loc[y.index < now.floor('h')]
    complete = known.dropna()
    if complete.empty:
        raise Unavailable('counts_missing', 'No completed hourly observations for this sensor.')
    observation_end = complete.index[-1] + pd.Timedelta(hours=1)
    origin = observation_end
    readiness = count_readiness(y, now, stats['history_hours'], max_sensor_age_hours)
    fallback_reason = readiness['code'] if not readiness['counts_ready'] else None
    if fallback_reason and not fallback_seasonal:
        message = ('Recent complete sensor hours are required.' if fallback_reason == 'counts_stale'
                   else 'The model needs a complete recent count window; gaps are not filled.')
        raise Unavailable(fallback_reason, message)
    if fallback_reason:
        origin = now.floor('h')
    history_index = pd.date_range(end=origin-pd.Timedelta(hours=1),
                                  periods=stats['history_hours'], freq='h')
    history_count = known.reindex(history_index).to_numpy(dtype=float)
    if not fallback_reason and not np.isfinite(history_count).all():
        raise Unavailable('history_gap', 'The model needs a complete recent count window; gaps are not filled.')
    available = forecast_weather.loc[forecast_weather.index >= origin, WEATHER]
    if available.empty:
        raise Unavailable('weather_missing', 'No forecast weather from the prediction origin.')
    index = pd.date_range(origin, available.index.max(), freq='h')
    weather = forecast_weather.reindex(index)[WEATHER]
    valid = np.isfinite(weather.to_numpy()).all(axis=1)
    length = int(np.flatnonzero(~valid)[0]) if not valid.all() else len(index)
    if not length:
        raise Unavailable('weather_gap', 'Forecast weather is missing at the prediction origin.')
    if hours is not None and (isinstance(hours, bool) or not isinstance(hours, int) or hours < 1):
        raise ValueError('hours must be a positive integer.')
    requested = length if hours is None else min(hours, length)
    horizon = int(stats['horizon_hours'])
    served = min(requested, horizon)
    index, weather = index[:served], weather.iloc[:served]
    baseline = seasonal_reference(index, known, origin-pd.Timedelta(hours=1))
    if not np.isfinite(baseline).all():
        raise Unavailable('baseline_gap', 'Matching observed weekday/hour history is unavailable.')
    # Fresh delivered weather replaces older history where it overlaps. Missing
    # history weather uses the saved training means and explicit missing flags.
    if model is None or fallback_reason:
        counts = baseline
    else:
        history_weather = forecast_weather.combine_first(historical_weather)
        raw = features(history_index.append(index), history_weather)
        mean, scale = np.asarray(stats['feature_mean']), np.asarray(stats['feature_scale'])
        scaled = (np.where(np.isnan(raw), mean, raw)-mean)/scale
        ym, ys = stats['count_log_mean'], stats['count_log_scale']
        hist = np.column_stack([(np.log1p(history_count)-ym)/ys, scaled[:len(history_index)]]).astype(np.float32)
        future = np.column_stack([scaled[len(history_index):], (np.log1p(baseline)-ym)/ys]).astype(np.float32)
        with torch.inference_mode():
            output = model(torch.from_numpy(hist[None]), torch.from_numpy(future[None]))[0].numpy()
        if not np.isfinite(output).all():
            raise Unavailable('model_output_invalid', 'Model produced nonfinite values.')
        counts = counts_from_scaled(output, stats)
    warnings = list(weather_metadata.get('warnings', []))
    if fallback_reason:
        warnings.append('Seasonal fallback uses observed prior weekdays/hours. Recent counts are incomplete '
                        'or stale; the trained neural model was not run. Weather and events do not adjust fallback counts.')
    if length < len(valid):
        warnings.append('Forecast stops before a missing weather hour.')
    if requested > horizon:
        warnings.append('Weather extends beyond the trained horizon; retrain with a longer horizon to serve those hours.')
    if hours is not None and hours > length:
        warnings.append('Requested hours extend beyond available consecutive weather.')
    rows = []
    for stamp, count, values in zip(index, counts, weather.to_numpy()):
        activity = 'quiet' if count <= stats['quiet'] else 'busy' if count >= stats['busy'] else 'moderate'
        rows.append(dict(time=stamp.tz_convert(TZ).isoformat(), movements_per_hour=round(float(count), 2),
                         activity=activity, weather=dict(zip(WEATHER, map(float, values)))))
    return dict(status='available', sensor=stats['sensor'],
                model='seasonal_fallback' if fallback_reason else stats['selected_model'],
                selected_model=stats['selected_model'], neural_model_used=model is not None and not bool(fallback_reason),
                seasonal_fallback=bool(fallback_reason), fallback_reason=fallback_reason,
                source_age_hours=round(float((now-observation_end)/pd.Timedelta(hours=1)), 3),
                trained_at=stats.get('trained_at'), timezone=TZ,
                last_observed_at=complete.index[-1].tz_convert(TZ).isoformat(),
                forecast_origin=origin.tz_convert(TZ).isoformat(),
                weather_source=weather_metadata['hourly_source'], weather_fetched_at=issued.isoformat(),
                weather_available_hours=length, trained_horizon_hours=horizon, forecast_hours=served,
                activity_thresholds=dict(quiet_max=stats['quiet'], busy_min=stats['busy']),
                events_used_in_model=False, warnings=warnings, hourly=rows)


class ForecastService:
    """Reload forecast weather/checkpoints when their file signatures change.

    Counts and historical weather are read from PostgreSQL on every request (they change
    continuously, so there is no file signature to cache on). Each query is filtered in SQL
    to the window that sensor needs.
    """
    def __init__(self, artifacts=ROOT / 'artifacts', weather=ROOT / 'data/forecast',
                 selection=ROOT / 'config/cbd_sensor_selection.json',
                 max_sensor_age_hours=2, max_weather_age_hours=12, fallback_seasonal=False,
                 db_url=None):
        self.artifacts, self.weather = map(Path, (artifacts, weather))
        selection = json.loads(Path(selection).read_text())
        self.sensors = {int(s['id']): s for s in selection['sensors'] if s['id'] in selection['selectedIds']}
        self.max_sensor_age_hours = max_sensor_age_hours
        self.max_weather_age_hours = max_weather_age_hours
        self.fallback_seasonal = fallback_seasonal
        # None -> libpq falls back to the standard PG* environment variables.
        self.db_url = db_url
        self._signature, self._models = None, {}

    def _connect(self):
        conn = get_db_connection(self.db_url)
        conn.set_session(readonly=True)  # inference never writes
        return conn

    def observation_status(self, sensor_id=None, now=None):
        if sensor_id is not None and sensor_id not in self.sensors:
            raise ValueError('Sensor is outside the configured CBD selection.')
        now = utc(now) if now is not None else pd.Timestamp.now(tz='UTC')
        ids = [sensor_id] if sensor_id is not None else list(self.sensors)
        counts = {}
        conn = self._connect()
        try:
            for sid in ids:
                y = fetch_counts(conn, sid, now, model_history_hours(self.artifacts, sid))
                if y is not None:
                    counts[sid] = y
        finally:
            conn.close()
        selection = dict(selectedIds=ids, sensors=list(self.sensors.values()))
        return input_status(counts, selection, self.artifacts, now, self.max_sensor_age_hours)

    @staticmethod
    def signature(paths):
        return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths)

    def reload_inputs(self):
        """Forecast weather snapshot only; counts/history weather are queried per request."""
        paths = [self.weather/'weather_forecast.csv', self.weather/'weather_metadata.json']
        try:
            signature = self.signature(paths)
        except FileNotFoundError as exc:
            raise Unavailable('inputs_missing', 'Supply refreshed forecast weather.') from exc
        if signature != self._signature:
            metadata = json.loads(paths[1].read_text())
            if not metadata.get('hourly_source') or not metadata.get('fetched_at'):
                raise ValueError('Weather metadata needs hourly_source and fetched_at.')
            expected_hash = metadata.get('forecast_sha256')
            if expected_hash and hashlib.sha256(paths[0].read_bytes()).hexdigest() != expected_hash:
                raise Unavailable('inputs_updating', 'Weather refresh is in progress; retry shortly.')
            future = read_weather(paths[0])
            if self.signature(paths) != signature:
                raise Unavailable('inputs_updating', 'Input snapshots changed during loading; retry shortly.')
            self._future, self._metadata = future, metadata
            self._signature = signature

    def forecast(self, sensor_id, hours=None, now=None, conn=None):
        """`conn` lets forecast_all share one connection; if omitted one is opened and closed here."""
        if sensor_id not in self.sensors:
            raise ValueError('Sensor is outside the configured CBD selection.')
        now = utc(now) if now is not None else pd.Timestamp.now(tz='UTC')
        owns_conn = conn is None
        y, history_hours = None, 168  # defaults for the readiness report if we fail before loading either
        try:
            self.reload_inputs()
            paths = [self.artifacts/f'sensor_{sensor_id}'/'model.json']
            paths.extend(sorted(paths[0].parent.glob('*.pt')))
            try:
                signature = self.signature(paths)
            except FileNotFoundError:
                raise Unavailable('model_missing', 'No trained artifact for this sensor.')
            if sensor_id not in self._models or self._models[sensor_id][0] != signature:
                try:
                    artifact = load_artifact(self.artifacts, sensor_id)
                except Unavailable:
                    raise
                except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
                    raise Unavailable('model_incompatible', 'Cannot load this sensor artifact; check backend logs/files.') from exc
                self._models[sensor_id] = (signature, *artifact)
            _, stats, model = self._models[sensor_id]
            history_hours = stats['history_hours']
            # Query only this sensor's counts and the history weather window it needs.
            try:
                if owns_conn:
                    conn = self._connect()
                y = fetch_counts(conn, sensor_id, now, history_hours)
                past = fetch_history_weather(conn, now, history_hours, self.max_sensor_age_hours)
            except psycopg2.Error as exc:
                raise Unavailable('database_unavailable', 'Cannot read counts/weather from the database.') from exc
            if y is None:
                raise Unavailable('counts_missing', 'No observations for this sensor.')
            return predict_sensor(y, past, self._future, self._metadata, stats, model, now, hours,
                                  self.max_sensor_age_hours, self.max_weather_age_hours, self.fallback_seasonal)
        except Unavailable as exc:
            result = dict(status='unavailable', sensor=self.sensors[sensor_id], code=exc.code,
                          reason=str(exc), hourly=[])
            if exc.code in {'counts_missing', 'counts_stale', 'history_gap'}:
                result['inputs'] = count_readiness(y, now, history_hours, self.max_sensor_age_hours)
                result['next_step'] = ('Check /api/input-status; missing observations require ongoing collection '
                                       'or a complete hourly feed into digital_twin_v2.pedestrian_counts, '
                                       'not retraining.')
            return result
        finally:
            if owns_conn and conn is not None:
                conn.close()

    def forecast_all(self, hours=None, now=None):
        now = utc(now) if now is not None else pd.Timestamp.now(tz='UTC')
        try:
            conn = self._connect()  # one connection shared across all sensors
        except psycopg2.Error:
            conn = None  # each forecast() retries and reports database_unavailable itself
        try:
            results = [self.forecast(sid, hours, now, conn) for sid in sorted(self.sensors)]
        finally:
            if conn is not None:
                conn.close()
        available = sum(r['status'] == 'available' for r in results)
        return dict(schema_version=1, generated_at=now.isoformat(), timezone=TZ,
                    selected_sensors=len(results), available_sensors=available,
                    unavailable_sensors=len(results)-available, forecasts=results)


def add_paths(parser):
    parser.add_argument('--artifacts', type=Path, default=ROOT/'artifacts')
    parser.add_argument('--weather', type=Path, default=ROOT/'data/forecast')
    parser.add_argument('--selection', type=Path, default=ROOT/'config/cbd_sensor_selection.json')
    parser.add_argument('--fallback-seasonal', action='store_true',
                        help='Explicitly allow labelled prior-week estimates when recent counts are stale or incomplete.')


def main():
    load_dotenv(ENV_FILE)
    db_url = os.environ.get('DB_URL')
    
    parser = argparse.ArgumentParser(description=__doc__)
    add_paths(parser)
    parser.add_argument('--sensor', type=int)
    parser.add_argument('--hours', type=int)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    service = ForecastService(args.artifacts, args.weather, args.selection,
                              fallback_seasonal=args.fallback_seasonal, db_url=db_url)
    result = service.forecast(args.sensor, args.hours) if args.sensor else service.forecast_all(args.hours)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()