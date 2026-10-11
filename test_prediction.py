"""Prediction/input parity and frontend HTTP contract checks."""
import copy
import hashlib
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__:
    from .api import make_handler
    from .features import FEATURE_NAMES, prepare, seasonal_reference
    from .models import ActivityNetwork
    from .predict import ForecastService, Unavailable, load_artifact, predict_sensor, read_counts
    from .load_data import WEATHER
else:
    from api import make_handler
    from features import FEATURE_NAMES, prepare, seasonal_reference
    from models import ActivityNetwork
    from predict import ForecastService, Unavailable, load_artifact, predict_sensor, read_counts
    from load_data import WEATHER


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.origin = pd.Timestamp('2026-10-05T12:00:00Z')
        self.now = self.origin + pd.Timedelta(minutes=15)
        index = pd.date_range(end=self.origin-pd.Timedelta(hours=1), periods=35*24, freq='h')
        self.y = pd.Series(100+index.hour.astype(float), index=index)
        self.past = pd.DataFrame({WEATHER[0]: 18., WEATHER[1]: 0., WEATHER[2]: 12.}, index=index)
        future_index = pd.date_range(self.origin-pd.Timedelta(hours=24), periods=96, freq='h')
        self.weather = pd.DataFrame({WEATHER[0]: 20., WEATHER[1]: 0., WEATHER[2]: 10.}, index=future_index)
        self.metadata = dict(fetched_at=self.origin.isoformat(), hourly_source='Synthetic test weather')
        self.stats = dict(format_version=1, sensor=dict(id=3, name='Test sensor', lat=-37.81, lon=144.96),
                          feature_names=FEATURE_NAMES, feature_mean=[0.]*len(FEATURE_NAMES),
                          feature_scale=[1.]*len(FEATURE_NAMES), count_log_mean=4., count_log_scale=1.,
                          quiet=80., busy=150., timezone='Australia/Melbourne',
                          history_hours=24, horizon_hours=48, selected_model='seasonal',
                          trained_at='2026-10-01T00:00:00Z')

    def predict(self, **kwargs):
        return predict_sensor(self.y, self.past, self.weather, self.metadata, self.stats, None,
                              now=self.now, **kwargs)

    def assert_unavailable(self, code, **kwargs):
        with self.assertRaises(Unavailable) as caught:
            self.predict(**kwargs)
        self.assertEqual(caught.exception.code, code)

    def test_weather_and_trained_horizon_bounds(self):
        result = self.predict()
        self.assertEqual(result['forecast_hours'], 48)
        self.assertEqual(result['weather_available_hours'], 72)
        self.assertTrue(any('trained horizon' in w for w in result['warnings']))
        self.assertEqual(self.predict(hours=6)['forecast_hours'], 6)
        self.assertEqual(result['hourly'][0]['time'], '2026-10-05T23:00:00+11:00')
        json.dumps(result, allow_nan=False)

    def test_weather_gap_stops_without_skipping(self):
        self.weather.loc[self.origin+pd.Timedelta(hours=5), WEATHER[0]] = np.nan
        self.assertEqual(self.predict()['forecast_hours'], 5)
        self.weather.loc[self.origin, WEATHER[1]] = np.nan
        self.assert_unavailable('weather_gap')

    def test_stale_weather_or_sensor_is_unavailable(self):
        self.metadata['fetched_at'] = (self.now-pd.Timedelta(hours=13)).isoformat()
        self.assert_unavailable('weather_stale')
        self.metadata['fetched_at'] = self.origin.isoformat()
        self.y = self.y.iloc[:-3]
        self.assert_unavailable('counts_stale')

    def test_weather_issue_in_future_is_rejected(self):
        self.metadata['fetched_at'] = (self.now+pd.Timedelta(minutes=1)).isoformat()
        self.assert_unavailable('weather_stale')

    def test_opt_in_fallback_reports_stale_inputs_and_never_runs_neural_model(self):
        self.y = self.y.iloc[:-3]

        class MustNotRun:
            def __call__(self, *args):
                raise AssertionError('Neural inference must not run on stale inputs.')

        result = predict_sensor(self.y, self.past, self.weather, self.metadata, self.stats,
                                MustNotRun(), now=self.now, hours=8, fallback_seasonal=True)
        self.assertEqual(result['model'], 'seasonal_fallback')
        self.assertEqual(result['fallback_reason'], 'counts_stale')
        self.assertFalse(result['neural_model_used'])
        self.assertEqual(result['forecast_origin'], self.origin.tz_convert('Australia/Melbourne').isoformat())
        self.assertEqual(result['source_age_hours'], 3.25)

    def test_fallback_does_not_bypass_weather_or_missing_seasonal_history(self):
        self.y = self.y.iloc[-24:-3]
        self.assert_unavailable('baseline_gap', fallback_seasonal=True)
        self.metadata['fetched_at'] = (self.now-pd.Timedelta(hours=13)).isoformat()
        self.assert_unavailable('weather_stale', fallback_seasonal=True)

    def test_count_gap_not_filled_but_zero_valid(self):
        self.y.iloc[-5] = np.nan
        self.assert_unavailable('history_gap')
        self.y.iloc[-5] = 0
        self.assertEqual(self.predict()['status'], 'available')

    def test_future_and_incomplete_hour_counts_cannot_leak(self):
        expected = self.predict(hours=8)
        extra = pd.Series(999999., index=pd.date_range(self.origin, periods=60, freq='h'))
        self.y = pd.concat([self.y, extra])
        self.assertEqual(self.predict(hours=8), expected)

    def test_missing_seasonal_history_is_unavailable(self):
        self.y = self.y.iloc[-24:]
        self.assert_unavailable('baseline_gap')

    def test_training_and_prediction_features_match(self):
        index = pd.date_range('2025-01-01', periods=100*24, freq='h', tz='UTC')
        y = pd.Series(200+index.hour.astype(float), index=index)
        weather = self.past.reindex(index).fillna(15.)
        arrays, stats = prepare(y, weather, history=24, horizon=12, stride=24, min_windows=1)
        stats.update(sensor=self.stats['sensor'], selected_model='gru')
        last = pd.Timestamp(stats['window_origins'][2][0])
        origin = last+pd.Timedelta(hours=1)

        class Capture:
            def __call__(self, history, future):
                self.history, self.future = history[0].numpy(), future[0].numpy()
                return future[:, :, -1]

        capture = Capture()
        predict_sensor(y, weather, weather, dict(fetched_at=origin.isoformat(), hourly_source='Test'),
                       stats, capture, now=origin, hours=12)
        np.testing.assert_allclose(capture.history, arrays[2][0][0], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(capture.future, arrays[2][1][0], rtol=1e-6, atol=1e-6)

    def write_fixture(self, root, kind='seasonal'):
        for name in ['data', 'weather', 'artifacts/sensor_3']:
            (root/name).mkdir(parents=True, exist_ok=True)
        stats = copy.deepcopy(self.stats)
        stats['selected_model'] = kind
        (root/'artifacts/sensor_3/model.json').write_text(json.dumps(stats))
        if kind != 'seasonal':
            model = ActivityNetwork(kind, len(FEATURE_NAMES), 24, 48)
            torch.save(dict(format_version=1, sensor_id=3, model_config=model.config, state_dict=model.state_dict()),
                       root/f'artifacts/sensor_3/{kind}.pt')
        self.y.rename('count').rename_axis('timestamp').reset_index().assign(location_id=3).to_csv(root/'data/counts.csv', index=False)
        self.past.rename_axis('timestamp').to_csv(root/'data/weather.csv')
        self.weather.rename_axis('timestamp').to_csv(root/'weather/weather_forecast.csv')
        (root/'weather/weather_metadata.json').write_text(json.dumps(self.metadata))
        (root/'selection.json').write_text(json.dumps(dict(selectedIds=[3, 4], sensors=[stats['sensor'], dict(id=4, name='Missing')])) )
        return stats

    def test_both_checkpoints_eval_deterministic_and_round_trip(self):
        expected = seasonal_reference(pd.date_range(self.origin, periods=8, freq='h'), self.y, self.origin-pd.Timedelta(hours=1))
        for kind in ['gru', 'transformer']:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.write_fixture(root, kind)
                stats, model = load_artifact(root/'artifacts', 3)
                self.assertFalse(model.training)
                args = (self.y, self.past, self.weather, self.metadata, stats, model)
                a = predict_sensor(*args, now=self.now, hours=8)
                b = predict_sensor(*args, now=self.now, hours=8)
                self.assertEqual(a, b)
                np.testing.assert_allclose([r['movements_per_hour'] for r in a['hourly']], expected, atol=.01)

    def test_partial_sensor_results_and_input_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_fixture(root)
            service = ForecastService(root/'data', root/'artifacts', root/'weather', root/'selection.json')
            result = service.forecast_all(8, now=self.now)
            self.assertEqual(result['available_sensors'], 1)
            self.assertEqual(result['forecasts'][1]['code'], 'model_missing')
            self.assertEqual(result['forecasts'][1]['hourly'], [])
            self.weather.loc[self.origin, WEATHER[0]] = np.nan
            self.weather.rename_axis('timestamp').to_csv(root/'weather/weather_forecast.csv')
            self.assertEqual(service.forecast(3, now=self.now)['code'], 'weather_gap')

    def test_stale_counts_include_actionable_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.y = self.y.iloc[:-3]
            self.write_fixture(root)
            service = ForecastService(root/'data', root/'artifacts', root/'weather', root/'selection.json')
            result = service.forecast(3, now=self.now)
            self.assertEqual(result['code'], 'counts_stale')
            self.assertEqual(result['inputs']['source_age_hours'], 3.25)
            self.assertIn('not retraining', result['next_step'])
            status = service.observation_status(3, now=self.now)
            self.assertEqual(status['sensors'][0]['required_history_hours'], 24)
            self.assertEqual(status['counts_ready_sensors'], 0)

    def test_mismatched_feature_order_and_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats = self.write_fixture(root, 'gru')
            stats['feature_names'] = list(reversed(FEATURE_NAMES))
            (root/'artifacts/sensor_3/model.json').write_text(json.dumps(stats))
            with self.assertRaises(Unavailable):
                load_artifact(root/'artifacts', 3)
            stats['feature_names'] = FEATURE_NAMES
            stats['horizon_hours'] = 47
            (root/'artifacts/sensor_3/model.json').write_text(json.dumps(stats))
            with self.assertRaises(Unavailable):
                load_artifact(root/'artifacts', 3)

    def test_invalid_cleaned_input_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'counts.csv'
            path.write_text('timestamp,location_id,count\n2026-10-05T01:00:00Z,3,-1\n')
            with self.assertRaises(ValueError):
                read_counts(path)

    def test_weather_refresh_hash_prevents_mixed_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_fixture(root)
            path = root/'weather/weather_forecast.csv'
            self.metadata['forecast_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            (root/'weather/weather_metadata.json').write_text(json.dumps(self.metadata))
            service = ForecastService(root/'data', root/'artifacts', root/'weather', root/'selection.json')
            self.assertEqual(service.forecast(3, now=self.now)['status'], 'available')
            path.write_text(path.read_text()+'\n')
            self.assertEqual(service.forecast(3, now=self.now)['code'], 'inputs_updating')

    def test_http_routes_cors_validation_and_unavailable(self):
        # HTTP uses the real wall clock; synthetic snapshots are shifted to it.
        current = pd.Timestamp.now(tz='UTC')
        shift = current.floor('h')-self.origin
        self.y.index += shift
        self.past.index += shift
        self.weather.index += shift
        self.metadata['fetched_at'] = current.isoformat()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_fixture(root)
            service = ForecastService(root/'data', root/'artifacts', root/'weather', root/'selection.json')
            server = HTTPServer(('127.0.0.1', 0), make_handler(service, ['http://localhost:5174']))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f'http://127.0.0.1:{server.server_port}'
                request = urllib.request.Request(base+'/api/forecast?sensor_id=3&hours=8', headers={'Origin': 'http://localhost:5174'})
                with urllib.request.urlopen(request, timeout=10) as response:
                    self.assertEqual(response.headers['Access-Control-Allow-Origin'], 'http://localhost:5174')
                    self.assertEqual(json.load(response)['forecast_hours'], 8)
                for route, code in [('/api/forecast?sensor_id=4', 503), ('/api/forecast?hours=0', 400),
                                    ('/api/forecast?sensor_id=999', 400), ('/missing', 404)]:
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        urllib.request.urlopen(base+route, timeout=10)
                    self.assertEqual(caught.exception.code, code)
                with urllib.request.urlopen(base+'/api/forecast?hours=8', timeout=10) as response:
                    self.assertEqual(json.load(response)['unavailable_sensors'], 1)
                with urllib.request.urlopen(base+'/api/sensors', timeout=10) as response:
                    self.assertEqual(len(json.load(response)['sensors']), 2)
                with urllib.request.urlopen(base+'/api/input-status?sensor_id=3', timeout=10) as response:
                    self.assertEqual(json.load(response)['counts_ready_sensors'], 1)
                with urllib.request.urlopen(base+'/health', timeout=10) as response:
                    self.assertEqual(json.load(response)['status'], 'ok')
                with urllib.request.urlopen(urllib.request.Request(base+'/health', headers={'Origin': 'http://unknown.example'}), timeout=10) as response:
                    self.assertIsNone(response.headers.get('Access-Control-Allow-Origin'))
                with urllib.request.urlopen(urllib.request.Request(base+'/api/forecast', method='OPTIONS', headers={'Origin': 'http://localhost:5174'}), timeout=10) as response:
                    self.assertEqual(response.headers['Access-Control-Allow-Methods'], 'GET, OPTIONS')
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)


if __name__ == '__main__':
    unittest.main()
