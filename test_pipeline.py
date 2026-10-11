"""Checks for data loss, ambiguous time, leakage and runnable training artifacts."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from clean_data import clean_counts, clean_weather, extend_complete_minutes
from features import FEATURE_NAMES, prepare, seasonal_reference
from load_data import WEATHER, get, selected_ids


class DataChecks(unittest.TestCase):
    def test_missing_is_not_zero_and_identical_duplicates_are_collapsed(self):
        frame = pd.DataFrame(dict(location_id=[3]*4, sensing_date=['2026-07-01']*4,
                                  hourday=[0, 0, 2, 3], pedestriancount=[0, 0, -1, 50]))
        cleaned, report = clean_counts(frame)
        self.assertEqual(len(cleaned), 4)
        self.assertEqual(cleaned['count'].iloc[0], 0)
        self.assertTrue(cleaned['count'].iloc[1:3].isna().all())
        self.assertEqual(report['identical_duplicates_removed'], 1)

    def test_conflicting_duplicates_fail(self):
        frame = pd.DataFrame(dict(location_id=[3, 3], sensing_date=['2026-07-01']*2,
                                  hourday=[0, 0], pedestriancount=[1, 2]))
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            clean_counts(frame)

    def test_dst_ambiguous_and_nonexistent_hours_are_excluded(self):
        for date in ['2026-04-05', '2026-10-04']:
            frame = pd.DataFrame(dict(location_id=[3]*3, sensing_date=[date]*3,
                                      hourday=[1, 2, 3], pedestriancount=[10]*3))
            _, report = clean_counts(frame)
            self.assertEqual(report['ambiguous_or_nonexistent_local_hours'], 1)

    def test_invalid_hour_is_not_rolled_into_next_day(self):
        frame = pd.DataFrame(dict(location_id=[3]*3, sensing_date=['2026-07-01']*3,
                                  hourday=[0, 24, 1.5], pedestriancount=[10]*3))
        cleaned, report = clean_counts(frame)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(report['invalid_or_excluded_rows'], 2)

    def test_live_stops_at_a_partial_hour(self):
        start = pd.Timestamp('2026-07-01', tz='UTC')
        counts = pd.DataFrame(dict(location_id=[3], timestamp=[start], count=[5.]))
        minutes = pd.date_range(start+pd.Timedelta(hours=1), periods=180, freq='min').delete(70)
        live = pd.DataFrame(dict(location_id=3, sensing_datetime=minutes.astype(str), total_of_directions=1))
        result, appended = extend_complete_minutes(counts, live)
        self.assertEqual(appended, 1)
        self.assertEqual(result['count'].iloc[-1], 60)

    def test_live_cannot_jump_an_archive_gap(self):
        start = pd.Timestamp('2026-07-01', tz='UTC')
        counts = pd.DataFrame(dict(location_id=[3], timestamp=[start], count=[5.]))
        minutes = pd.date_range(start+pd.Timedelta(hours=3), periods=60, freq='min')
        live = pd.DataFrame(dict(location_id=3, sensing_datetime=minutes.astype(str), total_of_directions=1))
        _, appended = extend_complete_minutes(counts, live)
        self.assertEqual(appended, 0)

    def test_weather_missing_stays_missing_and_offsets_required(self):
        frame = pd.DataFrame(dict(timestamp=['2026-07-01T00:00:00Z', '2026-07-01T02:00:00Z'],
                                  temperature_2m=[12, 14], precipitation=[0, -2], wind_speed_10m=[0, np.inf]))
        result = clean_weather(frame)
        self.assertTrue(result.loc[1, WEATHER].isna().all())
        self.assertTrue(result.loc[2, ['precipitation', 'wind_speed_10m']].isna().all())
        frame.timestamp = ['2026-07-01 00:00', '2026-07-01 02:00']
        with self.assertRaisesRegex(ValueError, 'timezone'):
            clean_weather(frame)

    def test_request_and_selection_are_independent_of_frontend(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'selection.json'
            path.write_text('{"selectedIds": [3, 1, 3]}')
            self.assertEqual(selected_ids(path), [1, 3])
        with patch('load_data.urlopen') as mock:
            mock.return_value.__enter__.return_value.read.return_value = b'{}'
            self.assertEqual(get('https://example.org/records', {'location_id': 3}), b'{}')
            request = mock.call_args.args[0]
            self.assertIn('location_id=3', request.full_url)


class TrainingChecks(unittest.TestCase):
    def test_eager_adam_bias_correction_and_gradient_reset(self):
        import torch
        from optimizers import EagerAdam
        parameter = torch.nn.Parameter(torch.tensor([1., -1.]))
        optimizer = EagerAdam([parameter], lr=.01)
        for _ in range(3):
            parameter.grad = torch.tensor([2., -2.])
            optimizer.step()
        self.assertTrue(torch.allclose(parameter, torch.tensor([.97, -.97]), atol=1e-6))
        optimizer.zero_grad()
        self.assertIsNone(parameter.grad)

    @staticmethod
    def series(days=160):
        index = pd.date_range('2025-01-01', periods=24*days, freq='h', tz='UTC')
        y = pd.Series(100+20*np.sin(np.arange(len(index))/24), index=index)
        weather = pd.DataFrame(0., index=index, columns=WEATHER)
        return y, weather

    def test_scalers_and_training_windows_ignore_test_values(self):
        y, weather = self.series()
        before, stats = prepare(y, weather, horizon=24)
        changed_y, changed_weather = y.copy(), weather.copy()
        changed_y.iloc[int(len(y)*.85):] = 99999
        changed_weather.iloc[int(len(y)*.85):] = 99999
        after, later = prepare(changed_y, changed_weather, horizon=24)
        for key in ['feature_mean', 'feature_scale', 'count_log_mean', 'count_log_scale', 'quiet', 'busy']:
            self.assertEqual(stats[key], later[key])
        for before_array, after_array in zip(before[0], after[0]):
            np.testing.assert_array_equal(before_array, after_array)
        for part, origins in enumerate(stats['window_origins']):
            first_target = pd.DatetimeIndex(origins)+pd.Timedelta(hours=1)
            last_target = first_target+pd.Timedelta(hours=23)
            if part == 0:
                self.assertTrue((last_target <= pd.Timestamp(stats['train_end'])).all())
            if part == 1:
                self.assertTrue((first_target > pd.Timestamp(stats['train_end'])).all())
                self.assertTrue((last_target <= pd.Timestamp(stats['validation_end'])).all())
            if part == 2:
                self.assertTrue((first_target > pd.Timestamp(stats['validation_end'])).all())

    def test_long_horizon_reference_never_reads_future_counts(self):
        y, _ = self.series()
        origin = y.index[24*80]
        target = pd.date_range(origin+pd.Timedelta(hours=1), periods=384, freq='h')
        expected = seasonal_reference(target, y, origin)
        y.loc[y.index > origin] = 99999
        np.testing.assert_array_equal(expected, seasonal_reference(target, y, origin))

    def test_full_horizon_and_missing_windows(self):
        y, weather = self.series()
        arrays, stats = prepare(y, weather)
        self.assertEqual(arrays[0][1].shape[1], 384)
        self.assertEqual(stats['feature_scale'][-1], 1.)
        y.iloc[1000] = np.nan
        _, missing = prepare(y, weather)
        self.assertLess(missing['windows'][0], stats['windows'][0])

    def test_both_models_train_save_and_reload(self):
        import json
        import torch
        from models import ActivityNetwork
        from train import train_sensor
        torch.set_num_threads(2)
        y, weather = self.series(days=60)
        with tempfile.TemporaryDirectory() as tmp:
            stats = train_sensor(y, weather, tmp, dict(id=3, name='Test fixture'),
                                 history=24, horizon=24, epochs=1, stride=24)
            saved = json.loads((Path(tmp)/'model.json').read_text())
            self.assertEqual(saved['feature_names'], FEATURE_NAMES)
            self.assertFalse(stats['event_features'])
            for kind in ['gru', 'transformer']:
                artifact = torch.load(Path(tmp)/f'{kind}.pt', map_location='cpu', weights_only=True)
                model = ActivityNetwork(**artifact['model_config'])
                model.load_state_dict(artifact['state_dict'])
                model.eval()
                with torch.inference_mode():
                    result = model(torch.zeros(1, 24, len(FEATURE_NAMES)+1), torch.zeros(1, 7, len(FEATURE_NAMES)+1))
                self.assertEqual(tuple(result.shape), (1, 7))
                with self.assertRaisesRegex(ValueError, 'horizon'):
                    model(torch.zeros(1, 24, len(FEATURE_NAMES)+1), torch.zeros(1, 25, len(FEATURE_NAMES)+1))


if __name__ == '__main__':
    unittest.main()
