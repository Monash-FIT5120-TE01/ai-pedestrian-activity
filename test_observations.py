"""Persistent collection, gaps, readiness and atomic output checks."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .observations import count_readiness, input_status, read_counts
    from .refresh_counts import ObservationStore, refresh
else:
    from observations import count_readiness, input_status, read_counts
    from refresh_counts import ObservationStore, refresh


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = ObservationStore(self.root/'observations.sqlite')
        self.hour = pd.Timestamp('2026-10-08T02:00:00Z')
        self.now = self.hour+pd.Timedelta(hours=1, minutes=15)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def minutes(self, start=None, periods=60):
        index = pd.date_range(start or self.hour, periods=periods, freq='min')
        return pd.DataFrame(dict(location_id=3, sensing_datetime=index.astype(str), total_of_directions=1))

    def hourly(self, index, values):
        return pd.DataFrame(dict(location_id=3, timestamp=index, count=values))

    def test_overlapping_snapshots_persist_across_restart(self):
        live = self.minutes()
        self.store.save_minutes(live.iloc[:40], [3], self.now)
        self.assertTrue(self.store.counts().empty)
        self.store.close()
        self.store = ObservationStore(self.root/'observations.sqlite')
        self.store.save_minutes(live.iloc[30:], [3], self.now)
        self.store.save_minutes(live, [3], self.now)
        self.assertEqual(self.store.counts()['count'].tolist(), [60])
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM minutes').fetchone()[0], 60)

    def test_archive_gap_stays_missing_but_later_observations_retained(self):
        self.store.save_hours(self.hourly([self.hour-pd.Timedelta(hours=3)], [100]), 1, self.now)
        self.store.save_minutes(self.minutes(), [3], self.now)
        y = self.store.counts()['count']
        self.assertEqual(len(y), 4)
        self.assertEqual(y.iloc[0], 100)
        self.assertTrue(y.iloc[1:3].isna().all())
        self.assertEqual(y.iloc[-1], 60)

    def test_conflicting_minute_invalidates_sum_archive_can_resolve(self):
        live = self.minutes()
        self.store.save_minutes(live, [3], self.now)
        changed = live.iloc[:1].copy()
        changed['total_of_directions'] = 2
        self.store.save_minutes(changed, [3], self.now)
        self.assertTrue(self.store.counts().empty)
        self.store.save_hours(self.hourly([self.hour], [65]), 3, self.now)
        self.store.save_minutes(live, [3], self.now)
        self.assertEqual(self.store.counts()['count'].tolist(), [65])

    def test_missing_invalid_and_current_hour_not_published(self):
        live = self.minutes(periods=120)
        live.loc[5, 'total_of_directions'] = -1
        self.store.save_minutes(live, [3], self.now)
        self.assertTrue(self.store.counts().empty)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM minutes WHERE count IS NULL').fetchone()[0], 1)

    def test_explicit_dst_offsets_preserve_repeated_hour(self):
        first = pd.Timestamp('2026-04-05T02:00:00+11:00')
        last = pd.Timestamp('2026-04-05T02:00:00+10:00')
        a, b = self.minutes(first), self.minutes(last)
        self.store.save_minutes(pd.concat([a, b]), [3], last+pd.Timedelta(hours=2))
        frame = self.store.counts()
        self.assertEqual(frame['count'].tolist(), [60, 60])
        self.assertEqual(frame.timestamp.iloc[1]-frame.timestamp.iloc[0], pd.Timedelta(hours=1))

    def test_refresh_uses_new_observations_without_touching_artifacts(self):
        data, artifacts = self.root/'data', self.root/'artifacts'
        data.mkdir()
        (artifacts/'sensor_3').mkdir(parents=True)
        model = artifacts/'sensor_3/model.json'
        original = '{"history_hours": 4, "saved_weights": "unchanged"}'
        model.write_text(original)
        selection = self.root/'selection.json'
        selection.write_text(json.dumps(dict(selectedIds=[3], sensors=[dict(id=3, name='Test')])))
        # Old history plus a gap: downloaded hourly observations bridge it.
        seed = self.hourly([self.hour-pd.Timedelta(hours=5)], [10])
        seed.to_csv(data/'counts.csv', index=False)
        times = pd.date_range(self.hour-pd.Timedelta(hours=3), periods=3, freq='h').tz_convert('Australia/Melbourne')
        archive = pd.DataFrame(dict(location_id=3, sensing_date=times.strftime('%Y-%m-%d'),
                                    hourday=times.hour, pedestriancount=[20, 21, 22]))

        def fetch(url, params):
            self.assertIn('location_id IN (3)', params['where'])
            frame = self.minutes() if 'past-hour' in url else archive
            return frame.to_csv(index=False).encode()

        report = refresh(self.store, data, selection, artifacts, now=self.now, fetch=fetch)
        self.assertEqual(report['counts_ready_sensors'], 1)
        self.assertEqual(model.read_text(), original)
        y = read_counts(data/'counts.csv')[3]
        self.assertEqual(y.iloc[-1], 60)
        self.assertTrue(np.isnan(y.iloc[1]))
        self.assertEqual(json.loads((data/'counts_status.json').read_text())['download_errors'], [])
        self.assertEqual(list(data.glob('.counts.csv*')), [])

    def test_network_failure_preserves_real_history_and_reports_failure(self):
        data = self.root/'data'
        data.mkdir()
        self.hourly([self.hour-pd.Timedelta(hours=10)], [17]).to_csv(data/'counts.csv', index=False)
        selection = self.root/'selection.json'
        selection.write_text('{"selectedIds":[3],"sensors":[{"id":3}]}')

        def fail(*args):
            raise OSError('Feed offline')

        report = refresh(self.store, data, selection, self.root/'artifacts', now=self.now, fetch=fail)
        self.assertEqual(len(report['download_errors']), 2)
        self.assertEqual(report['sensors'][0]['code'], 'counts_stale')
        self.assertEqual(read_counts(data/'counts.csv')[3].iloc[-1], 17)


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.origin = pd.Timestamp('2026-10-08T03:00:00Z')
        self.now = self.origin+pd.Timedelta(minutes=15)
        self.y = pd.Series(0., index=pd.date_range(end=self.origin-pd.Timedelta(hours=1), periods=168, freq='h'))

    def test_zero_observations_and_exact_freshness_boundary(self):
        self.assertTrue(count_readiness(self.y, self.origin+pd.Timedelta(hours=2))['counts_ready'])
        result = count_readiness(self.y, self.origin+pd.Timedelta(hours=2, seconds=1))
        self.assertEqual(result['code'], 'counts_stale')
        self.assertEqual(result['last_observation_end'], '2026-10-08T14:00:00+11:00')

    def test_gap_and_future_observations_cannot_make_history_ready(self):
        self.y.iloc[-5] = np.nan
        self.y = pd.concat([self.y, pd.Series(999., index=[self.origin])])
        result = count_readiness(self.y, self.now)
        self.assertEqual(result['code'], 'history_gap')
        self.assertEqual(result['missing_history_hours'], 1)
        self.assertEqual(result['consecutive_history_hours'], 4)

    def test_status_uses_saved_history_length_and_handles_unobserved_sensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'sensor_3').mkdir()
            (root/'sensor_3/model.json').write_text('{"history_hours":24}')
            selection = dict(selectedIds=[3,4], sensors=[dict(id=3),dict(id=4)])
            result = input_status({3:self.y.iloc[-24:]}, selection, root, self.now)
            self.assertEqual(result['counts_ready_sensors'], 1)
            self.assertEqual(result['sensors'][0]['required_history_hours'], 24)
            self.assertEqual(result['sensors'][1]['code'], 'counts_missing')


if __name__ == '__main__':
    unittest.main()
