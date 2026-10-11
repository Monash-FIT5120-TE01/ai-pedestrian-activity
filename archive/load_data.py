"""Download training inputs only. No forecast, event feed, API server or UI."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API = 'https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/'
COUNTS = 'pedestrian-counting-system-monthly-counts-per-hour'
MINUTES = 'pedestrian-counting-system-past-hour-counts-per-minute'
WEATHER = ['temperature_2m', 'precipitation', 'wind_speed_10m']
ROOT = Path(__file__).resolve().parent


def get(url, params):
    request = Request(url + '?' + urlencode(params), headers={'User-Agent': 'MyCityTwin-training/1.0'})
    with urlopen(request, timeout=120) as response:
        return response.read()


def selected_ids(selection):
    data = json.loads(Path(selection).read_text())
    ids = sorted(set(int(x) for x in data['selectedIds']))
    if not ids or any(x <= 0 for x in ids):
        raise ValueError('Selection must contain positive selectedIds.')
    return ids


def download(selection, output, days=730, include_live=False):
    """Use the supplied CBD selection, not an assumed network size."""
    import pandas as pd
    if days < 1:
        raise ValueError('days must be positive')
    ids = selected_ids(selection)
    where = f"location_id IN ({','.join(map(str, ids))})"
    latest = json.loads(get(API + COUNTS + '/records', dict(
        where=where, limit=1, order_by='sensing_date DESC,hourday DESC')))['results']
    if not latest:
        raise ValueError('No archive history for the selected sensors.')
    end = pd.Timestamp(latest[0]['sensing_date']).normalize()
    start = end - pd.Timedelta(days=days - 1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    query = f"{where} AND sensing_date >= date'{start.date()}' AND sensing_date <= date'{end.date()}'"
    (output / 'counts.csv').write_bytes(get(API + COUNTS + '/exports/csv', dict(where=query, delimiter=',')))
    # Cover the local-date edges in UTC. One shared CBD grid, not street microclimate.
    weather_start = start.tz_localize('Australia/Melbourne').tz_convert('UTC').date()
    weather_end = (end + pd.Timedelta(days=1)).tz_localize('Australia/Melbourne').tz_convert('UTC').date()
    weather = json.loads(get('https://archive-api.open-meteo.com/v1/archive', dict(
        latitude=-37.8136, longitude=144.9631, start_date=str(weather_start),
        end_date=str(weather_end), hourly=','.join(WEATHER), timezone='UTC', wind_speed_unit='kmh')))
    frame = pd.DataFrame(weather['hourly']).rename(columns={'time': 'timestamp'})
    frame['timestamp'] = pd.to_datetime(frame['timestamp'], utc=True).astype(str)
    frame.to_csv(output / 'weather_history.csv', index=False)
    if include_live:
        (output / 'live_snapshot.csv').write_bytes(get(API + MINUTES + '/exports/csv', dict(where=where, delimiter=',')))
    selection_data = json.loads(Path(selection).read_text())
    (output / 'sensor_selection.json').write_text(json.dumps(selection_data, indent=2))
    metadata = dict(fetched_at=datetime.now(timezone.utc).isoformat(), sensor_ids=ids,
                    counts_source=API + COUNTS, live_included=include_live,
                    archive_start=str(start.date()), archive_end=str(end.date()),
                    weather_source='Open-Meteo historical reanalysis; not direct BOM observations',
                    weather_url='https://open-meteo.com/en/docs/historical-weather-api',
                    weather_spatial_scope='Shared Melbourne CBD grid', weather_units=weather.get('hourly_units', {}))
    (output / 'sources.json').write_text(json.dumps(metadata, indent=2))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, default=ROOT / 'config/cbd_sensor_selection.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/raw')
    parser.add_argument('--days', type=int, default=730)
    parser.add_argument('--include-live', action='store_true')
    args = parser.parse_args()
    print(json.dumps(download(args.selection, args.output, args.days, args.include_live), indent=2))


if __name__ == '__main__':
    main()
