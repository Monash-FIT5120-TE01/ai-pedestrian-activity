"""Documented weather feeds; preserve source and availability metadata."""
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

VARIABLES = ['temperature_2m', 'precipitation', 'wind_speed_10m']
BOM_XML = 'ftp://ftp.bom.gov.au/anon/gen/fwo/IDV10753.xml'


def download(url, params=None):
    if params:
        url += '?' + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def parse_bom_daily(raw):
    root = ET.fromstring(raw)
    issued = root.findtext('./amoc/issue-time-utc')
    area = next((a for a in root.findall('./forecast/area') if a.get('description') == 'Melbourne'), None)
    if area is None:
        raise ValueError('Melbourne is absent from BOM daily XML.')
    rows = []
    for period in area.findall('forecast-period'):
        values = {c.get('type'): c.text for c in period}
        chance = values.get('probability_of_precipitation')
        rows.append(dict(date=pd.Timestamp(period.get('start-time-local')).date().isoformat(),
                         start=pd.Timestamp(period.get('start-time-utc')).isoformat(),
                         end=pd.Timestamp(period.get('end-time-utc')).isoformat(),
                         bom_min_temp=pd.to_numeric(values.get('air_temperature_minimum'), errors='coerce'),
                         bom_max_temp=pd.to_numeric(values.get('air_temperature_maximum'), errors='coerce'),
                         bom_daily_rain_chance=pd.to_numeric(chance.rstrip('%') if chance else None, errors='coerce'),
                         bom_precis=values.get('precis'), bom_issued_at=issued))
    if not rows:
        raise ValueError('BOM daily XML contains no Melbourne periods.')
    return pd.DataFrame(rows)


def refresh(directory, lat, lon, provider='auto'):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    params = dict(latitude=lat, longitude=lon, hourly=','.join(VARIABLES),
                  timezone='UTC', past_days=7, forecast_days=10, wind_speed_unit='kmh')
    warnings = []
    hourly = None
    source = None
    if provider in ['auto', 'bom']:
        try:
            payload = json.loads(download('https://api.open-meteo.com/v1/bom', params))
            candidate = pd.DataFrame(payload['hourly']).rename(columns={'time': 'timestamp'})
            candidate['timestamp'] = pd.to_datetime(candidate['timestamp'], utc=True)
            candidate = candidate.set_index('timestamp')
            now = pd.Timestamp.now(tz='UTC').ceil('h')
            if candidate.loc[candidate.index >= now, VARIABLES].dropna().empty:
                raise ValueError('BOM ACCESS-G feed returns no complete future values.')
            hourly, source = candidate, 'BOM ACCESS-G via Open-Meteo'
        except Exception as exc:
            if provider == 'bom':
                raise ValueError(f'Hourly BOM weather is unavailable: {exc}. Use --weather-provider auto for an explicit fallback.') from exc
            warnings.append(str(exc))
    if hourly is None:
        params['forecast_days'] = 16
        payload = json.loads(download('https://api.open-meteo.com/v1/forecast', params))
        hourly = pd.DataFrame(payload['hourly']).rename(columns={'time': 'timestamp'})
        hourly['timestamp'] = pd.to_datetime(hourly['timestamp'], utc=True)
        hourly = hourly.set_index('timestamp')
        source = 'Open-Meteo best match (hourly fallback; not official BOM forecasts)'
    # Do not interpolate, forward-fill or extend a forecast past delivered values.
    hourly.to_csv(directory / 'weather_forecast.csv')
    daily = None
    try:
        raw = download(BOM_XML)
        daily = parse_bom_daily(raw)
        (directory / 'bom_daily.xml').write_bytes(raw)
        daily.to_csv(directory / 'bom_daily.csv', index=False)
    except Exception as exc:
        warnings.append(f'BOM daily feed unavailable: {exc}')
        # A failed refresh must not silently reuse a previously saved daily feed.
        (directory / 'bom_daily.csv').unlink(missing_ok=True)
    metadata = dict(hourly_source=source, hourly_endpoint=('https://api.open-meteo.com/v1/bom' if source.startswith('BOM')
                                                         else 'https://api.open-meteo.com/v1/forecast'),
                    fetched_at=pd.Timestamp.now(tz='UTC').isoformat(),
                    complete_weather_end=hourly[VARIABLES].dropna().index.max().isoformat(),
                    bom_daily_source=BOM_XML if daily is not None else None,
                    bom_daily_issued_at=daily['bom_issued_at'].iloc[0] if daily is not None else None,
                    warnings=warnings)
    (directory / 'weather_metadata.json').write_text(json.dumps(metadata, indent=2))
    print('Hourly weather:', source)
    print('Last complete weather hour:', metadata['complete_weather_end'])
    for warning in warnings:
        print('Weather availability:', warning)
    if daily is not None:
        print('BOM daily Melbourne forecast:', len(daily), 'days; issued', metadata['bom_daily_issued_at'])
    return metadata
