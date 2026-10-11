"""Refresh hourly weather separately from frontend prediction requests."""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

if __package__:
    from .load_data import ROOT
    from .weather_sources import refresh
else:
    from load_data import ROOT
    from weather_sources import refresh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data/forecast')
    parser.add_argument('--latitude', type=float, default=-37.8136)
    parser.add_argument('--longitude', type=float, default=144.9631)
    parser.add_argument('--weather-provider', choices=['auto', 'bom', 'open-meteo'], default='auto')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output) as temporary:
        staged = Path(temporary)
        metadata = refresh(staged, args.latitude, args.longitude, args.weather_provider)
        metadata['forecast_sha256'] = hashlib.sha256((staged/'weather_forecast.csv').read_bytes()).hexdigest()
        (staged/'weather_metadata.json').write_text(json.dumps(metadata, indent=2))
        for name in ['weather_forecast.csv', 'bom_daily.csv', 'bom_daily.xml']:
            source = staged/name
            if source.exists():
                source.replace(args.output/name)
            else:
                (args.output/name).unlink(missing_ok=True)
        # Publish metadata last; the reader checks the CSV hash during updates.
        (staged/'weather_metadata.json').replace(args.output/'weather_metadata.json')


if __name__ == '__main__':
    main()
