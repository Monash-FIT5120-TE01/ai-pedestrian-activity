# Prediction and frontend integration

`predict.py` loads the selected saved model and its training scalers. `api.py`
exposes forecasts as JSON. Neither trains models nor downloads data during a
frontend request. The standalone `backend_prediction` ZIP contains these files
and their shared dependencies, without training code or frontend assets.

## Start

From this folder, using the same environment as training:

```sh
python -m pip install -r requirements.txt
python refresh_counts.py --data data/cleaned --artifacts artifacts
python refresh_weather.py --output data/forecast --weather-provider auto
python api.py --data data/cleaned --artifacts artifacts --weather data/forecast
```

If using the separate prediction folder alongside the earlier training folder:

```sh
python refresh_counts.py --data ../backend_training/data/cleaned --artifacts ../backend_training/artifacts
python api.py --data ../backend_training/data/cleaned --artifacts ../backend_training/artifacts --weather data/forecast
```

Run the collector in another terminal to retain successive snapshots:

```sh
python refresh_counts.py --watch --interval 600 --data data/cleaned --artifacts artifacts
```

Use the same `--data`, `--selection` and `--artifacts` paths as the API. In the
separate-folder setup, use `--data ../backend_training/data/cleaned` and
`--artifacts ../backend_training/artifacts` for the collector too. Keep its
SQLite store (`data/live/observations.sqlite`) between runs. It seeds itself from
the existing cleaned counts, backfills the recent hourly archive, stores minute
observations and atomically publishes `counts.csv` plus `counts_status.json`.
It never modifies checkpoints or runs training. Refresh forecast weather
separately before it becomes 12 hours old. Restart the API after installing this
code update; subsequent count/weather file updates reload automatically.

Check `GET /api/input-status?sensor_id=3` or `GET /api/input-status` for the latest
observation end, its age, required history, consecutive observed hours and gaps.
This endpoint reports count readiness, not model or weather validity.

### Training, input refresh and prediction are different steps

Run `train.py` to create saved weights/scalers; it is not needed each time you
want a forecast. Run `refresh_counts.py` and `refresh_weather.py` to update the
observations and future weather; run `api.py` to serve predictions with those
saved artifacts. The model's `model.json` sets `history_hours` (normally 168).
Live neural inference needs that many consecutive observed hourly totals and
the last hour must end no more than two hours before the request. A snapshot is
enough only if it contains that complete recent window.

### Why `--include-live` alone may not fix `counts_stale`

`load_data.py --days 730` downloads the historical archive and historical
weather. To include a one-off minute snapshot when preparing training data:

```sh
python load_data.py --days 730 --include-live
python clean_data.py --counts data/raw/counts.csv --weather data/raw/weather_history.csv --live data/raw/live_snapshot.csv
```

There is no `--live-date` argument. A refresh cannot recover observations missing
from the source. The one-off cleaner only appends immediately consecutive
complete live hours; the collector retains later observed hours as well and
leaves the intervening gap missing. Neither invents zeroes, interpolates missing
hours nor changes timestamps to make old observations look current.

The official minute feed can omit minutes with no pedestrian detections and
does not guarantee that every sensor has complete readings. Without a source
coverage/health signal, absent readings cannot safely be distinguished from
outages. The collector therefore requires 60 distinct valid minute readings for
a minute-derived hour; valid official hourly totals, including true zeroes, are
also accepted. Ongoing collection alone cannot guarantee a usable 168-hour
window for every sensor. A complete recent hourly source is needed for those
that remain incomplete.

### Optional estimates while live history is incomplete

Strict behavior is the default: incomplete/stale counts remain unavailable.
To explicitly permit observed prior-week estimates while collecting data:

```sh
python api.py --data data/cleaned --artifacts artifacts --weather data/forecast --fallback-seasonal
```

Fallback responses contain `model: "seasonal_fallback"`, `seasonal_fallback:
true`, `neural_model_used: false`, `fallback_reason`, `source_age_hours`, the actual
last observation time and warnings. Counts come from the median of up to four
observed matching local weekdays/hours. They are not GRU/Transformer results;
weather and events do not adjust these fallback counts. Fresh forecast weather
and sufficient matching observed historical counts are still required. Once
valid recent history is available, the service uses the selected saved model
again. The frontend should visibly label fallback results as seasonal estimates.

Required inputs (not included in the code ZIP):

```text
data/cleaned/counts.csv                 timestamp,location_id,count
data/cleaned/weather.csv                timestamp,temperature_2m,precipitation,wind_speed_10m
data/forecast/weather_forecast.csv      same weather columns
data/forecast/weather_metadata.json     hourly_source,fetched_at (UTC/offset timestamp)
artifacts/sensor_<id>/model.json         exact training metadata
artifacts/sensor_<id>/<selected_model>.pt
```

Keep each checkpoint with its matching `model.json`. The training job creates
them. A selected `seasonal` model needs metadata but no checkpoint. The bundled
selection has 46 CBD sensor IDs; an untrained sensor returns unavailable. The
ZIP does not contain trained weights or downloaded observations.
Each selected sensor needs its own matching trained artifact before serving forecasts.

## Frontend requests

| Request | Response |
| --- | --- |
| `GET /api/forecast` | All selected CBD sensors; available/unavailable counts. |
| `GET /api/forecast?sensor_id=3` | One sensor, as far as consecutive weather and the trained horizon allow. |
| `GET /api/forecast?sensor_id=3&hours=24` | At most 24 hourly predictions. |
| `GET /api/sensors` | Selected sensor IDs, names and coordinates. |
| `GET /api/input-status` | Count freshness and history completeness per selected sensor. |
| `GET /api/input-status?sensor_id=3` | Count readiness using this sensor's saved history length. |
| `GET /health` | Server health; does not assert forecast/data availability. |

```js
const response = await fetch('http://127.0.0.1:8000/api/forecast?sensor_id=3');
const forecast = await response.json();
if (forecast.status === 'available') {
  // Use forecast.sensor.lat/lon for the map.
  // Use forecast.hourly for the selected day/hour.
  renderForecast(forecast);
} else {
  showUnavailable(forecast.reason);
}
```

Available single-sensor response shape (example, not a real prediction):

```json
{
  "status": "available",
  "sensor": {"id": 3, "name": "Melbourne Central", "lat": -37.8106, "lon": 144.9643},
  "model": "transformer",
  "timezone": "Australia/Melbourne",
  "last_observed_at": "2026-10-05T11:00:00+11:00",
  "forecast_origin": "2026-10-05T12:00:00+11:00",
  "weather_source": "Delivered hourly weather source",
  "forecast_hours": 24,
  "trained_horizon_hours": 384,
  "events_used_in_model": false,
  "warnings": [],
  "hourly": [{
    "time": "2026-10-05T12:00:00+11:00",
    "movements_per_hour": 225.4,
    "activity": "moderate",
    "weather": {"temperature_2m": 18.5, "precipitation": 0.0, "wind_speed_10m": 12.0}
  }]
}
```

There are also `trained_at`, `weather_fetched_at`, `weather_available_hours` and
`activity_thresholds`. Quiet/busy thresholds are sensor-specific training
quantiles, not a common pedestrian count cutoff. Predicted movements are
expected counts; they can be fractional.

Single-sensor unavailability is HTTP 503 with `status: "unavailable"`, `code`,
`reason`, `sensor` and `hourly: []`. All-sensor requests return HTTP 200 with
`forecasts: [...]`, including those unavailable entries. Never render an empty
forecast as zero traffic. Invalid queries return 400, unknown routes 404, and
unexpected failures 500. An all-sensor response reports `schema_version: 1`,
`generated_at`, `timezone`, `selected_sensors`, `available_sensors` and
`unavailable_sensors`. Origins may differ by sensor; match records by their
actual `time` for a map at one selected hour.

Local CORS origins default to `http://localhost:5174` and
`http://127.0.0.1:5174`. Add the actual frontend origin explicitly:

```sh
python api.py --origin https://your-frontend.example --port 8000
```

Repeated `--origin` flags permit multiple origins. This adapter is a small
standard-library HTTP server for integration. An existing backend can instead
import `ForecastService`, call `forecast(sensor_id, hours)` or
`forecast_all(hours)`, and expose those functions through its own web framework.
Use an HTTPS API URL when connecting a hosted HTTPS frontend.

## Refresh and forecast limits

Keep data refresh, training and prediction separate. Periodically update cleaned
observations and forecast weather; retrain on a schedule you choose, then replace
the matching artifact files together. Inputs and checkpoints are cached and
reloaded after file changes. Publish refreshed CSVs using a temporary file and
rename, so requests cannot see half-written files. Weather refresh publishes a
CSV hash with metadata; mixed weather snapshots return `inputs_updating`.

Inference requires the same complete recent count window used for training
(default 168 hours). Zero is valid; missing counts are not filled. The newest
completed observation must end within two hours of the request. Weather must
have been fetched within 12 hours and must not have a future issue timestamp.
The current incomplete hour and any future observations are excluded. A source
archive that lags by days plus only a past-hour minute feed cannot satisfy this
requirement. Use the persistent collector or another complete hourly feed;
collection may still leave a sparse sensor unavailable. Missing historical weather uses the training means and missing
flags. Future weather must be complete and consecutive.

The default horizon is every consecutive delivered weather hour, capped by the
saved model horizon. A gap stops output before that hour. If weather extends
beyond training, output includes a warning; retrain with a longer horizon to
serve those additional hours. No weather or count interpolation is used.

`refresh_weather.py --weather-provider auto` attempts BOM ACCESS-G through
Open-Meteo and falls back to the Open-Meteo forecast endpoint if needed. The
actual provider and any failure warnings are saved and returned. `--weather-provider
bom` fails if hourly BOM data is unavailable. `--weather-provider open-meteo`
selects that provider explicitly. Temperature is Celsius, rain is mm/hour and
wind is km/hour. Official BOM daily XML is saved as optional context; its daily
rain chance is not converted to hourly model inputs. Current feed availability
is determined by each refresh, not assumed. See [Open-Meteo weather API](https://open-meteo.com/en/docs)
and [BOM Melbourne daily XML](ftp://ftp.bom.gov.au/anon/gen/fwo/IDV10753.xml).

Events remain separate warning data in `event_warnings/`; they do not change
these model predictions. Neural prediction does not assert a measured or
guaranteed accuracy. Test metrics from the training artifacts use historical
weather reanalysis, not archived forecasts issued at the time.

## Export a JSON file instead of running an API

```sh
python predict.py --sensor 3 --output output/sensor-3.json
python predict.py --output output/cbd-forecasts.json
```

Use the same `--data`, `--artifacts`, `--weather` and `--selection` options as
`api.py`. A backend job can publish these JSON files for frontend fetches.

## Checks

```sh
python -m unittest -v test_prediction.py test_observations.py
```

Tests cover training/inference feature parity, both neural checkpoint types and
deterministic evaluation, horizon/weather gaps, stale inputs, future-count
exclusion, genuine zero counts, missing seasonal history, model incompatibility,
partial sensor results, cache refresh, snapshot consistency, HTTP status codes,
query validation and CORS. These checks verify inference behavior, not accuracy.
