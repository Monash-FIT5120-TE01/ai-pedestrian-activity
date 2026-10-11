# My City Twin — data loading, cleaning and training

This `backend_training` folder is a standalone Python handoff for the backend team. It now includes saved-model prediction, a JSON API and weather refresh. See [PREDICTION.md](PREDICTION.md) for frontend integration. Nothing here depends on the React repository or the old research notebook.

| File | Purpose |
| --- | --- |
| `load_data.py` | Download City of Melbourne hourly counts and historical weather; optionally download minute readings. |
| `clean_data.py` | Convert Melbourne date/hour records to UTC, validate and deduplicate counts, keep missing hours missing, validate hourly weather. |
| `features.py` | Build chronological training/validation/test windows, calendar/weather features, train-only scalers and the prior-week reference. |
| `models.py` | GRU and Transformer network definitions, reusable by the eventual backend service. |
| `optimizers.py` | Dense eager Adam updates used by default, with no compiler startup dependency. |
| `train.py` | Train one model pair per sensor, evaluate against a seasonal baseline, select by validation MAE and save artifacts. |
| `predict.py` | Load matching saved artifacts and predict from recent counts and delivered weather. |
| `api.py` | Serve single-sensor or all-sensor forecast JSON for frontend fetches. |
| `refresh_weather.py` | Refresh forecast weather separately, preserving BOM/fallback source metadata. |
| `refresh_counts.py` | Keep a persistent observation store and publish real hourly counts without retraining. |
| `observations.py` | Validate counts and report per-sensor freshness/history gaps. |
| `config/cbd_sensor_selection.json` | Existing app's sensor selection snapshot: 46 locations inside its CBD boundary from 99 reporting locations. Refresh this selection when the sensor network/boundary changes. |

## Setup

Use a supported Python/PyTorch combination for your backend environment. Verified locally with Python 3.14.4, NumPy 2.5.3, pandas 3.0.6 and PyTorch 2.14.1. The existing project environment can run these scripts. From this folder:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
. .venv/bin/activate
```

On Windows use `.venv\Scripts\python.exe` instead. All paths below are relative to this folder. Loading, cleaning and training are separate calls so a backend job can reuse saved inputs without downloading them again. You can also copy this package into your backend and import `backend_training.clean_data.clean_files` or `backend_training.train.train_all`; the same functions power the command-line scripts.

## Use the data already downloaded in this project

```sh
python clean_data.py --counts ../data/city_counts_cbd.csv --weather ../data/weather_history.csv
python train.py --sensors 3 --models gru,transformer --horizon-hours 384 --epochs 20
```

Omit `--sensors` to attempt every location in the supplied CBD selection:

```sh
python train.py --models gru,transformer --horizon-hours 384 --epochs 20
```

These commands train models; they do not publish or serve forecasts. A location with too little complete history is recorded as skipped in `artifacts/training_summary.json`. If all locations are skipped, the command fails. Never substitute an unavailable model with zero pedestrian counts. Sensor notes are preserved in model metadata; relocation/device changes need review before combining their history. The included two-year history starts after the recorded 2023 move of sensor 47.

## Refresh source data

```sh
python load_data.py --days 730
python clean_data.py --counts data/raw/counts.csv --weather data/raw/weather_history.csv
python train.py --horizon-hours 384 --epochs 20
```

Optional recent-minute snapshot:

```sh
python load_data.py --days 730 --include-live
python clean_data.py --counts data/raw/counts.csv --weather data/raw/weather_history.csv --live data/raw/live_snapshot.csv
```

Only hours containing 60 distinct valid minute readings are appended, and only if they immediately follow that sensor's archive. The past-hour feed cannot fill an archive-to-live gap. The cleaned hourly grid keeps outages as missing values, and training windows crossing count gaps are skipped. True observed zeroes remain zeroes. Identical duplicate rows are collapsed; conflicting rows fail for review.

The archive date/hour fields cannot disambiguate the repeated autumn hour. Ambiguous and nonexistent Melbourne DST hours are excluded. UTC indexes are used for storage; calendar features use `Australia/Melbourne`. Missing weather is filled with training-period means with explicit missing-value features, not future interpolation.

## Predict using saved models

Training is not required for each prediction. The model weights/scalers are
reused; recent observed counts and forecast weather are refreshed separately.
The default neural input is 168 consecutive hourly totals, with the last hour
ending no more than two hours before a prediction request. A minute snapshot
alone usually does not supply that whole history.

The complete handoff includes `refresh_counts.py` for persistent count
collection and `/api/input-status` for freshness and gap diagnostics. Use
`--include-live` in `load_data.py` to obtain a one-off minute snapshot; this
script has no `--live-date` argument. For ongoing collection and an optional
explicitly labelled seasonal fallback, see the complete handoff's
`backend_training/PREDICTION.md` (or `backend_prediction/README.md` in this
workspace). The training-only ZIP does not include the prediction collector.

## Training and backend artifacts

Defaults: previous 168 hours, next 384 hours (16 days), daily training origins, chronological 70/15/15 split, 20 epochs maximum with validation early stopping. Targets cannot straddle split boundaries. Validation/test history may use earlier observed hours, as a rolling backtest would. Overlapping multi-day targets make error samples correlated. Scalers and quiet/busy thresholds are fitted on training data only. The test period is never used to choose a model.

Both networks learn residual corrections to the same median reference from up to four matching past local weekdays/hours. All baseline observations are dated at or before each training origin, including for horizons longer than one week. The seasonal baseline may win; `selected_model` records that rather than claiming a neural model is necessarily better.

The default optimiser is the prototype's dense eager Adam (learning rate 0.001, betas 0.9/0.999, epsilon 1e-8, no weight decay), checked for bias correction. It keeps normal PyTorch autograd but avoids compiler imports that stall in the existing local environment. Use `--optimizer torch-adam` to select standard `torch.optim.Adam` in your backend environment. This implementation is for these dense networks; it is not a general replacement for all PyTorch optimisers.

Each successful sensor produces:

```text
artifacts/sensor_3/
  gru.pt                 # state_dict and exact model_config
  transformer.pt         # state_dict and exact model_config
  model.json             # scaler, feature order, sensor, split dates, horizons, thresholds, metrics
  training_log.json       # sample-weighted loss and validation MAE per epoch
  metrics.csv            # validation MAE and held-out test MAE/RMSE in movements/hour
  metrics_by_lead_day.csv # test error by lead day
```

The prediction backend loads a checkpoint with `torch.load(..., map_location='cpu', weights_only=True)`, constructs `ActivityNetwork(**checkpoint['model_config'])`, loads its `state_dict` and keeps `model.json` alongside it. History channels, future channels, scaler values and model dimensions are recorded. Validation absolute-error 90th percentiles are descriptive ranges, not guaranteed prediction intervals. `predict.py` and `api.py` provide inference and frontend access; see [PREDICTION.md](PREDICTION.md).

Training covers up to `--horizon-hours`; the network accepts shorter future sequences, and rejects longer ones. Train with a horizon at least as long as the weather coverage you later intend to serve. A backend still needs complete recent sensor inputs and timestamp-aligned forecast weather before it can produce future predictions.

Events are not model inputs. Concert, comedy, festival, AFL and tennis warnings remain separate from model training. Optional source loaders are in ../event_warnings/. Weather training inputs here are Open-Meteo historical reanalysis, not direct BOM observations. Test windows also use reanalysis for future weather; these metrics do not measure deployment accuracy using forecasts issued at the time. For that evaluation, the backend team should supply archived weather forecast runs with issue times. BOM daily summaries cannot be turned into hourly training observations.

Sources: [City of Melbourne hourly counts](https://data.melbourne.vic.gov.au/explore/dataset/pedestrian-counting-system-monthly-counts-per-hour/), [minute readings](https://data.melbourne.vic.gov.au/explore/dataset/pedestrian-counting-system-past-hour-counts-per-minute/), [Open-Meteo historical weather](https://open-meteo.com/en/docs/historical-weather-api).

## Verification

```sh
python -m unittest -v test_pipeline.py
```

Tests include daylight saving, missing/invalid observations, duplicate conflicts, complete-minute aggregation, train/test leakage, long-horizon baseline leakage, and a one-epoch training/checkpoint reload for both model types. These short runs check that training works; use full training and held-out future evaluation before relying on its forecasts.

Verified in this workspace: cleaning processed 766,912 source records across 46 sensors; all 13 checks passed; both models completed a one-epoch real-data Melbourne Central run with a 168-hour history and 384-hour horizon (463/70/94 train/validation/test windows). This was a runtime check, not full model training or a deployment-accuracy evaluation. The temporary test checkpoints are not included in the handoff. All-sensor full training has not been run.
