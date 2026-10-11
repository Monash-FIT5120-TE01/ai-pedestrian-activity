"""Train and evaluate per-sensor models; save backend artifacts only."""
import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
if __package__:
    from .features import FEATURE_NAMES, counts_from_scaled, prepare
    from .load_data import ROOT, WEATHER
    from .models import ActivityNetwork
    from .optimizers import EagerAdam
else:
    from features import FEATURE_NAMES, counts_from_scaled, prepare
    from load_data import ROOT, WEATHER
    from models import ActivityNetwork
    from optimizers import EagerAdam


def evaluate(model, arrays, stats, batch_size=32):
    model.eval()
    parts = []
    with torch.inference_mode():
        for i in range(0, len(arrays[0]), batch_size):
            parts.append(model(torch.from_numpy(arrays[0][i:i+batch_size]),
                               torch.from_numpy(arrays[1][i:i+batch_size])).numpy())
    return counts_from_scaled(np.concatenate(parts), stats)


def score(prediction, truth):
    residual = prediction-truth
    return dict(mae=float(np.abs(residual).mean()), rmse=float(np.sqrt(np.square(residual).mean())))


def train_sensor(y, weather, output, sensor, models=('gru', 'transformer'), history=168,
                 horizon=384, epochs=20, batch_size=32, stride=24, patience=4,
                 seed=42, min_windows=5, optimizer_kind='eager-adam'):
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError('epochs, batch_size and patience must be positive')
    if not models or not set(models).issubset({'gru', 'transformer'}):
        raise ValueError('models must contain gru and/or transformer')
    if optimizer_kind not in {'eager-adam', 'torch-adam'}:
        raise ValueError('optimizer_kind must be eager-adam or torch-adam')
    torch.manual_seed(seed)
    np.random.seed(seed)
    arrays, stats = prepare(y, weather, history, horizon, stride, min_windows)
    training, validation, test = arrays
    validation_truth = counts_from_scaled(validation[2], stats)
    test_truth = counts_from_scaled(test[2], stats)
    results = {'seasonal': dict(validation=score(validation[3], validation_truth), test=score(test[3], test_truth))}
    states, progress, predictions = {}, {}, {'seasonal': (validation[3], test[3])}
    for kind in models:
        torch.manual_seed(seed)
        model = ActivityNetwork(kind, len(FEATURE_NAMES), history, horizon)
        optimizer = (EagerAdam(model.parameters(), lr=.001) if optimizer_kind == 'eager-adam'
                     else torch.optim.Adam(model.parameters(), lr=.001))
        loader = DataLoader(TensorDataset(*(torch.from_numpy(x) for x in training[:3])),
                            batch_size=batch_size, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(seed))
        best, best_state, waiting = float('inf'), None, 0
        progress[kind] = []
        for epoch in range(epochs):
            model.train()
            total, elements = 0., 0
            for hist, future, target in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(model(hist, future), target)
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite training loss')
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                total += loss.item()*target.numel()
                elements += target.numel()
            val_prediction = evaluate(model, validation, stats, batch_size)
            mae = score(val_prediction, validation_truth)['mae']
            progress[kind].append(dict(epoch=epoch+1, training_mse=total/elements, validation_mae=mae))
            print(f"sensor {sensor['id']} {kind} epoch {epoch+1}: validation MAE {mae:.2f}", flush=True)
            if mae < best:
                best, best_state, waiting = mae, copy.deepcopy(model.state_dict()), 0
            else:
                waiting += 1
            if waiting >= patience:
                break
        model.load_state_dict(best_state)
        val_prediction = evaluate(model, validation, stats, batch_size)
        test_prediction = evaluate(model, test, stats, batch_size)
        predictions[kind] = (val_prediction, test_prediction)
        results[kind] = dict(validation=score(val_prediction, validation_truth), test=score(test_prediction, test_truth))
        states[kind] = dict(model_config=model.config, state_dict=best_state)
    winner = min(results, key=lambda name: results[name]['validation']['mae'])
    lead_rows = []
    validation_ranges = {}
    for kind, (val_prediction, test_prediction) in predictions.items():
        validation_ranges[kind] = []
        for start in range(0, horizon, 24):
            validation_ranges[kind].append(float(np.quantile(np.abs(val_prediction[:, start:start+24]-validation_truth[:, start:start+24]), .9)))
            lead_rows.append(dict(model=kind, lead_day=start//24+1, **score(test_prediction[:, start:start+24], test_truth[:, start:start+24])))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stats.update(sensor=sensor, selected_model=winner, trained_at=datetime.now(timezone.utc).isoformat(),
                 seed=seed, optimizer=optimizer_kind, event_features=False, metrics=results, validation_absolute_error_p90_by_lead_day=validation_ranges,
                 weather_evaluation='Historical reanalysis inputs, including future target weather; not a live forecast accuracy estimate',
                 framework=dict(torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__))
    for kind, state in states.items():
        torch.save(dict(format_version=1, sensor_id=sensor['id'], **state), output / f'{kind}.pt')
    (output / 'model.json').write_text(json.dumps(stats, indent=2, allow_nan=False))
    (output / 'training_log.json').write_text(json.dumps(progress, indent=2, allow_nan=False))
    pd.DataFrame([dict(model=kind, validation_mae=value['validation']['mae'],
                       test_mae=value['test']['mae'], test_rmse=value['test']['rmse']) for kind, value in results.items()]).to_csv(output / 'metrics.csv', index=False)
    pd.DataFrame(lead_rows).to_csv(output / 'metrics_by_lead_day.csv', index=False)
    return stats


def train_all(data, output, ids=None, **kwargs):
    data, output = Path(data), Path(output)
    counts = pd.read_csv(data / 'counts.csv')
    counts['timestamp'] = pd.to_datetime(counts.timestamp, utc=True)
    weather = pd.read_csv(data / 'weather.csv')
    weather['timestamp'] = pd.to_datetime(weather.timestamp, utc=True)
    weather = weather.set_index('timestamp')[WEATHER]
    if counts.duplicated(['location_id', 'timestamp']).any() or weather.index.duplicated().any():
        raise ValueError('Run clean_data.py before training: duplicate timestamps.')
    metadata = json.loads((data / 'sensor_selection.json').read_text()) if (data / 'sensor_selection.json').exists() else {}
    sensors = {int(s['id']): s for s in metadata.get('sensors', [])}
    wanted = sorted(set(ids if ids is not None else metadata.get('selectedIds', counts.location_id.unique())))
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for sid in wanted:
        group = counts.loc[counts.location_id.eq(sid)]
        if group.empty:
            summary.append(dict(sensor_id=int(sid), status='skipped', reason='No downloaded counts'))
            continue
        y = group.set_index('timestamp')['count'].sort_index().astype(float)
        sensor = sensors.get(int(sid), dict(id=int(sid), name=f'Sensor {sid}'))
        try:
            stats = train_sensor(y, weather, output / f'sensor_{sid}', sensor, **kwargs)
        except ValueError as error:
            summary.append(dict(sensor_id=int(sid), status='skipped', reason=str(error)))
            print(f'Sensor {sid} skipped: {error}', flush=True)
            continue
        summary.append(dict(sensor_id=int(sid), status='trained', selected_model=stats['selected_model'], windows=stats['windows']))
    (output / 'training_summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/cleaned')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--sensors', help='Comma-separated IDs. Default: every ID in the CBD selection.')
    parser.add_argument('--models', default='gru,transformer')
    parser.add_argument('--history-hours', type=int, default=168)
    parser.add_argument('--horizon-hours', type=int, default=384)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--stride-hours', type=int, default=24)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--optimizer', choices=['eager-adam', 'torch-adam'], default='eager-adam')
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.threads))
    summary = train_all(args.data, args.output,
        ids=[int(s) for s in args.sensors.split(',')] if args.sensors else None,
        models=tuple(args.models.split(',')), history=args.history_hours, horizon=args.horizon_hours,
        epochs=args.epochs, batch_size=args.batch_size, stride=args.stride_hours, optimizer_kind=args.optimizer)
    print(json.dumps(summary, indent=2))
    if not any(row['status'] == 'trained' for row in summary):
        raise SystemExit('No sensor trained. See training_summary.json; do not treat skipped sensors as forecasts.')


if __name__ == '__main__':
    main()
