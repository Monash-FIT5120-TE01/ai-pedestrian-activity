"""Training windows, train-only scaling and an observed-week reference."""
import numpy as np
import pandas as pd
if __package__:
    from .clean_data import TZ
    from .load_data import WEATHER
else:
    from clean_data import TZ
    from load_data import WEATHER

FEATURE_NAMES = ['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos',
                 'year_sin', 'year_cos', 'weekend', *WEATHER,
                 *[x + '_missing' for x in WEATHER]]


def features(index, weather):
    local = index.tz_convert(TZ)
    calendar = np.column_stack([np.sin(2*np.pi*local.hour/24), np.cos(2*np.pi*local.hour/24),
                               np.sin(2*np.pi*local.dayofweek/7), np.cos(2*np.pi*local.dayofweek/7),
                               np.sin(2*np.pi*local.dayofyear/365.25), np.cos(2*np.pi*local.dayofyear/365.25),
                               (local.dayofweek >= 5).astype(float)])
    values = weather.reindex(index)[WEATHER].to_numpy(dtype=float)
    return np.column_stack([calendar, values, np.isnan(values).astype(float)]).astype(np.float32)


def seasonal_reference(index, y, origin):
    """Same local weekday/hour across up to four weeks, using only known counts."""
    wall = index.tz_convert(TZ).tz_localize(None)
    origin_wall = origin.tz_convert(TZ).tz_localize(None)
    weeks = np.maximum(1, np.ceil((wall-origin_wall).total_seconds()/(7*86400))).astype(int)
    known = y.loc[y.index <= origin]
    values = []
    for offset in range(4):
        previous = wall - pd.to_timedelta((weeks+offset)*7, unit='D')
        previous = previous.tz_localize(TZ, ambiguous='NaT', nonexistent='NaT').tz_convert('UTC')
        values.append(known.reindex(previous).to_numpy())
    stack = np.asarray(values)
    result = np.full(len(index), np.nan)
    valid = np.isfinite(stack).any(axis=0)
    result[valid] = np.nanmedian(stack[:, valid], axis=0)
    return result


def prepare(y, weather, history=168, horizon=384, stride=24, min_windows=5):
    """Targets never cross a chronological split. Gappy count windows are skipped."""
    if min(history, horizon, stride, min_windows) < 1:
        raise ValueError('Window settings must be positive.')
    if y.empty or y.index.tz is None or y.index.duplicated().any():
        raise ValueError('Counts need a nonempty unique timezone-aware hourly index.')
    expected = pd.date_range(y.index.min(), y.index.max(), freq='h')
    if not y.index.equals(expected):
        raise ValueError('Reindex counts to a complete hourly grid before training.')
    n = len(y)
    a, b = int(n*.70), int(n*.85)
    raw = features(y.index, weather)
    mean, scale = np.zeros(raw.shape[1]), np.ones(raw.shape[1])
    for col in range(raw.shape[1]):
        valid = raw[:a, col][np.isfinite(raw[:a, col])]
        if valid.size:
            mean[col] = valid.mean()
            deviation = float(valid.std())
            scale[col] = deviation if deviation > 1e-6 else 1.
    counts = y.to_numpy(dtype=float)
    if ((~np.isnan(counts)) & (~np.isfinite(counts) | (counts < 0))).any():
        raise ValueError('Counts must be nonnegative or missing; run cleaning first.')
    logy = np.log1p(counts)
    observed = logy[:a][np.isfinite(logy[:a])]
    if not observed.size:
        raise ValueError('No valid training counts.')
    ym, ys = float(observed.mean()), max(float(observed.std()), 1e-6)
    partitions = [[], [], []]
    for i in range(history, n-horizon+1, stride):
        part = 0 if i+horizon <= a else 1 if i >= a and i+horizon <= b else 2 if i >= b else None
        if part is None or not np.isfinite(logy[i-history:i+horizon]).all():
            continue
        target_index = y.index[i:i+horizon]
        baseline = seasonal_reference(target_index, y, y.index[i-1])
        if not np.isfinite(baseline).all():
            continue
        block = raw[i-history:i+horizon]
        scaled = (np.where(np.isnan(block), mean, block)-mean)/scale
        hist = np.column_stack([(logy[i-history:i]-ym)/ys, scaled[:history]])
        future = np.column_stack([scaled[history:], (np.log1p(baseline)-ym)/ys])
        target = (logy[i:i+horizon]-ym)/ys
        partitions[part].append((hist, future, target, baseline, i))
    sizes = [len(p) for p in partitions]
    if min(sizes) < min_windows:
        raise ValueError(f'Too few complete train/validation/test windows: {sizes}; need {min_windows} each.')
    arrays = [tuple(np.asarray([row[j] for row in part], dtype=np.float32) for j in range(4)) for part in partitions]
    stats = dict(format_version=1, feature_names=FEATURE_NAMES, feature_mean=mean.tolist(),
                 feature_scale=scale.tolist(), count_log_mean=ym, count_log_scale=ys,
                 quiet=float(y.iloc[:a].quantile(.33)), busy=float(y.iloc[:a].quantile(.67)),
                 timezone=TZ, train_end=y.index[a-1].isoformat(), validation_end=y.index[b-1].isoformat(),
                 test_end=y.index[-1].isoformat(), history_hours=history, horizon_hours=horizon,
                 stride_hours=stride, windows=sizes,
                 window_origins=[[y.index[row[4]-1].isoformat() for row in part] for part in partitions],
                 baseline_method='Median of up to four matching observed prior local weekdays/hours',
                 history_channels=['scaled_log_count', *FEATURE_NAMES],
                 future_channels=[*FEATURE_NAMES, 'scaled_log_seasonal_reference'])
    return arrays, stats


def counts_from_scaled(values, stats):
    return np.maximum(0, np.expm1(np.clip(values*stats['count_log_scale']+stats['count_log_mean'], -20, 20)))
