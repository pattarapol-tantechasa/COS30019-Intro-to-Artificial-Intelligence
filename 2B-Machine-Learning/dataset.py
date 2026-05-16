import numpy as np


def normalise(values):
    """
    Min-max normalise to [0, 1]. Returns the scaled array plus (min, max)
    so we can invert predictions back to vehicle counts later.
    """
    vmin, vmax = float(values.min()), float(values.max())
    if vmax == vmin:
        # Edge case — a site that recorded zero traffic (or constant) all month.
        return np.zeros_like(values, dtype=float), (vmin, vmax)
    scaled = (values - vmin) / (vmax - vmin)
    return scaled.astype(float), (vmin, vmax)


def make_windows(series, window, horizon):
    """
    Sliding window. For an input series s of length N, build:
      X[i] = s[i : i+window]
      y[i] = s[i + window + horizon - 1]

    horizon=1 means y is the very next reading after the window.
    """
    X, y = [], []
    n = len(series)
    for i in range(n - window - horizon + 1):
        X.append(series[i:i + window])
        y.append(series[i + window + horizon - 1])
    return np.array(X), np.array(y)


def build_per_site(long_df, window, horizon, train_split):
    """
    Walk through each (SCATS site, approach) pair, build windowed X/y and
    split chronologically. Returns concatenated train/test arrays plus the
    per-approach min/max scalers.

    Each SCATS intersection has up to 4 directional approaches (e.g. N/S/E/W)
    that share one SCATS Number but record very different flows, so we treat
    each approach as its own time series.
    """
    train_X, train_y = [], []
    test_X, test_y = [], []
    scalers = {}

    skipped = 0
    for (sid, location), group in long_df.groupby(['SCATS Number', 'Location']):
        vols = group['Volume'].values.astype(float)

        # If an approach has very little data the windows would barely cover
        # it — skip rather than train/test on a sliver.
        if len(vols) < window + horizon + 50:
            skipped += 1
            continue

        scaled, (vmin, vmax) = normalise(vols)
        X, y = make_windows(scaled, window, horizon)

        cut = int(len(X) * train_split)
        train_X.append(X[:cut])
        train_y.append(y[:cut])
        test_X.append(X[cut:])
        test_y.append(y[cut:])

        # Key the scaler by both fields so we can match predictions back
        # to the right approach later.
        scalers[f"{int(sid)}|{location}"] = {'min': vmin, 'max': vmax}

    if skipped:
        print(f"  skipped {skipped} site(s) with too little data")

    train_X = np.concatenate(train_X)
    train_y = np.concatenate(train_y)
    test_X = np.concatenate(test_X)
    test_y = np.concatenate(test_y)

    # LSTM/GRU layers in Keras expect a 3-D tensor (samples, timesteps, features).
    # We have 1 feature (volume), so add the trailing axis here so the ML code
    # downstream doesn't have to remember to do it.
    train_X = train_X[..., np.newaxis]
    test_X = test_X[..., np.newaxis]

    return train_X, train_y, test_X, test_y, scalers
