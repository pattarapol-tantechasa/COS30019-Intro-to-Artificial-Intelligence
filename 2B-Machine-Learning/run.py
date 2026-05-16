import json
import sys

import numpy as np

import config
from data import load_raw, get_volume_cols, clean, to_long
from dataset import build_per_site


def main():
    if not config.SCATS_XLS.exists():
        print(f"Could not find {config.SCATS_XLS}")
        print(f"Drop the SCATS xls into the 'data' folder and run again.")
        sys.exit(1)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {config.SCATS_XLS.name} ...")
    df = load_raw(config.SCATS_XLS)
    vol_cols = get_volume_cols(df)
    print(f"  {len(df)} day-rows, {len(vol_cols)} volume columns")

    print("Cleaning ...")
    df = clean(df, vol_cols)

    # Per-approach location lookup — needed later for graph construction
    # (Task 3/4). Each SCATS intersection has up to 4 directional approaches,
    # so we keep them all here.
    sites = (df[['SCATS Number', 'Location', 'NB_LATITUDE', 'NB_LONGITUDE']]
             .drop_duplicates(subset=['SCATS Number', 'Location'])
             .sort_values(['SCATS Number', 'Location'])
             .reset_index(drop=True))
    sites.to_csv(config.OUTPUT_DIR / 'sites.csv', index=False)
    print(f"  {sites['SCATS Number'].nunique()} SCATS intersections "
          f"({len(sites)} approaches)")

    print("Reshaping wide -> long ...")
    long_df = to_long(df, vol_cols)
    long_df.to_csv(config.OUTPUT_DIR / 'scats_long.csv', index=False)
    print(f"  {len(long_df):,} (site, timestamp) rows")

    print(f"Building windows: window={config.WINDOW}, horizon={config.HORIZON} ...")
    train_X, train_y, test_X, test_y, scalers = build_per_site(
        long_df, config.WINDOW, config.HORIZON, config.TRAIN_SPLIT
    )
    print(f"  train: X={train_X.shape}  y={train_y.shape}")
    print(f"  test:  X={test_X.shape}  y={test_y.shape}")

    np.save(config.OUTPUT_DIR / 'train_X.npy', train_X)
    np.save(config.OUTPUT_DIR / 'train_y.npy', train_y)
    np.save(config.OUTPUT_DIR / 'test_X.npy', test_X)
    np.save(config.OUTPUT_DIR / 'test_y.npy', test_y)

    with open(config.OUTPUT_DIR / 'scalers.json', 'w') as f:
        json.dump(scalers, f, indent=2)

    print(f"\nDone. Files written to: {config.OUTPUT_DIR}")


if __name__ == '__main__':
    main()
