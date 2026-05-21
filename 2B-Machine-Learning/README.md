# Data cleaning and processing

Pipeline that takes the raw VicRoads SCATS xls (October 2006, Boroondara) and
turns it into windowed numpy arrays ready for LSTM/GRU/etc.

## Quick start

1. Drop `Scats Data October 2006.xls` into `./data/`.
2. Install the deps:
   ```
   pip install pandas numpy xlrd
   ```
3. Run it:
   ```
   python run.py
   ```

## Output

Files written to `./output/`:

| File | What it is |
| --- | --- |
| `sites.csv` | One row per SCATS site with location + lat/long. Used later for the routing graph. |
| `scats_long.csv` | Long-format readings: `(SCATS Number, Location, lat, long, Timestamp, Volume)`. |
| `train_X.npy`, `train_y.npy` | Training set. `train_X` shape is `(samples, window, 1)`, `train_y` is `(samples,)`. |
| `test_X.npy`, `test_y.npy` | Test set, same shape, chronological split per site. |
| `scalers.json` | Per-site `{min, max}` so predictions can be inverted back to vehicle counts. |

## Notes on the pipeline

- **Window / horizon**: defaults to 12 / 1 — the model gets the last 3 hours
  of 15-min readings and predicts the next one. Change `WINDOW` and `HORIZON`
  in `config.py` if you want to experiment with different settings.
- **Cleaning**: rows missing SCATS Number or Date are dropped. Negative
  volumes get clipped to zero. Small gaps within a day are interpolated;
  whole days that are still blank afterwards are dropped.
- **Scaling**: each site is normalised to `[0, 1]` independently. This is
  important — sites have very different traffic levels, so a single global
  scaler would squash the quieter ones.
- **Train/test split**: 80/20, chronological, per site. No shuffling — the
  model never sees future readings during training.

## Files

- `config.py` — paths and parameters.
- `data.py` — load, clean, reshape from wide to long.
- `dataset.py` — normalise, build sliding windows, split per site.
- `run.py` — entry point. Runs the whole pipeline.


## Running the ML Models
Run each model script after `run.py` has been executed:

```
python lstm.py    # LSTM model
python gru.py     # GRU model
python saes.py    # Stacked Autoencoders (SAEs)
```


Each script reads from `./output/` and writes results to its own subdirectory:

| Directory | Contents |
| --- | --- |
| `output/lstm/` | `lstm_model.keras`, `lstm_metrics.json`, loss + prediction plots |
| `output/gru/` | `gru_model.keras`, `gru_metrics.json`, loss + prediction plots |
| `output/saes/` | `saes_model.keras`, `saes_metrics.json`, loss + prediction plots |

## Comparing Models
Once all three models have been trained, run:
```
python compare.py
```

Output is written to `output/comparison/`:

| File | What it is |
| --- | --- |
| `comparison_metrics.csv` | MAE, RMSE, R² for all three models in one table |
| `comparison_bar.png` | Separate bar chart per metric |
| `comparison_grouped.png` | All metrics grouped side by side |

## Dependencies
Full list — install with:
```
pip install -r requirements.txt
```
Requires Python 3.11. 
Models were developed and tested with TensorFlow 2.21 / Keras 3.13.