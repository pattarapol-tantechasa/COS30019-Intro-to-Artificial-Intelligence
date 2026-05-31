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

## Routing

The routing pipeline connects the ML traffic predictions to the A* search
algorithm to find optimal paths between SCATS intersections.

### Files
- `travel_time.py` — converts ML predicted volume to travel time (seconds)
- `routing/graph.py` — Graph class used by A* (uses unchanged graph.py from part 2A)
- `routing/graph_builder.py` — builds the graph from SCATS sites and travel time predictions
- `routing/astar.py` — A* search algorithm (uses unchanged A* algorithm from part 2A)
- `routing/yen.py` — Yen's k-shortest paths algorithm, wraps A* to find up to 5 routes

### Running the routing pipeline
Requires at least one trained model (`lstm`, `gru`, or `saes`) before running.

You can test the full routing pipeline with the sample origin/destination from the assignment 2B brief:
```
python routing/yen.py
```

This will:
1. Load the trained SAEs model (change `MODEL_NAME` in `yen.py` to use `lstm` or `gru`)
2. Build a graph of SCATS intersections with travel time edge costs
3. Find the top 5 routes between site 2000 and site 3002
4. Print each route with estimated travel time

## GUI

A Tkinter-based graphical interface is available with three tabs:
- **Route Finder** — enter origin/destination SCATS site numbers, select a model, and find up to 5 routes with estimated travel times and an interactive graph
- **Model Visualisation** — browse loss curve and prediction plots per model
- **Model Comparison** — side-by-side MAE/RMSE/R² bar charts across all three models

### Running the GUI
Requires at least one trained model before launching.
Full GUI implementation can be ran using
```
python gui.py
```

### Usage
1. Select the **Route Finder** tab
2. Enter an origin and destination SCATS site number (e.g. Origin: `2000`, Destination: `3002`)
3. Select a model (`lstm`, `gru`, or `saes`)
4. Click **Find Routes** — the graph builds in approximately 30 seconds
5. Up to 5 routes appear in the list with travel time and link count
6. Click any route to highlight it on the map and see the full node sequence

### Notes
- All scripts must be run from the project root directory
- The GUI loads whichever models have a `.keras` file in `output/` — run the model scripts first
- Model plots in the Visualisation tab require the corresponding model script to have been run


### Notes
- All scripts must be run from the project root directory
- `K_NEIGHBOURS = 5` in `config.py` controls how many nearest neighbours each site connects to
- A small number of edges may be skipped if a site has insufficient traffic data
- Travel time formula derived from the provided PDF on Canvas: `flow = -1.4648375 * speed² + 93.75 * speed`
- Speed is capped at 60 km/h when flow is below 351 vehicles/hour
- Each intersection adds a 30 second delay to the travel time

## Dependencies
Full list — install with:
```
pip install -r requirements.txt
```
Requires Python 3.11. 
Models were developed and tested with TensorFlow 2.21 / Keras 3.13.