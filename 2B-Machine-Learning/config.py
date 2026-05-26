from pathlib import Path

# All paths are relative to this file so the script works regardless of where
# you run it from.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

# Drop the SCATS xls into ./data/ before running. Default name matches what
# was provided on Canvas.
SCATS_XLS = DATA_DIR / "Scats Data October 2006.xls"

# How many past 15-min readings the ML model gets to look at.
# 12 = 3 hours of history. Most papers on traffic flow prediction use
# something between 8 and 16 — feel free to tune this later.
WINDOW = 12

# How far ahead we predict. 1 = the next 15 minutes.
HORIZON = 1

# Train/test split per site, chronological (no shuffling — we don't want the
# model peeking at future readings during training).
TRAIN_SPLIT = 0.8

# Sites flagged in the dataset's Notes sheet as having very poor data on
# selected days. The notes say data was already removed for those days, so
# we don't drop the entire site — just kept here for reference.
KNOWN_BAD_SITES = [970, 2000, 2820, 3001]

# Number of nearest SCATS neighbours to connect in the routing graph.
# Increase for a denser graph, decrease for faster search.
K_NEIGHBOURS = 5
