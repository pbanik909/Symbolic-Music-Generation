import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from src.config import RAW_MAESTRO_DIR, PROCESSED_DIR
from src.preprocessing.midi_parser import process_single_file

df = pd.read_csv(RAW_MAESTRO_DIR / "maestro-v3.0.0.csv")
genre_mapping = {
    "Johann Sebastian Bach": "Baroque", "Domenico Scarlatti": "Baroque", "George Frideric Handel": "Baroque",
    "Wolfgang Amadeus Mozart": "Classical", "Joseph Haydn": "Classical", "Ludwig van Beethoven": "Classical",
    "Fr\u00e9d\u00e9ric Chopin": "Romantic", "Franz Liszt": "Romantic", "Johannes Brahms": "Romantic",
    "Franz Schubert": "Romantic", "Robert Schumann": "Romantic", "Felix Mendelssohn": "Romantic",
    "Claude Debussy": "Impressionist_Modern", "Maurice Ravel": "Impressionist_Modern",
    "Alexander Scriabin": "Impressionist_Modern", "Sergei Rachmaninoff": "Impressionist_Modern",
    "Isaac Alb\u00e9niz": "Impressionist_Modern",
}
df["proxy_genre"] = df["canonical_composer"].map(genre_mapping)
df = df.dropna(subset=["proxy_genre"]).copy()
print("Files to parse:", len(df))

buckets = {"train": [], "validation": [], "test": []}
total = len(df)
for i, (_, row) in enumerate(df.iterrows()):
    p = RAW_MAESTRO_DIR / row["midi_filename"]
    if p.exists():
        w = process_single_file(p)
        if w.size > 0:
            buckets[row["split"]].append(w)
    if (i + 1) % 100 == 0:
        print(f"  processed {i+1}/{total}")

for split, lst in buckets.items():
    if lst:
        arr = np.concatenate(lst, axis=0)
        out = PROCESSED_DIR / f"maestro_{split}_windows.npy"
        np.save(out, arr)
        print(f"{split}: {arr.shape} -> saved")
    else:
        print(f"{split}: NO DATA")
print("Done.")
