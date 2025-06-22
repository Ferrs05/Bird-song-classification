import os
import csv
import random

# Hard‐code path root proyek
ROOT_DIR = r"C:\Users\muhfe\OneDrive\Documents\BELAJAR\File Project\Project Birdsong"
RAW_DIR  = os.path.join(ROOT_DIR, "data", "raw")
AUG_DIR  = os.path.join(ROOT_DIR, "data", "augmented")

# Proporsi split
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

def get_all_records():
    records = []
    # Original
    for fname in os.listdir(RAW_DIR):
        if not fname.lower().endswith(".wav"):
            continue
        label = fname.split("_")[0]
        records.append({"filename": fname, "label": label, "type": "original"})
    # Augmented
    for fname in os.listdir(AUG_DIR):
        if not fname.lower().endswith(".wav"):
            continue
        parts = fname.split("_")
        label = parts[1] if parts[0] in ["orig","stretch","pitch","noise"] else parts[0]
        records.append({"filename": fname, "label": label, "type": "augmented"})
    return records

def assign_splits(records):
    random.seed(RANDOM_SEED)
    by_label = {}
    for r in records:
        by_label.setdefault(r["label"], []).append(r)
    final = []
    for label, recs in by_label.items():
        random.shuffle(recs)
        n = len(recs)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)
        for i, r in enumerate(recs):
            if   i < n_train:          split = "train"
            elif i < n_train + n_val:  split = "val"
            else:                      split = "test"
            r["split"] = split
            final.append(r)
    return final

def write_csv(records):
    dst = os.path.join(ROOT_DIR, "metadata.csv")
    with open(dst, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename","label","type","split"])
        writer.writeheader()
        writer.writerows(records)
    print(f"metadata.csv generated: {dst} ({len(records)} rows)")

if __name__ == "__main__":
    recs = get_all_records()
    recs = assign_splits(recs)
    write_csv(recs)
