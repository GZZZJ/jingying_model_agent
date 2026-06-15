#!/usr/bin/env python
"""Convert 筛选300维度特征样本.feather: string features → float32, save slim version."""
import gc, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

src = "/root/notebook/draft/筛选300维度特征样本.feather"
dst = "/root/notebook/draft/筛选300维度特征样本_f32.feather"

# Load feature list
project_dir = Path(__file__).resolve().parent
feat_file = project_dir / "runs/feature_refine_feather/final_features.txt"
feat_list = [l.strip() for l in feat_file.read_text().splitlines() if l.strip() and not l.startswith("#")]
print(f"Feature list: {len(feat_list)} features")

# Load full feather
print(f"Loading {src} ...")
t0 = time.time()
df = pd.read_feather(src)
n_rows = len(df)
print(f"Loaded {n_rows} rows × {len(df.columns)} cols in {time.time()-t0:.0f}s")
print(f"Memory before conversion: {df.memory_usage(deep=True).sum() / 1e9:.1f} GB")

# Convert feature columns string→float32 in-place
print("Converting features to float32...")
for col in tqdm(feat_list):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].astype('float32')

# Also convert any non-feature float64 columns to float32
for col in df.columns:
    if col not in feat_list and df[col].dtype == np.float64:
        df[col] = df[col].astype(np.float32)

gc.collect()
mem_gb = df.memory_usage(deep=True).sum() / 1e9
print(f"Memory after conversion: {mem_gb:.1f} GB")

# Save
print(f"Writing {dst} ...")
df.to_feather(dst)
del df
gc.collect()

size_gb = Path(dst).stat().st_size / 1e9
elapsed = time.time() - t0
print(f"Done! Output: {dst} ({size_gb:.1f} GB), total time: {elapsed:.0f}s")
