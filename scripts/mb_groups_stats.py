#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp

# ───────────────────────── CONFIG
DATA_CSV = "MBmerged-z-scores_MLready_correction.csv"
OUTDIR   = Path("MBgroups_stats")
OUTDIR.mkdir(exist_ok=True)

EPS = 1e-6  # small constant for numerical stability

# ───────────────────────── HELPERS
def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d for two independent samples (pooled SD). Returns NaN if SD=0 or insufficient data."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    nx, ny = x.size, y.size
    if nx < 2 or ny < 2:
        return np.nan
    mx, my = x.mean(), y.mean()
    sx = x.std(ddof=1)
    sy = y.std(ddof=1)
    # pooled SD
    sp2 = ((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2) if (nx + ny - 2) > 0 else np.nan
    sp = np.sqrt(sp2) if np.isfinite(sp2) else np.nan
    if not np.isfinite(sp) or sp == 0:
        return np.nan
    return (mx - my) / sp

# ───────────────────────── LOAD DATA
df_all = pd.read_csv(DATA_CSV).dropna(subset=["MOLECULAR"])
# keep only numeric features
X_full = df_all.drop(columns=["MOLECULAR"], errors="ignore").select_dtypes(include=[np.number])
y_full = df_all["MOLECULAR"]

# Consistent group ordering
groups = sorted(pd.unique(y_full))

# ───────────────────────── Kruskal–Wallis + Dunn post-hoc
results_rows = []

for col in X_full.columns:
    col_vals = df_all[col]
    # prepare samples for KW
    samples = [df_all.loc[y_full == g, col].dropna().values for g in groups]
    if sum(len(s) > 0 for s in samples) < 2:
        continue

    # Omnibus KW (across all groups for this feature)
    try:
        kw_stat, kw_p = kruskal(*samples)
    except Exception:
        # if a degenerate case occurs, skip KW but still do pairwise if possible
        kw_stat, kw_p = np.nan, np.nan

    results_rows.append({
        "feature": col,
        "test": "Kruskal-Wallis",
        "group1": "ALL",
        "group2": "ALL",
        "statistic": kw_stat,
        "pval": kw_p,
        "mean_diff": np.nan,
        "cohen_d": np.nan,
        "log2fc_shifted": np.nan
    })

    # Dunn post-hoc p-values (unadjusted; we will adjust per-feature later)
    dunn = sp.posthoc_dunn(df_all, val_col=col, group_col="MOLECULAR", p_adjust=None)

    # define a per-feature shift so (mean + shift) > 0 even if means ≤ 0
    # choose shift based on the minimum finite value in the feature column
    finite_vals = col_vals[np.isfinite(col_vals)]
    min_val = finite_vals.min() if finite_vals.size else 0.0
    shift = max(EPS, -(min_val) + EPS)

    # Pairwise rows
    for i, g1 in enumerate(groups):
        for g2 in groups[i+1:]:
            # robustly extract p from Dunn matrix (index/columns may be unordered)
            try:
                p_pair = dunn.loc[g1, g2]
            except KeyError:
                try:
                    p_pair = dunn.loc[g2, g1]
                except KeyError:
                    p_pair = np.nan

            vals1 = df_all.loc[y_full == g1, col].dropna().values
            vals2 = df_all.loc[y_full == g2, col].dropna().values

            # Means (could be negative)
            mean1 = np.mean(vals1) if vals1.size else np.nan
            mean2 = np.mean(vals2) if vals2.size else np.nan

            # Effect sizes for z-scores
            mean_diff = (mean1 - mean2) if np.isfinite(mean1) and np.isfinite(mean2) else np.nan
            d = cohen_d(vals1, vals2)

            # Shifted log2 fold change to avoid NaNs with ≤0 means
            if np.isfinite(mean1) and np.isfinite(mean2):
                m1 = mean1 + shift
                m2 = mean2 + shift
                log2fc_shifted = np.log2(m1 / m2) if (m1 > 0 and m2 > 0) else np.nan
            else:
                log2fc_shifted = np.nan

            results_rows.append({
                "feature": col,
                "test": "Dunn",
                "group1": g1,
                "group2": g2,
                "statistic": np.nan,
                "pval": p_pair,
                "mean_diff": mean_diff,
                "cohen_d": d,
                "log2fc_shifted": log2fc_shifted
            })

# Put into DataFrame
res_df = pd.DataFrame(results_rows)

# ───────────────────────── FDR adjust p-values PER FEATURE
# (keeps your behavior of adjusting all rows for the feature, including the KW row)
out_rows = []
for feat, subdf in res_df.groupby("feature", sort=False):
    # Replace NaN p with 1.0 so they don't affect FDR
    pvals = subdf["pval"].fillna(1.0).to_numpy()
    rej, p_adj, _, _ = multipletests(pvals, method="fdr_bh")
    subdf = subdf.copy()
    subdf["pval_adj"] = p_adj
    subdf["significant"] = rej
    out_rows.append(subdf)

final_df = pd.concat(out_rows, ignore_index=True)

# ───────────────────────── SAVE
out_csv = OUTDIR / "pairwise_stats.csv"
final_df.to_csv(out_csv, index=False)
print(f"Pairwise stats saved → {out_csv}")
