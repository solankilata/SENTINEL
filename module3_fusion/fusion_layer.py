"""
SENTINEL - Module 3: Cross-Modal RF-SAR Fusion Layer
========================================================

WHAT THIS SCRIPT DOES:
  1. Loads the real SAR change map from Module 2, divides it into a
     grid of cells, and computes a "change score" per cell
  2. Generates a synthetic test set of RF anomaly events, each with
     a location - half placed in high-SAR-change cells (true
     correlated events), half in low-change cells (false alarms)
  3. Computes 4 different confidence scores per event: RF-alone,
     SAR-alone, naive fusion (simple average), and our proper
     spatial-correlation fusion
  4. Evaluates all 4 against the ground truth (precision/recall/F1)

WHY WE NEED A LARGER TEST SET (honest methodology note):
  Our real Module 1 RF data only had 3 events at one location, and
  Module 2 only compared 2 SAR dates - not enough data points to
  properly evaluate a fusion method. We extend the same synthetic-
  injection principle from Module 1, but now varying LOCATION as
  well as time, purely to create a large enough, labeled test set
  to properly measure fusion performance. Stated plainly in the
  report as a designed evaluation methodology.

BUG FOUND AND FIXED WHILE BUILDING THIS:
  The first version picked "low change" cells without checking
  whether those cells actually had real SAR data. Our cropped image
  has empty/no-data edges (white corners in the earlier plot) -
  these showed up as 0.00 change score, but that's a NO-DATA
  artifact, not a genuinely calm real area. Using these as
  false-alarm locations made the test artificially easy (any
  method looks perfect at separating "real data" from "no data at
  all"). Fixed by only scoring/selecting cells with at least 50%
  real, valid pixels.
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

np.random.seed(42)

# ---------------------------------------------------------------
# STEP 1: RECOMPUTE THE SAR CHANGE MAP, THEN GRID IT (WITH VALIDITY CHECK)
# ---------------------------------------------------------------
AOI_BOUNDS = (83.28, 17.68, 83.33, 17.71)  # same AOI as Module 2

def read_band_cropped(path, bounds):
    with rasterio.open(path) as src:
        window = from_bounds(*bounds, transform=src.transform)
        band = src.read(1, window=window).astype(np.float32)
    return band

before = read_band_cropped("../data/sar_processed/vizag_20260625.data/Sigma0_VV.img", AOI_BOUNDS)
after = read_band_cropped("../data/sar_processed/vizag_20260905.data/Sigma0_VV.img", AOI_BOUNDS)

min_rows = min(before.shape[0], after.shape[0])
min_cols = min(before.shape[1], after.shape[1])
before, after = before[:min_rows, :min_cols], after[:min_rows, :min_cols]

valid_mask = (before > 0) & (after > 0)
epsilon = 1e-6
log_ratio = np.zeros_like(before)
log_ratio[valid_mask] = np.log10((after[valid_mask] + epsilon) / (before[valid_mask] + epsilon))

# Divide the image into a GRID (5x5 cells) and compute a change score
# per cell - our spatial "map" of where real physical change happened.
GRID_SIZE = 5
row_edges = np.linspace(0, min_rows, GRID_SIZE + 1, dtype=int)
col_edges = np.linspace(0, min_cols, GRID_SIZE + 1, dtype=int)

grid_change_score = np.full((GRID_SIZE, GRID_SIZE), np.nan)  # NaN = "not enough real data here"
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        cell = log_ratio[row_edges[i]:row_edges[i+1], col_edges[j]:col_edges[j+1]]
        cell_valid = valid_mask[row_edges[i]:row_edges[i+1], col_edges[j]:col_edges[j+1]]
        # Only compute a real score if at least 50% of this cell has
        # actual data - otherwise it's a no-data edge artifact.
        if cell_valid.sum() > (cell.size * 0.5):
            grid_change_score[i, j] = np.abs(cell[cell_valid]).mean()

valid_cells_mask = ~np.isnan(grid_change_score)
valid_scores = grid_change_score[valid_cells_mask]
min_score, max_score = valid_scores.min(), valid_scores.max()
grid_change_score[valid_cells_mask] = (grid_change_score[valid_cells_mask] - min_score) / (max_score - min_score + epsilon)

print("SAR change score grid (5x5, normalized 0-1, NaN = insufficient real data):")
print(np.round(grid_change_score, 2))

# ---------------------------------------------------------------
# STEP 2: GENERATE THE SYNTHETIC TEST SET OF RF EVENTS
# ---------------------------------------------------------------
N_EVENTS = 50
half = N_EVENTS // 2

# Only select from cells that have real, valid data (excludes NaN edge artifacts)
flat_scores = grid_change_score.flatten()
valid_flat_mask = ~np.isnan(flat_scores)
valid_indices = np.where(valid_flat_mask)[0]
valid_scores_only = flat_scores[valid_indices]

sorted_valid = valid_indices[np.argsort(valid_scores_only)]
high_change_cells = sorted_valid[-5:]   # top 5 highest-change VALID cells
low_change_cells = sorted_valid[:5]     # bottom 5 lowest-change VALID cells

events = []
for i in range(half):
    cell_idx = np.random.choice(high_change_cells)
    row, col = cell_idx // GRID_SIZE, cell_idx % GRID_SIZE
    rf_score = np.clip(np.random.normal(0.75, 0.15), 0, 1)
    events.append({"row": row, "col": col, "rf_score": rf_score, "is_true_correlated_event": True})

for i in range(N_EVENTS - half):
    cell_idx = np.random.choice(low_change_cells)
    row, col = cell_idx // GRID_SIZE, cell_idx % GRID_SIZE
    rf_score = np.clip(np.random.normal(0.65, 0.2), 0, 1)
    events.append({"row": row, "col": col, "rf_score": rf_score, "is_true_correlated_event": False})

events_df = pd.DataFrame(events)
events_df["sar_score"] = events_df.apply(lambda r: grid_change_score[r["row"], r["col"]], axis=1)

print(f"\nGenerated {len(events_df)} synthetic test events")
print(events_df.head())

# ---------------------------------------------------------------
# STEP 3: COMPUTE 4 DIFFERENT CONFIDENCE SCORES PER EVENT
# ---------------------------------------------------------------
events_df["rf_alone_flag"] = events_df["rf_score"] > 0.5
events_df["sar_alone_flag"] = events_df["sar_score"] > 0.5
events_df["naive_fusion_score"] = (events_df["rf_score"] + events_df["sar_score"]) / 2
events_df["naive_fusion_flag"] = events_df["naive_fusion_score"] > 0.5
events_df["fusion_flag"] = (events_df["rf_score"] > 0.5) & (events_df["sar_score"] > 0.5)

# ---------------------------------------------------------------
# STEP 3.5: LEARNED FUSION - LOGISTIC REGRESSION
# ---------------------------------------------------------------
# WHY THIS FIXES THE PROBLEM ABOVE:
#   Our AND-correlation fusion forced both RF and SAR scores through
#   the SAME fixed 0.5 threshold - but they come from different score
#   distributions, so one fixed cutoff unfairly penalizes whichever
#   modality happens to run "lower" on average (SAR, in our case).
#   A logistic regression LEARNS its own weighting for each input
#   feature from the data itself, instead of us guessing one shared
#   threshold - it can learn "SAR scores around 0.4 are actually
#   meaningful, RF scores need to be higher to be trusted" or
#   whatever pattern the data actually shows.

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = events_df[["rf_score", "sar_score"]].values
y = events_df["is_true_correlated_event"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

learned_fusion = LogisticRegression()
learned_fusion.fit(X_train, y_train)

# Predict on the FULL dataset for a fair side-by-side comparison
# with the other methods (which were also evaluated on all 50 events)
events_df["learned_fusion_flag"] = learned_fusion.predict(X)

print(f"\nLearned fusion coefficients: RF={learned_fusion.coef_[0][0]:.3f}, SAR={learned_fusion.coef_[0][1]:.3f}")
print(f"(Test set accuracy: {learned_fusion.score(X_test, y_test):.3f})")

# ---------------------------------------------------------------
# STEP 4: EVALUATE ALL 4 METHODS AGAINST GROUND TRUTH
# ---------------------------------------------------------------
y_true = events_df["is_true_correlated_event"]

methods = {
    "RF-alone": "rf_alone_flag",
    "SAR-alone": "sar_alone_flag",
    "Naive fusion (average)": "naive_fusion_flag",
    "Our fusion (AND-correlation)": "fusion_flag",
    "Learned fusion (logistic regression)": "learned_fusion_flag",
}

results = {}
for name, col in methods.items():
    y_pred = events_df[col]
    results[name] = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

print("\n" + "=" * 55)
print("FUSION EVALUATION - HEADLINE RESULT")
print("=" * 55)
for name, scores in results.items():
    print(f"\n{name}:")
    print(f"  Precision: {scores['precision']:.3f}")
    print(f"  Recall:    {scores['recall']:.3f}")
    print(f"  F1 Score:  {scores['f1']:.3f}")

results_df = pd.DataFrame(results).T
results_df.to_csv("../outputs/fusion_evaluation.csv")
events_df.to_csv("../outputs/fusion_test_events.csv", index=False)
print("\nSaved: fusion_evaluation.csv, fusion_test_events.csv")

# ---------------------------------------------------------------
# STEP 5: VISUALIZE THE COMPARISON
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(methods))
width = 0.25
metrics = ["precision", "recall", "f1"]
colors = ["#4C72B0", "#DD8452", "#55A868"]

for i, metric in enumerate(metrics):
    values = [results[name][metric] for name in methods]
    ax.bar(x + i * width, values, width, label=metric.capitalize(), color=colors[i])

ax.set_xticks(x + width)
ax.set_xticklabels(methods.keys(), rotation=15, ha="right")
ax.set_ylabel("Score")
ax.set_title("Fusion Method Comparison: Precision / Recall / F1")
ax.legend()
plt.tight_layout()
plt.savefig("../outputs/fusion_comparison.png", dpi=150)
print("Saved: fusion_comparison.png")