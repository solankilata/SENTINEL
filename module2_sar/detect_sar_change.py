"""
SENTINEL - Module 2: SAR Change Detection
============================================

WHAT THIS SCRIPT DOES:
  1. Loads the two processed SAR images (before/after) we made with SNAP
  2. Aligns them so they're directly comparable pixel-by-pixel
  3. Computes a "log-ratio" change map - the standard technique for
     detecting change in SAR imagery
  4. Thresholds that into a clear change/no-change map
  5. Visualizes before/after/change side by side

WHY LOG-RATIO, SPECIFICALLY:
  SAR images have "multiplicative" noise (speckle) rather than the
  simple additive noise a normal camera has. If you just subtract
  (after - before), the noise doesn't behave nicely and swamps real
  change. Taking the RATIO (after / before), then a LOG of that ratio,
  turns this multiplicative noise into something statistically closer
  to normal, additive noise - which makes thresholding far more
  reliable. This is the standard, citable technique in SAR change
  detection literature.

IMPORTANT HONEST LIMITATION (for your report):
  Unlike our RF module, we don't have synthetic ground truth here -
  we don't know beforehand exactly what "really changed" at the port.
  So evaluation here is visual/qualitative, not a precision/recall
  score like Module 1. This is worth stating plainly in your report,
  not hidden.
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# STEP 1: LOAD ONLY OUR AOI (CROPPED), NOT THE WHOLE SWATH
# ---------------------------------------------------------------
# IMPORTANT FIX: our SNAP preprocessing graph processed the ENTIRE
# satellite swath (a huge strip of coastline), not just our small
# port AOI - the box we drew in Copernicus Browser only filtered
# search results, it didn't crop the actual data. Instead of
# reprocessing in SNAP (slow), we crop directly here using the
# geocoded coordinates - windowed reading also means we only load
# the small region into memory, not the full 2.5GB file.

from rasterio.windows import from_bounds

# Approximate bounding box around Visakhapatnam Port / container
# terminal (lon_min, lat_min, lon_max, lat_max) in WGS84 degrees.
# NOTE: this is an estimate - if the cropped image below doesn't
# show the port clearly, we'll adjust these numbers.
AOI_BOUNDS = (83.28, 17.68, 83.33, 17.71)  # (lon_min, lat_min, lon_max, lat_max)

def read_band_cropped(path, bounds):
    with rasterio.open(path) as src:
        window = from_bounds(*bounds, transform=src.transform)
        band = src.read(1, window=window).astype(np.float32)
    return band

BEFORE_PATH = "../data/sar_processed/vizag_20260625.data/Sigma0_VV.img"
AFTER_PATH = "../data/sar_processed/vizag_20260905.data/Sigma0_VV.img"

before = read_band_cropped(BEFORE_PATH, AOI_BOUNDS)
after = read_band_cropped(AFTER_PATH, AOI_BOUNDS)

print(f"Before image shape (cropped): {before.shape}")
print(f"After image shape (cropped): {after.shape}")

# ---------------------------------------------------------------
# STEP 2: ALIGN THE TWO IMAGES
# ---------------------------------------------------------------
# The two images might differ very slightly in size (a pixel or two)
# due to how each pass was individually geocoded. We crop both down
# to the smaller common size so every pixel position lines up.

min_rows = min(before.shape[0], after.shape[0])
min_cols = min(before.shape[1], after.shape[1])
before = before[:min_rows, :min_cols]
after = after[:min_rows, :min_cols]
print(f"Cropped both images to common shape: {before.shape}")

# ---------------------------------------------------------------
# STEP 3: MASK OUT NO-DATA / ZERO PIXELS
# ---------------------------------------------------------------
# Terrain-corrected images often have black (zero-value) borders
# where no actual data exists (edge of the imaged swath). We exclude
# these from our analysis so they don't get mistaken for "change."

valid_mask = (before > 0) & (after > 0)
print(f"Valid (non-zero) pixels: {valid_mask.sum()} / {valid_mask.size}")

# ---------------------------------------------------------------
# STEP 4: LOG-RATIO CHANGE DETECTION
# ---------------------------------------------------------------
epsilon = 1e-6  # tiny value to avoid dividing by zero
log_ratio = np.zeros_like(before)
log_ratio[valid_mask] = np.log10((after[valid_mask] + epsilon) / (before[valid_mask] + epsilon))

# ---------------------------------------------------------------
# STEP 5: THRESHOLD INTO A BINARY CHANGE MAP
# ---------------------------------------------------------------
# We flag a pixel as "changed" if its log-ratio is unusually far from
# the overall average log-ratio - same statistical logic as our RF
# Z-score baseline, just applied to an image instead of a signal.

valid_values = log_ratio[valid_mask]
mean_lr = valid_values.mean()
std_lr = valid_values.std()
THRESHOLD = 2.0  # standard deviations away to count as "changed"

change_mask = np.zeros_like(before, dtype=bool)
change_mask[valid_mask] = np.abs(log_ratio[valid_mask] - mean_lr) > THRESHOLD * std_lr

pct_changed = 100 * change_mask.sum() / valid_mask.sum()
print(f"Log-ratio mean={mean_lr:.4f}, std={std_lr:.4f}")
print(f"Flagged {change_mask.sum()} pixels as changed ({pct_changed:.2f}% of valid area)")

# ---------------------------------------------------------------
# STEP 6: VISUALIZE - BEFORE, AFTER, AND CHANGE MAP SIDE BY SIDE
# ---------------------------------------------------------------
def db_display(img):
    # Converts linear backscatter to decibels (dB) - a more
    # human-viewable scale for SAR images, standard practice for display.
    out = np.full_like(img, np.nan)
    mask = img > 0
    out[mask] = 10 * np.log10(img[mask])
    return out

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(db_display(before), cmap="gray")
axes[0].set_title("Before (2026-06-25)")
axes[0].axis("off")

axes[1].imshow(db_display(after), cmap="gray")
axes[1].set_title("After (2026-09-05)")
axes[1].axis("off")

axes[2].imshow(db_display(before), cmap="gray")
axes[2].imshow(np.ma.masked_where(~change_mask, change_mask), cmap="Reds", alpha=0.6)
axes[2].set_title(f"Detected Change ({pct_changed:.1f}% of area)")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("../outputs/sar_change_detection.png", dpi=150)
print("Saved: sar_change_detection.png")