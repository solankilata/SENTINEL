import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the RF data and ground truth we generated earlier
# "../" means "go up one folder from module1_rf, into data/rf_synthetic"
signal_df = pd.read_csv("../data/rf_synthetic/rf_signal_data.csv", parse_dates=["timestamp"])
ground_truth_df = pd.read_csv("../data/rf_synthetic/rf_ground_truth.csv", parse_dates=["start_timestamp", "end_timestamp"])

print(f"Loaded {len(signal_df)} signal readings")
print(f"Loaded {len(ground_truth_df)} ground truth events")
# ---------------------------------------------------------------
# STEP 2: BUILD THE "ANSWER KEY" AS A PER-SECOND LABEL
# ---------------------------------------------------------------
# Ground truth currently lists events as "start time to end time."
# We convert this into a True/False label for every single second,
# so we can later check each second's prediction against the truth.
# We only mark TRUE CORRELATED events here - the false-alarm event
# is deliberately left out, since that one is meant for testing the
# FUSION layer later, not the RF detector alone.

signal_df["is_true_anomaly"] = False

for _, row in ground_truth_df.iterrows():
    if row["is_true_correlated_event"]:
        mask = (signal_df["timestamp"] >= row["start_timestamp"]) & (signal_df["timestamp"] <= row["end_timestamp"])
        signal_df.loc[mask, "is_true_anomaly"] = True

print(f"Total true-anomaly seconds: {signal_df['is_true_anomaly'].sum()}")
# ---------------------------------------------------------------
# STEP 3: BASELINE DETECTOR - ROLLING Z-SCORE
# ---------------------------------------------------------------
# We use a 300-second (5 minute) window - deliberately much LARGER
# than our 30-second anomalies. This matters: if the window were too
# close in size to the anomaly itself, the anomaly would end up
# inside its own "what's normal" comparison window, dragging the
# local average up and hiding itself. Keeping the window much wider
# avoids this "self-contamination" problem.

WINDOW_SIZE = 300
Z_THRESHOLD = 3.0

rolling_mean = signal_df["signal_power_dbfs"].rolling(WINDOW_SIZE, center=True, min_periods=30).mean()
rolling_std = signal_df["signal_power_dbfs"].rolling(WINDOW_SIZE, center=True, min_periods=30).std()

z_scores = (signal_df["signal_power_dbfs"] - rolling_mean) / rolling_std
signal_df["baseline_flag"] = (z_scores.abs() > Z_THRESHOLD).fillna(False)

print(f"Baseline (Z-score) flagged {signal_df['baseline_flag'].sum()} seconds as anomalous")
# ---------------------------------------------------------------
# STEP 4: UPGRADED DETECTOR - AUTOENCODER RECONSTRUCTION ERROR
# ---------------------------------------------------------------
# An autoencoder is a small neural network trained to compress a
# short window of signal down to a few numbers, then reconstruct it
# back. If it's only ever trained on "normal" data, it gets very
# good at reconstructing normal patterns - but bad at reconstructing
# anomalies it's never seen. So a HIGH reconstruction error is a
# strong signal that "this doesn't look like normal data."
#
# Unlike the Z-score baseline (which looks at single points), the
# autoencoder looks at short SEQUENCES/PATTERNS - so it can catch
# anomalies with an unusual shape, not just an unusual single value.

from sklearn.neural_network import MLPRegressor

SEQ_LEN = 10  # look at 10-second windows as the "shape" the autoencoder learns

def make_sequences(values, seq_len):
    """Cuts the signal into overlapping short windows for the autoencoder to learn from."""
    sequences = []
    for i in range(len(values) - seq_len + 1):
        sequences.append(values[i:i + seq_len])
    return np.array(sequences)

sequences = make_sequences(signal_df["signal_power_dbfs"].values, SEQ_LEN)

# Train ONLY on data that's NOT a true anomaly - mirrors a realistic
# setup: in the real world you'd train on a known-normal recording
# period, not on data containing the very anomalies you want to catch.
train_mask = ~signal_df["is_true_anomaly"].values[:len(sequences)]
train_sequences = sequences[train_mask]

# A small autoencoder: input -> compress to 3 numbers -> reconstruct back to 10
autoencoder = MLPRegressor(
    hidden_layer_sizes=(6, 3, 6),  # compress down to 3, then back up - the "bottleneck"
    max_iter=500,
    random_state=42,
)
autoencoder.fit(train_sequences, train_sequences)  # input = target: learns to reconstruct itself

# Run ALL sequences (including anomalies) through it, measure reconstruction error
reconstructed = autoencoder.predict(sequences)
reconstruction_error = np.mean((sequences - reconstructed) ** 2, axis=1)

# Pad to match original signal length (sequences are shorter than the original)
padded_error = np.concatenate([reconstruction_error, [np.nan] * (len(signal_df) - len(reconstruction_error))])
signal_df["reconstruction_error"] = padded_error

# Flag anomalies: error above the 95th percentile of error seen on normal data
error_threshold = np.percentile(reconstruction_error[train_mask], 95)
signal_df["autoencoder_flag"] = (signal_df["reconstruction_error"] > error_threshold).fillna(False)

print(f"Autoencoder flagged {signal_df['autoencoder_flag'].sum()} seconds as anomalous")
print(f"(error threshold used: {error_threshold:.3f})")

# ---------------------------------------------------------------
# STEP 5: EVALUATE BOTH DETECTORS AGAINST GROUND TRUTH
# ---------------------------------------------------------------
# PRECISION: of everything we flagged, how much was actually real?
#            (high precision = few false alarms)
# RECALL:    of all the real anomalies, how many did we catch?
#            (high recall = few missed events)
# F1:        a single combined score balancing precision and recall

y_true = signal_df["is_true_anomaly"]

results = {}
for method in ["baseline_flag", "autoencoder_flag"]:
    y_pred = signal_df[method]
    results[method] = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

print("\n" + "=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)
for method, scores in results.items():
    print(f"\n{method}:")
    print(f"  Precision: {scores['precision']:.3f}")
    print(f"  Recall:    {scores['recall']:.3f}")
    print(f"  F1 Score:  {scores['f1']:.3f}")

# ---------------------------------------------------------------
# STEP 6: SAVE RESULTS AND A COMPARISON PLOT
# ---------------------------------------------------------------
signal_df.to_csv("../outputs/rf_detection_results.csv", index=False)

results_df = pd.DataFrame(results).T
results_df.to_csv("../outputs/rf_detector_evaluation.csv")
print("\nSaved: rf_detection_results.csv, rf_detector_evaluation.csv")

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(signal_df["timestamp"], signal_df["signal_power_dbfs"], linewidth=0.5, color="steelblue")
axes[0].scatter(signal_df.loc[signal_df["baseline_flag"], "timestamp"],
                 signal_df.loc[signal_df["baseline_flag"], "signal_power_dbfs"],
                 color="orange", s=10, label="Baseline (Z-score) flagged", zorder=5)
axes[0].set_title("Baseline Detector (Rolling Z-Score)")
axes[0].set_ylabel("Signal Power (dBFS)")
axes[0].legend()

axes[1].plot(signal_df["timestamp"], signal_df["signal_power_dbfs"], linewidth=0.5, color="steelblue")
axes[1].scatter(signal_df.loc[signal_df["autoencoder_flag"], "timestamp"],
                 signal_df.loc[signal_df["autoencoder_flag"], "signal_power_dbfs"],
                 color="red", s=10, label="Autoencoder flagged", zorder=5)
axes[1].set_title("Upgraded Detector (Autoencoder Reconstruction Error)")
axes[1].set_ylabel("Signal Power (dBFS)")
axes[1].set_xlabel("Time")
axes[1].legend()

plt.tight_layout()
plt.savefig("../outputs/rf_detector_comparison.png", dpi=150)
print("Saved: rf_detector_comparison.png")