import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ---------------------------------------------------------------
# STEP 1: SET THE BASIC "SHAPE" OF OUR SIMULATED DAY
# ---------------------------------------------------------------
# We are simulating one continuous RF monitoring session.
# Think of this like a sound-level meter that takes a reading
# every second for a set duration. We'll do that but for radio
# signal power instead of sound.

np.random.seed(42)  # makes results reproducible - same "random" numbers every run

DURATION_SECONDS = 3600          # simulate 1 hour of monitoring (3600 readings, 1/sec)
SAMPLE_RATE_HZ = 1               # 1 reading per second (simple, easy to reason about)
N_SAMPLES = DURATION_SECONDS * SAMPLE_RATE_HZ

# The "session" starts at a specific date/time - we'll pretend this
# recording happened on 2026-09-05 (matching our SAR "after" date),
# so later we can test whether our fusion layer correctly links them.
SESSION_START = datetime(2026, 9, 5, 0, 0, 0)

timestamps = [SESSION_START + timedelta(seconds=i) for i in range(N_SAMPLES)]

# ---------------------------------------------------------------
# STEP 2: BUILD THE "NORMAL" BACKGROUND SIGNAL
# ---------------------------------------------------------------
# Real ambient RF power isn't a flat line - it wobbles around a
# baseline "noise floor" value, with small random fluctuations.
# We calibrate our numbers to be realistic using published
# real-world occupancy statistics (~20-35% band occupancy is
# typical for a busy indoor/harbor-adjacent environment, per
# published spectrum-sensing measurements).

NOISE_FLOOR_DBFS = -70.0          # baseline "quiet" signal level (dBFS = decibels relative to full scale, a standard relative power unit)
NOISE_STD_DB = 2.0                # how much natural random wobble around the floor

# Base signal: noise floor + small random (Gaussian) wobble
background = np.random.normal(loc=NOISE_FLOOR_DBFS, scale=NOISE_STD_DB, size=N_SAMPLES)

# ---------------------------------------------------------------
# STEP 3: ADD REALISTIC "NORMAL" ACTIVITY BURSTS
# ---------------------------------------------------------------
# Real environments aren't just quiet noise - normal radio traffic
# (routine port communications, ship radios, etc.) causes short
# bursts of elevated signal. We add these so a detector has to
# tell the difference between "normal burst" and "real anomaly" -
# otherwise the problem would be too easy to be a real test.

OCCUPANCY_FRACTION = 0.25   # ~25% of the time, some normal activity is happening (grounded in real published occupancy figures)
n_burst_samples = int(N_SAMPLES * OCCUPANCY_FRACTION)
burst_indices = np.random.choice(N_SAMPLES, size=n_burst_samples, replace=False)

# Normal bursts: modest power increase (5-15 dB above floor), short-lived
for idx in burst_indices:
    burst_strength = np.random.uniform(5, 15)
    background[idx] += burst_strength

# ---------------------------------------------------------------
# STEP 4: DEFINE AND INJECT THE SYNTHETIC ANOMALIES
# ---------------------------------------------------------------
# An anomaly here represents something like a jamming/interference
# event: a MUCH stronger, MUCH more sustained power increase than
# normal traffic bursts - this mirrors how real jamming signals are
# described in RF literature (broadband, high-power, sustained).
#
# We choose the exact times ourselves - this becomes our ground
# truth. Some anomalies are placed to "align" with our SAR event
# (for later fusion testing = a TRUE correlated event), and some
# are placed randomly elsewhere (a FALSE ALARM case - anomaly
# happened but nothing physically changed nearby).

ANOMALY_DURATION_SEC = 30     # each anomaly lasts 30 seconds (sustained, unlike brief normal bursts)
ANOMALY_STRENGTH_DB = 25      # how much stronger than the noise floor (much bigger than normal bursts' 5-15 dB)

# Define anomaly injection points (seconds into the session)
# In a real project you'd tie this to your actual SAR pass time -
# here we place a few examples across the hour for demonstration.
injected_events = [
    {"start_sec": 600,  "label": "correlated_event_1", "is_true_event": True},   # 10 min in
    {"start_sec": 1800, "label": "false_alarm_1",       "is_true_event": False}, # 30 min in - anomaly with NO matching physical change
    {"start_sec": 2700, "label": "correlated_event_2",  "is_true_event": True},  # 45 min in
]

ground_truth_rows = []

for event in injected_events:
    start_idx = event["start_sec"] * SAMPLE_RATE_HZ
    end_idx = start_idx + (ANOMALY_DURATION_SEC * SAMPLE_RATE_HZ)

    # Inject the anomaly: a sustained elevated power level, not just one spike
    background[start_idx:end_idx] += ANOMALY_STRENGTH_DB + np.random.normal(0, 1.5, end_idx - start_idx)

    ground_truth_rows.append({
        "event_label": event["label"],
        "start_timestamp": timestamps[start_idx],
        "end_timestamp": timestamps[end_idx - 1],
        "is_true_correlated_event": event["is_true_event"],
    })

# ---------------------------------------------------------------
# STEP 5: SAVE THE OUTPUTS
# ---------------------------------------------------------------
# Two files come out of this:
#   1. rf_signal_data.csv   -> the actual "sensor readings" (what our
#      detector will later analyze, WITHOUT knowing the answer key)
#   2. rf_ground_truth.csv  -> the answer key (ONLY used later for
#      scoring/evaluating the detector, never fed into it directly)

signal_df = pd.DataFrame({
    "timestamp": timestamps,
    "signal_power_dbfs": background,
})

ground_truth_df = pd.DataFrame(ground_truth_rows)

signal_df.to_csv("rf_signal_data.csv", index=False)
ground_truth_df.to_csv("rf_ground_truth.csv", index=False)

print("Generated files:")
print(f"  - rf_signal_data.csv    ({len(signal_df)} rows)")
print(f"  - rf_ground_truth.csv   ({len(ground_truth_df)} rows)")
print("\nGround truth (the 'answer key'):")
print(ground_truth_df.to_string(index=False))

# ---------------------------------------------------------------
# STEP 6: VISUALIZE IT (sanity check - always look at your data!)
# ---------------------------------------------------------------
plt.figure(figsize=(14, 5))
plt.plot(signal_df["timestamp"], signal_df["signal_power_dbfs"], linewidth=0.5, color="steelblue", label="RF signal power")

# Mark the true injected anomalies in red so we can visually confirm they show up
for _, row in ground_truth_df.iterrows():
    color = "red" if row["is_true_correlated_event"] else "orange"
    plt.axvspan(row["start_timestamp"], row["end_timestamp"], color=color, alpha=0.3)

plt.xlabel("Time")
plt.ylabel("Signal Power (dBFS)")
plt.title("Simulated RF Background with Injected Anomalies\n(red = correlated true event, orange = false-alarm-style event)")
plt.legend()
plt.tight_layout()
plt.savefig("rf_signal_plot.png", dpi=150)
print("\nSaved plot: rf_signal_plot.png")
