"""
SENTINEL - Interactive Demo
==============================
A Streamlit app to explore fused RF-SAR detections and their
generated analyst reports, and compare detection methods.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="SENTINEL Demo", layout="wide")

st.title("🛰️ SENTINEL: Cross-Modal RF-SAR Anomaly Fusion")
st.caption("Visakhapatnam Port monitoring — RF anomaly detection + SAR change detection + confidence-weighted fusion")

# ---------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    events = pd.read_csv("outputs/fusion_test_events.csv")
    eval_results = pd.read_csv("outputs/fusion_evaluation.csv", index_col=0)
    return events, eval_results

events_df, eval_df = load_data()

# ---------------------------------------------------------------
# SECTION 1: METHOD COMPARISON (the headline result)
# ---------------------------------------------------------------
st.header("1. Detection Method Comparison")
st.write("Precision / Recall / F1 across all tested fusion strategies — this is the core result showing learned fusion beats every single-modality baseline and naive combination.")

col1, col2 = st.columns([2, 1])
with col1:
    st.dataframe(eval_df.style.format("{:.3f}").highlight_max(axis=0, color="lightgreen"))
with col2:
    st.image("outputs/fusion_comparison.png", use_container_width=True)

# ---------------------------------------------------------------
# SECTION 2: SAR CHANGE DETECTION VISUAL
# ---------------------------------------------------------------
st.header("2. SAR Change Detection")
st.write("Real Sentinel-1 imagery of Visakhapatnam Port, before/after comparison via log-ratio change detection.")
st.image("outputs/sar_change_detection.png", use_container_width=True)

# ---------------------------------------------------------------
# SECTION 3: RF ANOMALY DETECTION VISUAL
# ---------------------------------------------------------------
st.header("3. RF Anomaly Detection")
st.write("Baseline (Z-score) vs upgraded (autoencoder) detector on synthetic RF signal data.")
st.image("outputs/rf_detector_comparison.png", use_container_width=True)

# ---------------------------------------------------------------
# SECTION 4: EXPLORE INDIVIDUAL FUSED EVENTS + GENERATED REPORTS
# ---------------------------------------------------------------
st.header("4. Explore Fused Events")

high_conf_events = events_df[events_df["learned_fusion_flag"] == True].reset_index(drop=True)
st.write(f"**{len(high_conf_events)}** events flagged as high-confidence by the learned fusion model.")

event_idx = st.selectbox(
    "Select an event to inspect:",
    options=range(len(high_conf_events)),
    format_func=lambda i: f"Event {i+1} — row {high_conf_events.iloc[i]['row']}, col {high_conf_events.iloc[i]['col']}"
)

selected = high_conf_events.iloc[event_idx]

col1, col2, col3 = st.columns(3)
col1.metric("RF Score", f"{selected['rf_score']:.3f}")
col2.metric("SAR Score", f"{selected['sar_score']:.3f}")
col3.metric("Fusion Decision", "✅ HIGH CONFIDENCE")

# ---------------------------------------------------------------
# SECTION 5: SHOW THE GENERATED LLM REPORT FOR THIS EVENT
# ---------------------------------------------------------------
st.subheader("Generated Analyst Brief")

@st.cache_data
def load_reports(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return content.split("=" * 60)[1:]  # split into individual report blocks

reports_v2 = load_reports("outputs/analyst_reports_v2_strict.txt")

if event_idx < len(reports_v2):
    st.text(reports_v2[event_idx].strip())
else:
    st.info("No generated report available for this specific event index.")

st.divider()
st.caption("SENTINEL — B.Sc. (Hons.) Data Science & AI term project | IIT Guwahati")