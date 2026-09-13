# SENTINEL: Cross-Modal RF-SAR Anomaly Fusion for Confidence-Weighted Event Detection

A research project demonstrating that combining two independent, weak-ish sensor signals — RF anomaly detection and SAR change detection — through a properly learned fusion model produces meaningfully better event-detection performance than either sensor alone, or than naively combining them.

## Motivation

Single-modality anomaly detectors (RF-only or SAR-only) suffer from high false-positive rates in isolation. This project treats spatiotemporal correlation between independent modalities as the primary confidence signal, and quantifies the improvement this yields — with real numbers, not just intuition. The framework is positioned as dual-use, relevant to both ISRO (satellite/maritime monitoring) and IAF (ISR/border monitoring), demonstrated here on a public, unclassified testbed: Visakhapatnam Port.

## Architecture

| Module | What it does |
|---|---|
| 1. RF Anomaly Detection | Detects unusual radio signal activity (baseline Z-score vs. autoencoder) |
| 2. SAR Change Detection | Detects physical change between two Sentinel-1 SAR images of the same area |
| 3. Fusion Layer | Combines Module 1 & 2 outputs into a single confidence score — the core novel contribution |
| 4. LLM Report Generation | Turns a fused, high-confidence detection into a grounded, readable analyst brief |

## Key Results

**Module 1 (RF):** Baseline (Z-score) vs. autoencoder detector, evaluated against synthetic ground truth — a genuine precision/recall trade-off (baseline: P=0.391/R=0.150/F1=0.217; autoencoder: P=0.128/R=0.433/F1=0.198), neither dominating the other.

**Module 2 (SAR):** Real Sentinel-1 GRD imagery (June 25 vs. Sept 5, 2026) over Visakhapatnam Port, calibrated/speckle-filtered/geocoded via SNAP, compared via log-ratio change detection — ~5% of the area flagged as changed, visually concentrated on the container terminal.

**Module 3 (Fusion) — headline result:**

| Method | Precision | Recall | F1 |
|---|---|---|---|
| RF-alone | 0.524 | 0.880 | 0.657 |
| SAR-alone | 1.000 | 0.160 | 0.276 |
| Naive fusion (average) | 0.941 | 0.640 | 0.762 |
| AND-correlation fusion | 1.000 | 0.160 | 0.276 |
| **Learned fusion (logistic regression)** | **1.000** | **0.840** | **0.913** |

Learned fusion beats every single-modality baseline and naive combination method.

**Module 4 (LLM reports):** Local LLM (Ollama, llama3.2:3b) generates grounded analyst briefs. A real hallucination was found during manual review (v1 prompt: fabricated timestamp, unrequested speculative threat language) and fixed via a stricter grounding prompt (v2) — documented as a before/after comparison in `outputs/analyst_reports_v1_original.txt` vs. `outputs/analyst_reports_v2_strict.txt`.

## Setup

**Prerequisites:** Python 3.11+, [SNAP](https://step.esa.int/main/download/snap-download/) (for SAR preprocessing), [Ollama](https://ollama.com) (for Module 4).

```bash
pip install numpy pandas matplotlib scikit-learn rasterio requests streamlit
ollama pull llama3.2:3b
```

**Run the pipeline** (each module can be run independently from its own folder):
```bash
cd module1_rf && python detect_rf_anomalies.py
cd module2_sar && python detect_sar_change.py
cd module3_fusion && python fusion_layer.py
cd module4_llm && python generate_report.py
```

**Run the interactive demo:**
```bash
python -m streamlit run app.py
```

## Data

- **SAR:** Real Sentinel-1 GRD imagery, downloaded via [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu) (ESA open data). Raw/processed files are excluded from this repo via `.gitignore` due to size (~1.6GB each) — re-download using the dates/AOI noted in `module2_sar/S1_preprocessing_graph.xml`.
- **RF:** Synthetic, generated deterministically via `module1_rf/generate_rf_data.py` (seed=42) — reproducible from code, not stored as a file.

## Limitations

- **RF ground truth is synthetic**, since real labeled RF-anomaly data tied to a specific place/time isn't publicly available. SAR data is real; RF anomalies are designed injections with known ground truth, calibrated to realistic published occupancy statistics.
- **Module 3's test set is also synthetic** (50 generated events with locations tied to the real SAR change map), needed since the real Module 1/2 data alone (3 RF events, 1 SAR comparison) wasn't sufficient to properly evaluate a fusion method.
- **SAR AOI cropping used estimated coordinates**, not the exact original Copernicus Browser selection — visually validated as landing correctly on the port, but not pixel-exact.
- **Local LLM (3B parameters) is prone to fabrication** even with a grounding-focused prompt — one hallucination was caught and fixed (see Module 4 above), but this remains a known risk; production deployment would need automated fact-checking against source data before any report reaches a human analyst.

## Project Structure

```
SENTINEL/
├── data/                  # raw/processed data (large files gitignored)
├── module1_rf/            # RF anomaly detection
├── module2_sar/           # SAR preprocessing + change detection
├── module3_fusion/        # cross-modal fusion layer
├── module4_llm/           # LLM report generation
├── outputs/               # results, plots, evaluation CSVs, generated reports
├── app.py                 # Streamlit interactive demo
└── README.md
```