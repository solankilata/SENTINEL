"""
SENTINEL - Module 4: LLM-Generated Analyst Reports
======================================================

WHAT THIS SCRIPT DOES:
  1. Loads the fused detection results from Module 3
  2. Picks out the high-confidence events (the ones "learned fusion"
     flagged as real, correlated events)
  3. For each one, builds a prompt containing the actual numbers
     (RF score, SAR score, location, confidence) and asks a local
     LLM (via Ollama) to write a short, structured analyst brief
  4. Saves all the generated reports to a single text file

WHY WE GIVE THE LLM THE ACTUAL NUMBERS (not just "write a report"):
  This is the core idea of RAG-style prompting (from DAO 3022) -
  we don't want the LLM to invent or guess anything. We hand it the
  real, computed evidence directly in the prompt, and its only job
  is to explain that evidence clearly in natural language - not to
  make up new facts. This keeps the output grounded and trustworthy.
"""

import pandas as pd
import requests
import json

# ---------------------------------------------------------------
# STEP 1: LOAD THE FUSED EVENTS FROM MODULE 3
# ---------------------------------------------------------------
events_df = pd.read_csv("../outputs/fusion_test_events.csv")

# Only report on events the LEARNED FUSION model flagged as real -
# these are our highest-confidence, evidence-backed detections.
high_confidence_events = events_df[events_df["learned_fusion_flag"] == True].copy()
print(f"Found {len(high_confidence_events)} high-confidence fused events to report on")

# ---------------------------------------------------------------
# STEP 2: FUNCTION TO CALL THE LOCAL LLM VIA OLLAMA
# ---------------------------------------------------------------
# Ollama runs a local web server (started automatically when you
# install it) on your own machine, listening on port 11434. We send
# it a request the same way we'd call any web API - just pointed at
# our own computer instead of the internet.

OLLAMA_URL = "http://localhost:11434/api/generate"

def ask_llm(prompt):
    response = requests.post(OLLAMA_URL, json={
        "model": "llama3.2:3b",
        "prompt": prompt,
        "stream": False,  # get the full response at once, not word-by-word
    })
    return response.json()["response"]

# ---------------------------------------------------------------
# STEP 3: BUILD A GROUNDED PROMPT PER EVENT AND GENERATE A REPORT
# ---------------------------------------------------------------
reports = []

for idx, event in high_confidence_events.iterrows():
        # STRICTER VERSION (v2) - explicitly forbids the two hallucination
    # patterns we caught in manual review: (1) inventing timestamps/times
    # not present in the data, and (2) speculative threat interpretation
    # (e.g. "military activity", "surveillance") not supported by the
    # numbers alone. This is a direct fix based on reviewing v1 output.
    prompt = f"""You are a defense/space intelligence analyst assistant. Write a short, professional analyst brief (3-4 sentences) based STRICTLY on the following detected event data.

STRICT RULES:
- Do NOT invent a timestamp, date, or time of day - none was provided.
- Do NOT speculate about intent, threat type, or activity type (e.g. do not say "military," "surveillance," "reconnaissance," or similar) - only the sensor scores were provided, not intent.
- Do NOT add any detail not explicitly listed below.
- Only describe the scores, location, and confidence level factually.

Event data:
- Location grid cell: row {event['row']}, column {event['col']} (within the Visakhapatnam Port monitoring area)
- RF anomaly score: {event['rf_score']:.3f} (0-1 scale, higher = stronger anomaly)
- SAR change score: {event['sar_score']:.3f} (0-1 scale, higher = more physical change detected)
- Fusion confidence: HIGH (flagged by both independent sensors, confirmed by learned fusion model)

Write the brief now:"""

    print(f"\nGenerating report for event at row={event['row']}, col={event['col']}...")
    report_text = ask_llm(prompt)

    reports.append({
        "row": event["row"],
        "col": event["col"],
        "rf_score": event["rf_score"],
        "sar_score": event["sar_score"],
        "report": report_text,
    })

# ---------------------------------------------------------------
# STEP 4: SAVE ALL REPORTS TO A READABLE TEXT FILE
# ---------------------------------------------------------------
with open("../outputs/analyst_reports.txt", "w", encoding="utf-8") as f:
    for i, r in enumerate(reports, 1):
        f.write(f"{'='*60}\n")
        f.write(f"REPORT #{i} - Location: row {r['row']}, col {r['col']}\n")
        f.write(f"RF score: {r['rf_score']:.3f} | SAR score: {r['sar_score']:.3f}\n")
        f.write(f"{'-'*60}\n")
        f.write(f"{r['report']}\n\n")

print(f"\nSaved {len(reports)} reports to analyst_reports.txt")
