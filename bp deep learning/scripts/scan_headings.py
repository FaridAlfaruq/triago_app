from pathlib import Path
import re

text = Path("scratch/gemastik/logs/proposal_extracted_text.txt").read_text(encoding="utf-8")

# Let's inspect headings / BAB / Section titles
print("=== HEADINGS FOUND ===")
headings = re.findall(r"(?:BAB\s+[IVX]+|[0-9]+\.[0-9]+(?:\.[0-9]+)?\s+[^\n]+)", text)
for h in headings[:40]:
    print(h)

# Search for specific keywords: Tekanan Darah, Blood Pressure, Suhu, Model, XGBoost, etc.
keywords = ["tekanan darah", "blood pressure", "suhu", "temperature", "spo2", "respirasi", "pernapasan", "xgboost", "triage", "triase", "persamaan", "rumus", "vitaldb", "mimic"]

print("\n=== KEYWORD COUNTS ===")
for kw in keywords:
    matches = len(re.findall(re.escape(kw), text, re.IGNORECASE))
    print(f"{kw}: {matches}")
