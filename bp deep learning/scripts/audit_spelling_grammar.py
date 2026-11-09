import sys
import re
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

text = Path("scratch/gemastik/logs/proposal_extracted_text.txt").read_text(encoding="utf-8", errors="ignore")

# List of suspicious patterns or known typos
suspicious = [
    "melakuakan", "meskipul", "perforrmma", "prakti ", "prakti\n", "performa", "praktis",
    "dimana", "di mana", "disamping", "di samping", "merubah", "mengubah",
    "kualitet", "kualitas", "kwalitas", "standard", "standar",
    "aktifitas", "aktivitas", "efektifitas", "efektivitas", "frekwensi", "frekuensi",
    "analisa", "analisis", "metoda", "metode", "sistim", "sistem",
    "jadual", "jadwal", "antri", "antre", "antrean", "antrian",
    "resiko", "risiko", "prosentase", "persentase", "obyektif", "objektif",
    "subyektif", "subjektif", "detil", "detail", "hirarki", "hierarki",
    "sinkronisasi", "sinkron", "kontinyu", "kontinu", "kompleks", "komplek",
    "konkrit", "konkret", "respon", "respons", "pasien", "akurat", "akurasi",
    "downsample", "downsampling", "detrending", "filtering", "ground", "real-time",
    "reverse proxy", "throughput", "edge computing", "cuffless", "cloud", "backend",
    "database", "dashboard", "microcontroller", "ground truth", "over-triage", "under-triage"
]

print("=== SEARCHING SUSPICIOUS / TARGET WORDS ===")
for word in suspicious:
    matches = list(re.finditer(r"\b" + re.escape(word) + r"\b", text, re.IGNORECASE))
    if matches:
        print(f"Word: '{word}' -> Found {len(matches)} times")
        # Print snippet of first 2 occurrences
        for m in matches[:2]:
            s = max(0, m.start() - 40)
            e = min(len(text), m.end() + 40)
            print(f"   [{m.start()}]: ...{text[s:e].replace(chr(10), ' ')}...")
