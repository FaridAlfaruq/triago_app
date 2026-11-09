import sys
import re
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

for ch in [1, 2, 3, 5]:
    ch_text = Path(f"scratch/gemastik/logs/bab{ch}_new.txt").read_text(encoding="utf-8", errors="ignore")
    print(f"\n================ CHAPTER {ch} REF AUDIT ================")
    
    # Check Gambar
    g_calls = re.findall(r"(Gambar\s+\d+\.\d+)", ch_text)
    g_caps = re.findall(r"(Gambar\s+\d+\.\d+[^\n]+)", ch_text)
    print("Gambar Callouts:", g_calls)
    print("Gambar Captions:", g_caps)
    
    # Check Tabel
    t_calls = re.findall(r"(Tabel\s+\d+\.\d+)", ch_text)
    t_caps = re.findall(r"(Tabel\s+\d+\.\d+[^\n]+)", ch_text)
    print("Tabel Callouts:", t_calls)
    print("Tabel Captions:", t_caps)
    
    # Check Persamaan
    p_calls = re.findall(r"(Persamaan\s*(?:\([0-9\.]+\)|[0-9\.]+))", ch_text, re.IGNORECASE)
    print("Persamaan Callouts:", p_calls)
