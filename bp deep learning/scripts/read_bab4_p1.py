import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ch4 = Path("scratch/gemastik/logs/bab4_new.txt").read_text(encoding="utf-8", errors="ignore")
lines = ch4.splitlines()
for i in range(min(180, len(lines))):
    print(f"L{i+1:04d}: {lines[i]}")
