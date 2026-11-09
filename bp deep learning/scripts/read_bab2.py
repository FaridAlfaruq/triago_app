import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

text = Path("scratch/gemastik/logs/bab2_new.txt").read_text(encoding="utf-8", errors="ignore")
lines = text.splitlines()
for i, line in enumerate(lines, 1):
    print(f"L{i:04d}: {line}")
