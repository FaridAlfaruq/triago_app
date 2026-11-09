"""
Monitor Kaggle GPU job and pull artifacts on completion.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

KERNEL_ID = "faridalfaruq/triago-bpnet-v5-3-precision-gpu-training"
KAGGLE_EXE = Path(__file__).resolve().parents[3] / ".venv" / "Scripts" / "kaggle.exe"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def run_cmd(cmd: list) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    return res.stdout.strip()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Monitoring Kaggle kernel: {KERNEL_ID}")

    while True:
        status_out = run_cmd([str(KAGGLE_EXE), "kernels", "status", KERNEL_ID])
        print(f"[{time.strftime('%X')}] {status_out}")

        if "COMPLETE" in status_out.upper():
            print("Kernel finished successfully! Pulling output...")
            pull_out = run_cmd([str(KAGGLE_EXE), "kernels", "output", KERNEL_ID, "-p", str(OUTPUT_DIR)])
            print(pull_out)
            break
        elif "ERROR" in status_out.upper() or "CANCELLED" in status_out.upper():
            print(f"Kernel failed with status: {status_out}")
            run_cmd([str(KAGGLE_EXE), "kernels", "output", KERNEL_ID, "-p", str(OUTPUT_DIR)])
            sys.exit(1)

        time.sleep(15)

    print("Completed. Output directory contents:")
    for f in OUTPUT_DIR.glob("*"):
        print(f" - {f.name} ({f.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
