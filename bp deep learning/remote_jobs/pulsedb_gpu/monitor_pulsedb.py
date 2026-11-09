import subprocess, time, sys
from pathlib import Path

KERNEL_ID = "faridalfaruq/pulsedb-regularized-bpnet-training"
KAGGLE_EXE = Path(__file__).resolve().parents[3] / ".venv" / "Scripts" / "kaggle.exe"
OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Monitoring Kaggle GPU kernel: {KERNEL_ID}")
while True:
    res = subprocess.run([str(KAGGLE_EXE), "kernels", "status", KERNEL_ID], capture_output=True, text=True)
    status = res.stdout.strip()
    print(f"[{time.strftime('%X')}] {status}")
    if "COMPLETE" in status.upper():
        print("Kernel completed successfully! Pulling artifacts...")
        subprocess.run([str(KAGGLE_EXE), "kernels", "output", KERNEL_ID, "-p", str(OUT_DIR)])
        break
    elif "ERROR" in status.upper() or "CANCELLED" in status.upper():
        print(f"Kernel failed: {status}")
        subprocess.run([str(KAGGLE_EXE), "kernels", "output", KERNEL_ID, "-p", str(OUT_DIR)])
        sys.exit(1)
    time.sleep(15)

print("Finished monitoring. Output files:")
for f in OUT_DIR.glob("**/*"):
    if f.is_file():
        print(f" - {f.name} ({f.stat().st_size} bytes)")
