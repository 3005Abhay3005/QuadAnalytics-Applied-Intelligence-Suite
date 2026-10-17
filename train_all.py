"""
train_all.py — Train all 4 ML models in sequence with timestamped logging.
Run this once before starting the server.
Usage: python train_all.py
"""

import subprocess
import sys
import os
import time
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))

models = [
    ("Diabetes (SVM)",               "models/diabetes/train.py"),
    ("Breast Cancer (Random Forest)", "models/breast_cancer/train.py"),
    ("Car Price (Random Forest)",     "models/car_price/train.py"),
    ("IPL Score (Random Forest)",     "models/ipl_score/train.py"),
]


def ts():
    """Return a short timestamp string."""
    return datetime.now().strftime("%H:%M:%S")


print("=" * 62)
print(f"  ML Model Training Suite  —  started at {ts()}")
print("=" * 62)

total_start = time.time()
results = []

for name, path in models:
    full_path = os.path.join(BASE, path)
    print(f"\n[{ts()}] ▶  {name}")
    step_start = time.time()

    result = subprocess.run(
        [sys.executable, full_path],
        capture_output=True, text=True, errors="replace", cwd=BASE
    )
    elapsed = time.time() - step_start

    if result.returncode == 0:
        status = "✓ SUCCESS"
        # Print last 3 meaningful stdout lines
        lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
        for l in lines[-3:]:
            print(f"   {l}")
    else:
        status = "✗ FAILED"
        print(result.stderr[-600:] if result.stderr else "(no stderr)")

    print(f"   {status}  ({elapsed:.1f}s)")
    results.append((name, status, elapsed))

total_elapsed = time.time() - total_start

print("\n" + "=" * 62)
print(f"  Summary  —  finished at {ts()}  (total: {total_elapsed:.1f}s)")
print("=" * 62)
for name, status, elapsed in results:
    print(f"  {status:12s}  {name}  ({elapsed:.1f}s)")
print()

failed = [r for r in results if "FAILED" in r[1]]
if not failed:
    print("  All models trained successfully!")
    print("  Start the server with:  python run.py")
else:
    print(f"  ⚠  {len(failed)} model(s) failed — check errors above.")
print("=" * 62)
