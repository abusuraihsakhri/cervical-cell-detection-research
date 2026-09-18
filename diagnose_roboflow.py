"""
One-off diagnostic: verifies Roboflow auth + workspace/project access and
prints the FULL exception if download fails, since the silent-empty-folder
failure in download_datasets.py needs to be seen to be fixed.

Run this directly:
    python diagnose_roboflow.py
"""
import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

import config
from _logging_setup import setup_logging

setup_logging("diagnose_roboflow")

try:
    key = config.get_roboflow_key()
    print(f"ROBOFLOW_API_KEY loaded securely: yes, length={len(key)}")
except EnvironmentError as err:
    print(f"ROBOFLOW_API_KEY check failed: {err}")
    sys.exit(1)

try:
    import roboflow
    from roboflow import Roboflow
    print("roboflow import OK, version:", getattr(roboflow, "__version__", "unknown"))
except Exception as e:
    print("FAILED to import roboflow:", e)
    sys.exit(1)

try:
    rf = Roboflow(api_key=key)
    print("Roboflow() client created OK")
except Exception as e:
    print("FAILED creating Roboflow client:")
    traceback.print_exc()
    sys.exit(1)

for ws, proj, ver in [
    ("ik-zu-quan-o9tdm", "sipakmed-ioflq", 1),
    ("sipakmed", "sipakmed-t9emb", 1),
]:
    print(f"\n--- Trying {ws}/{proj} v{ver} ---")
    try:
        p = rf.workspace(ws).project(proj)
        print("  project() OK:", p)
        v = p.version(ver)
        print("  version() OK:", v)
        dest = str(Path(__file__).resolve().parent / "data" / f"diag_{proj}")
        print("  requesting download to:", dest)
        d = v.download("yolov8", location=dest)
        print("  download() returned, d.location:", d.location)
        import glob
        for check_dir in {d.location, dest}:
            files = glob.glob(f"{check_dir}/**/*", recursive=True)
            print(f"  files found under {check_dir}: {len(files)}")
            if files[:5]:
                print("  sample:", files[:5])
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
