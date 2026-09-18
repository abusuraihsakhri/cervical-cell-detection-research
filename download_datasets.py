"""
Dataset acquisition — Section 3 of AGENT-SPEC-FINAL.md.

Agent action order (per spec):
  1. Download both SIPaKMeD Roboflow mirrors, diff class counts/splits,
     pick the more complete one as primary.
  2. Attempt CRIC access; verify annotation format with a sample file
     BEFORE writing any code that assumes bounding boxes exist.
  3. If CRIC is blocked or unlabeled, fall back to Mendeley LBC
     (Ares has approved this fallback in advance).
  4. Report dataset status back to Ares before starting Phase 2 training.

This script downloads and verifies. It does NOT silently substitute
datasets beyond the pre-approved CRIC -> Mendeley LBC fallback, and it
does NOT proceed to training.

Requires a .env file (gitignored) with:
    ROBOFLOW_API_KEY=your_key_here
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

import config
from _logging_setup import setup_logging

def _require_api_key() -> str:
    try:
        return config.get_roboflow_key()
    except EnvironmentError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(1)


def download_sipakmed_mirror(mirror_key: str, mirror_info: dict, dest_root: Path) -> Path:
    """
    Download one SIPaKMeD Roboflow mirror in YOLOv8 export format.

    Raises RuntimeError (rather than returning an empty directory
    silently) if the download call succeeds but no files actually
    land on disk, or if the project's export type doesn't support
    YOLOv8 bbox format (e.g. a multilabel-classification project).
    """
    from roboflow import Roboflow

    rf = Roboflow(api_key=_require_api_key())
    project = rf.workspace(mirror_info["workspace"]).project(mirror_info["project"])
    version = project.version(mirror_info["version"])

    project_type = getattr(project, "type", None)
    if project_type and project_type not in ("object-detection", "instance-segmentation"):
        raise RuntimeError(
            f"[{mirror_key}] project type is '{project_type}', which cannot export "
            "bounding-box YOLOv8 data (needs 'object-detection'). Skipping this mirror."
        )

    dest = dest_root / mirror_key
    # Clear any stale/partial leftover from a previous failed run.
    if dest.exists():
        shutil.rmtree(dest)
    # NOTE: deliberately do NOT pre-create `dest` here. Roboflow's SDK
    # treats a pre-existing destination directory as "already downloaded"
    # in some versions and skips the actual download/unzip, which is the
    # likely cause of the empty-folder bug (the working ad-hoc diagnostic
    # downloaded into a path that had never existed before). Let
    # version.download() create the directory itself.

    print(f"[{mirror_key}] calling version.download(location={dest})...")
    dataset = version.download("yolov8", location=str(dest))
    print(f"[{mirror_key}] version.download() call returned: {dataset}")
    actual_dir = Path(dataset.location)

    # Verify files actually landed. Roboflow's SDK can return a location
    # object without raising even when the export/unzip hasn't fully
    # flushed to disk yet (observed race on some Windows setups), so
    # check the filesystem directly with a short settle-retry rather
    # than trusting the return value on the first check alone.
    found_files = []
    for attempt in range(5):
        found_files = [f for f in actual_dir.rglob("*") if f.is_file()] if actual_dir.exists() else []
        if found_files:
            break
        time.sleep(1)

    if not found_files:
        # Also check dest in case dataset.location pointed somewhere else
        dest_files = [f for f in dest.rglob("*") if f.is_file()]
        if dest_files:
            print(f"[{mirror_key}] WARNING: dataset.location ({actual_dir}) was empty "
                  f"but files were found under requested dest ({dest}); using dest.")
            actual_dir = dest
            found_files = dest_files
        else:
            raise RuntimeError(
                f"[{mirror_key}] download() returned without error but produced ZERO "
                f"files at either {actual_dir} or {dest} after retrying for 5s. This "
                "mirror may require a different export format, a different version "
                "number, or the workspace/project slug may be wrong. Not treating "
                "this as a valid download."
            )

    print(f"[{mirror_key}] downloaded {len(found_files)} files to {actual_dir}")
    return actual_dir


def summarize_yolo_dataset(dataset_dir: Path) -> dict:
    """Count images/labels per split for a YOLOv8-format dataset dir."""
    summary = {}
    for split in ("train", "valid", "test"):
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        n_images = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        n_labels = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        summary[split] = {"images": n_images, "labels": n_labels}

    data_yaml = dataset_dir / "data.yaml"
    if data_yaml.exists():
        import yaml

        with open(data_yaml) as f:
            y = yaml.safe_load(f)
        summary["classes"] = y.get("names")
        summary["nc"] = y.get("nc")
    return summary


def step1_download_and_diff_sipakmed():
    print("=== Step 1: SIPaKMeD — download both mirrors, diff, pick primary ===")
    _require_api_key()

    summaries = {}
    for key, info in config.ROBOFLOW_SIPAKMED_MIRRORS.items():
        try:
            ds_dir = download_sipakmed_mirror(key, info, config.SIPAKMED_DIR)
            summaries[key] = {"dir": str(ds_dir), **summarize_yolo_dataset(ds_dir)}
        except Exception as e:  # noqa: BLE001 — report and continue to the other mirror
            print(f"[{key}] FAILED: {e}", file=sys.stderr)
            summaries[key] = {"error": str(e)}

    report_path = config.RESULTS_DIR / "sipakmed_mirror_diff.json"
    report_path.write_text(json.dumps(summaries, indent=2))
    print(f"Mirror comparison written to {report_path}")

    # Pick the more complete mirror (most total images across splits).
    def total_images(s):
        if "error" in s:
            return -1
        return sum(v.get("images", 0) for k, v in s.items() if isinstance(v, dict))

    viable = {k: v for k, v in summaries.items() if "error" not in v}
    if not viable:
        print("ERROR: both SIPaKMeD mirrors failed to download. Stopping — report to Ares.", file=sys.stderr)
        sys.exit(1)

    primary_key = max(viable, key=lambda k: total_images(viable[k]))
    print(f"Selected '{primary_key}' as primary SIPaKMeD source ({total_images(viable[primary_key])} images).")

    primary_dir = Path(viable[primary_key]["dir"])
    primary_link = config.SIPAKMED_DIR / "primary"
    if primary_link.exists() or primary_link.is_symlink():
        if primary_link.is_symlink() or primary_link.is_file():
            primary_link.unlink()
        else:
            shutil.rmtree(primary_link)
    try:
        primary_link.symlink_to(primary_dir, target_is_directory=True)
    except OSError:
        # Symlinks can fail on some Windows configs without admin/dev-mode;
        # fall back to a plain copy so downstream code has a stable path.
        shutil.copytree(primary_dir, primary_link)

    print(f"Primary SIPaKMeD dataset available at: {primary_link}")

    # Some Roboflow exports (observed with mirror_a / ik-zu-quan-o9tdm)
    # put every image in train/ with no valid/ or test/ on disk, despite
    # data.yaml claiming those paths exist. Every downstream training and
    # evaluation script needs a real held-out val split, so create one now
    # if it's missing, rather than let training/eval silently fail later.
    val_dir = primary_dir / "valid" / "images"
    has_val = val_dir.exists() and any(val_dir.iterdir())
    if not has_val:
        print(f"\n[{primary_key}] No valid/images found on disk — creating a "
              "stratified train/val split (85/15) now via prepare_splits.py logic.")
        import prepare_splits
        import sys as _sys
        old_argv = _sys.argv
        try:
            _sys.argv = ["prepare_splits.py", "--dataset-dir", str(primary_dir)]
            prepare_splits.main()
        finally:
            _sys.argv = old_argv

        # Refresh the summary for the report — the version computed above
        # was taken before the split ran and would otherwise show a stale
        # valid:0 even though the split succeeded (cosmetic bug, but
        # confusing to read in dataset_status_report.json).
        summaries[primary_key] = {"dir": str(primary_dir), **summarize_yolo_dataset(primary_dir)}

    return primary_key, summaries


def step2_check_apcdata() -> bool:
    """
    Section 9's original CRIC-verification open item is now RESOLVED via
    web research (Aug 2026), not left as a runtime guess:

      - CRIC (database.cric.com.br): confirmed point-click cell-center
        coordinates, NOT bounding boxes. Not usable for YOLO detection
        without inventing a synthetic box radius per point.
      - Mendeley LBC (zddtpgzv63): confirmed classification-folder labels
        only, no bounding boxes.
      - APCData (ytd568rh3p, DOI 10.17632/ytd568rh3p.1): confirmed genuine
        YOLO .txt bounding-box annotations, ready to use. Selected as the
        Phase 2 second dataset. See config.PHASE2_DATASET_SOURCES for the
        full citation trail on all four candidates considered.

    Mendeley Data does not expose a documented public download API the
    way Roboflow does (no equivalent of the roboflow pip package here),
    so this step cannot silently fetch it the way SIPaKMeD is fetched.
    It checks whether Ares has already placed the manually-downloaded
    APCData_YOLO folder under data/apcdata/, and if not, prints the exact
    manual step needed rather than proceeding on an assumption.
    """
    print("=== Step 2: APCData (Phase 2 dataset — resolved via research, see config.py) ===")
    apcdata_yolo_dir = config.APCDATA_DIR / "APCData_YOLO"
    if apcdata_yolo_dir.exists() and any(apcdata_yolo_dir.rglob("*.txt")):
        n_labels = len(list(apcdata_yolo_dir.rglob("*.txt")))
        print(f"APCData_YOLO found with {n_labels} label files under {apcdata_yolo_dir}.")
        return True

    print(
        "APCData not found on disk yet. Manual step required (Mendeley Data has no "
        "documented public download API):\n"
        "  1. Visit https://data.mendeley.com/datasets/ytd568rh3p/1\n"
        "  2. Download the dataset (APCData_YOLO + APCData_points folders)\n"
        f"  3. Extract so APCData_YOLO ends up at: {apcdata_yolo_dir}\n"
        "  4. Re-run this script.\n"
        "STOPPING here rather than proceeding on an assumption that the data exists."
    )
    return False


def step2_verify_cric_annotations() -> bool:
    """
    Kept for reference / historical record only — CRIC's annotation format
    is now resolved (see step2_check_apcdata's docstring and
    config.PHASE2_DATASET_SOURCES): confirmed point-based, not bbox-based.
    This function is no longer called by main() but is left in place in
    case Ares wants to revisit CRIC-with-synthetic-boxes as a later,
    explicitly-chosen option rather than deleting the working verification
    logic outright.
    """
    print("=== [reference only] CRIC bbox check — already resolved as 'not usable as-is' ===")
    if not config.CRIC_DIR.exists() or not any(config.CRIC_DIR.iterdir()):
        print(f"CRIC_DIR ({config.CRIC_DIR}) is empty — nothing to check.")
        return False

    sample_files = list(config.CRIC_DIR.rglob("*"))[:200]
    found_bbox_evidence = False
    for f in sample_files:
        if f.suffix.lower() == ".json":
            try:
                data = json.loads(f.read_text())
                blob = json.dumps(data)
                if '"bbox"' in blob or "bounding_box" in blob:
                    found_bbox_evidence = True
                    break
            except Exception:
                continue
        elif f.suffix.lower() == ".txt":
            try:
                line = f.read_text().splitlines()[0]
                if len(line.split()) >= 5:
                    found_bbox_evidence = True
                    break
            except Exception:
                continue
        elif f.suffix.lower() == ".xml":
            if "<bndbox>" in f.read_text(errors="ignore"):
                found_bbox_evidence = True
                break

    if found_bbox_evidence:
        print("CRIC: bbox-level annotation evidence FOUND. Phase 2 with CRIC may proceed — "
              "but still have a human spot-check a few samples before full training.")
    else:
        print(
            "CRIC: no bbox-level annotation evidence found in the sample "
            "(only image-level metadata, or format unrecognized).\n"
            "Per spec Section 3/9: do NOT silently substitute — this script will "
            "fall back to Mendeley LBC only because Ares pre-approved that fallback."
        )
    return found_bbox_evidence


def step4_report_status(sipakmed_summaries: dict, apcdata_ok: bool):
    print("=== Step 4: Dataset status report ===")
    status = {
        "sipakmed": sipakmed_summaries,
        "phase2_dataset": config.PHASE2_DATASET,
        "phase2_dataset_ready": apcdata_ok,
        "phase2_dataset_sources_considered": config.PHASE2_DATASET_SOURCES,
    }
    report_path = config.RESULTS_DIR / "dataset_status_report.json"
    report_path.write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))
    print(f"\nFull report written to {report_path}")
    print(
        "\nReminder: per spec, Phase 2 training should not start until this "
        "report has been reviewed."
    )
    if not apcdata_ok:
        print(
            "\nAPCData is not yet on disk — Phase 1 (SIPaKMeD-only) training can "
            "proceed now via training_pipeline.py; Phase 2 cross-dataset eval "
            "will wait until APCData is manually downloaded (see step2 output above)."
        )


def main():
    import argparse

    setup_logging("download_datasets")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-apcdata-only",
        action="store_true",
        help="Skip SIPaKMeD download; only check whether APCData is present.",
    )
    args = parser.parse_args()

    if args.check_apcdata_only:
        step2_check_apcdata()
        return

    _, sipakmed_summaries = step1_download_and_diff_sipakmed()
    apcdata_ok = step2_check_apcdata()
    step4_report_status(sipakmed_summaries, apcdata_ok)


if __name__ == "__main__":
    main()
