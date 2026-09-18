"""
Builds a proper YOLOv8 train/valid dataset directory for APCData from the
raw Mendeley download, which does NOT ship as a ready-to-train YOLO folder
despite being labeled "APCData_YOLO":

  APCData cervical cytology cells/
    APCData_YOLO/
      classes.txt          (NILM, ASCUS, ASCH, LSIL, HSIL, SCC)
      labels/*.txt          (417 files, Roboflow-style names with .rf.<hash> suffix)
    APCData_points/
      images/*.jpg          (425 files, the actual images — YOLO folder has none)

Label and image filenames don't match exactly (dashes vs spaces, extra
.rf.<hash> suffix on labels), so this script matches them by a normalized
key (lowercased, non-alphanumeric stripped) rather than assuming identical
stems — a real filename mismatch found on inspection, not a guess.

Produces:
  data/apcdata/APCData_YOLO_prepared/
    train/images, train/labels
    valid/images, valid/labels
    data.yaml

Usage:
    python prepare_apcdata.py --raw-dir "data/apcdata/APCData cervical cytology cells" --val-fraction 0.15
"""

import argparse
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from _logging_setup import setup_logging

CLASS_NAMES = ["NILM", "ASCUS", "ASCH", "LSIL", "HSIL", "SCC"]


def normalize_key(stem: str) -> str:
    """
    Strip Roboflow's '_jpg.rf.<hash>' / '_png.rf.<hash>' suffix pattern
    (labels are named like '<original>_jpg.rf.<hash>.txt' while the
    matching image is just '<original>.jpg' with no such suffix — verified
    against real APCData filenames before writing this), then normalize
    to alnum-only lowercase so spaces/dashes/parens don't block matching.
    """
    stem = re.sub(r"\.rf\.[0-9a-f]{16,}$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_(jpg|jpeg|png)$", "", stem, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]", "", stem.lower())


def main():
    setup_logging("prepare_apcdata")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, help='e.g. "data/apcdata/APCData cervical cytology cells"')
    parser.add_argument("--out-dir", default=None, help="default: <raw-dir's parent>/APCData_YOLO_prepared")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    labels_dir = raw_dir / "APCData_YOLO" / "labels"
    images_dir = raw_dir / "APCData_points" / "images"
    out_dir = Path(args.out_dir).resolve() if args.out_dir else raw_dir.parent / "APCData_YOLO_prepared"

    if not labels_dir.exists():
        print(f"ERROR: {labels_dir} not found.", file=sys.stderr)
        sys.exit(1)
    if not images_dir.exists():
        print(f"ERROR: {images_dir} not found.", file=sys.stderr)
        sys.exit(1)

    label_files = list(labels_dir.glob("*.txt"))
    image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    print(f"Found {len(label_files)} label files, {len(image_files)} image files.")

    image_by_key = {}
    for img in image_files:
        key = normalize_key(img.stem)
        if key in image_by_key:
            print(f"WARNING: duplicate normalized key '{key}' for images "
                  f"{image_by_key[key].name} and {img.name} — keeping first.")
            continue
        image_by_key[key] = img

    matched = []
    unmatched_labels = []
    for lbl in label_files:
        key = normalize_key(lbl.stem)
        img = image_by_key.get(key)
        if img is None:
            unmatched_labels.append(lbl)
        else:
            matched.append((img, lbl))

    print(f"Matched {len(matched)} image/label pairs by normalized filename key.")
    if unmatched_labels:
        print(f"WARNING: {len(unmatched_labels)} label files had no matching image — "
              f"excluded. Sample unmatched: {[f.name for f in unmatched_labels[:5]]}")
    unlabeled_images = len(image_files) - len(matched)
    if unlabeled_images > 0:
        print(f"NOTE: {unlabeled_images} images have no label match (expected — "
              f"425 images vs 417 labels per the source dataset's own counts).")

    if not matched:
        print("ERROR: zero image/label pairs matched — check the normalization logic "
              "against actual filenames before proceeding.", file=sys.stderr)
        sys.exit(1)

    # Class-stratified split, same approach as prepare_splits.py for SIPaKMeD.
    by_class = defaultdict(list)
    for img, lbl in matched:
        lines = lbl.read_text().splitlines()
        first_class = int(lines[0].split()[0]) if lines else -1
        by_class[first_class].append((img, lbl))

    import random
    random.seed(args.seed)

    for split in ("train", "valid"):
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    train_count, val_count = 0, 0
    for cls, pairs in sorted(by_class.items()):
        random.shuffle(pairs)
        n_val = max(1, round(len(pairs) * args.val_fraction)) if len(pairs) > 1 else 0
        val_pairs = pairs[:n_val]
        train_pairs = pairs[n_val:]
        cls_name = CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else f"class{cls}"
        print(f"  {cls_name}: {len(pairs)} total -> {len(train_pairs)} train, {len(val_pairs)} val")

        for img, lbl in train_pairs:
            shutil.copy2(img, out_dir / "train" / "images" / img.name)
            shutil.copy2(lbl, out_dir / "train" / "labels" / (img.stem + ".txt"))
            train_count += 1
        for img, lbl in val_pairs:
            shutil.copy2(img, out_dir / "valid" / "images" / img.name)
            shutil.copy2(lbl, out_dir / "valid" / "labels" / (img.stem + ".txt"))
            val_count += 1

    data_yaml = out_dir / "data.yaml"
    import yaml
    with open(data_yaml, "w") as f:
        yaml.safe_dump(
            {"names": CLASS_NAMES, "nc": len(CLASS_NAMES), "train": "train/images", "val": "valid/images"},
            f, sort_keys=False,
        )

    print(f"\nDone. train={train_count} images, valid={val_count} images.")
    print(f"data.yaml written to {data_yaml}")
    print(f"\nUse this as --phase2-data in evaluation/cross_dataset_eval.py:\n    {data_yaml}")


if __name__ == "__main__":
    main()
