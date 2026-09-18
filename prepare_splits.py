"""
Fix for a real problem found on the first successful SIPaKMeD download:
the Roboflow 'ik-zu-quan-o9tdm/sipakmed-ioflq' v1 export puts ALL 1298
images into train/ only — valid/ and test/ don't exist on disk even
though data.yaml claims them. Every downstream script (training,
confusion matrix, cross-dataset eval, calibration, PR curve) needs a
real held-out validation split to report honest numbers on unseen
data — evaluating on the same images the model trained on would make
every metric in Section 6 meaningless.

This script creates a stratified train/val split (default 85/15) from
the train/ folder, moving (not copying, to avoid doubling disk usage)
a per-class-balanced sample into a new valid/ folder, and writes a
corrected data.yaml pointing train -> train/images, val -> valid/images
(test is left unset since no test images exist for this mirror).

Idempotent: if valid/ already has files, does nothing unless --force
is passed (to avoid accidentally re-splitting and shrinking train/ on
a re-run).

Usage:
    python prepare_splits.py --dataset-dir data/sipakmed/mirror_a --val-fraction 0.15
"""

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from _logging_setup import setup_logging


def class_of_label_file(label_path: Path):
    """Return the set of class ids present in a YOLO label file (for stratification by dominant class)."""
    if not label_path.exists():
        return []
    classes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if parts:
            classes.append(int(parts[0]))
    return classes


def main():
    setup_logging("prepare_splits")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, help="e.g. data/sipakmed/mirror_a")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Re-split even if valid/ already has files")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    train_img_dir = dataset_dir / "train" / "images"
    train_lbl_dir = dataset_dir / "train" / "labels"
    val_img_dir = dataset_dir / "valid" / "images"
    val_lbl_dir = dataset_dir / "valid" / "labels"

    if not train_img_dir.exists():
        print(f"ERROR: {train_img_dir} does not exist. Nothing to split.", file=sys.stderr)
        sys.exit(1)

    existing_val = list(val_img_dir.glob("*")) if val_img_dir.exists() else []
    if existing_val and not args.force:
        print(f"valid/images already has {len(existing_val)} files — nothing to do. "
              "Pass --force to re-split (will pull more files out of train/).")
        return

    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png")) + list(train_img_dir.glob("*.jpeg")))
    print(f"Found {len(img_files)} images in train/.")

    # Group by dominant (first-listed) class in each label file for a
    # simple class-stratified split, so val isn't accidentally skewed
    # toward/away from any one of the 5 classes.
    by_class = defaultdict(list)
    unlabeled = []
    for img in img_files:
        lbl = train_lbl_dir / (img.stem + ".txt")
        classes = class_of_label_file(lbl)
        if classes:
            by_class[classes[0]].append(img)
        else:
            unlabeled.append(img)

    if unlabeled:
        print(f"WARNING: {len(unlabeled)} images have no/empty label file — "
              "excluding them from the val split (they'll stay in train, which "
              "is also questionable; consider whether these should be dropped "
              "entirely once you inspect a few).")

    random.seed(args.seed)
    moved = 0
    for cls, imgs in sorted(by_class.items()):
        random.shuffle(imgs)
        n_val = max(1, round(len(imgs) * args.val_fraction))
        val_imgs = imgs[:n_val]
        print(f"  class {cls}: {len(imgs)} total -> moving {len(val_imgs)} to val")
        for img in val_imgs:
            lbl = train_lbl_dir / (img.stem + ".txt")
            shutil.move(str(img), str(val_img_dir / img.name))
            if lbl.exists():
                shutil.move(str(lbl), str(val_lbl_dir / lbl.name))
            moved += 1

    print(f"\nMoved {moved} images (+ labels) from train/ to valid/.")

    # Rewrite data.yaml to point at what actually exists on disk now.
    # Drop the `test` key entirely rather than leave it pointing at a
    # directory that doesn't exist and never will for this mirror.
    data_yaml_path = dataset_dir / "data.yaml"
    class_names = None
    if data_yaml_path.exists():
        import yaml
        with open(data_yaml_path) as f:
            existing = yaml.safe_load(f)
        class_names = existing.get("names")

    if class_names is None:
        print("WARNING: could not read class names from existing data.yaml; "
              "leaving names out of the rewritten file — check it manually.")
        class_names = []

    new_yaml = {
        "names": class_names,
        "nc": len(class_names),
        "train": "train/images",
        "val": "valid/images",
    }
    import yaml
    with open(data_yaml_path, "w") as f:
        yaml.safe_dump(new_yaml, f, sort_keys=False)

    print(f"Rewrote {data_yaml_path} with train/valid paths matching what's actually on disk (no test/).")
    print(f"\nFinal counts: train={len(list(train_img_dir.glob('*')))} images, "
          f"valid={len(list(val_img_dir.glob('*')))} images.")


if __name__ == "__main__":
    main()
