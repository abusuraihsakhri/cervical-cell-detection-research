"""
Converts the HMCHH-TCT-CellDet dataset (figshare DOI 10.6084/m9.figshare.27901206)
from its native custom XML annotation format into YOLOv8-ready train/valid
folders, matching the pattern used for SIPaKMeD and APCData.

Verified against real extracted files (not assumed) before writing this:
  - 8,037 images (JPEGImages/, actually .png despite the folder name) and
    8,037 matching XML files (Annotations_extracted/anno_copy/), 1:1 by stem.
  - XML schema is custom, not Pascal VOC: <doc><outputs><object><item>
    <name>...</name><bndbox><xmin>/<ymin>/<xmax>/<ymax></bndbox></item>
    </object></outputs><size><width>/<height></size></doc>
  - Single class observed across a 200-file sample: "异常" (Chinese for
    "abnormal") — matches the dataset's documented binary abnormal/normal
    design where only abnormal cells get bounding boxes; there is no
    separate "normal" box class. Mapped here to class index 0 = "Abnormal".
  - Every sampled file (500/500) has >=1 object — this appears to be an
    abnormal-cell-only detection set (every image was selected because it
    contains at least one flagged cell), not a mixed normal/abnormal set
    with true negatives. Documented as an assumption to flag to Ares,
    not silently treated as fact — if a normal-only class matters for
    your use, this dataset alone will not provide true-negative images.
  - Boxes are pixel xmin/ymin/xmax/ymax against <size><width>/<height>
    (observed 2048x2048), NOT the image's actual on-disk resolution in
    every case necessarily — this script reads each image's real
    dimensions via PIL rather than trusting the XML's <size> block, to
    avoid silently mis-normalizing boxes if any image differs.

Usage:
    python prepare_hmchh.py \
        --images-dir "D:\\pap_model\\JPEGImages_extracted\\JPEGImages" \
        --annotations-dir "D:\\pap_model\\Annotations_extracted\\anno_copy" \
        --out-dir "D:\\pap_model\\HMCHH_YOLO_prepared" \
        --val-fraction 0.15
"""

import argparse
import random
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from _logging_setup import setup_logging

CLASS_NAMES = ["Abnormal"]  # single class per verified sample; see docstring


def parse_xml(xml_path: Path):
    """Returns list of (xmin, ymin, xmax, ymax) pixel boxes; class is always 0 (Abnormal)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for item in root.iter("item"):
        bb = item.find("bndbox")
        if bb is None:
            continue
        try:
            xmin = float(bb.findtext("xmin"))
            ymin = float(bb.findtext("ymin"))
            xmax = float(bb.findtext("xmax"))
            ymax = float(bb.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        boxes.append((xmin, ymin, xmax, ymax))
    return boxes


def to_yolo_line(box, img_w, img_h, class_idx=0):
    xmin, ymin, xmax, ymax = box
    # Clamp to image bounds defensively — a source annotation slightly out
    # of range would otherwise produce an invalid (negative or >1) YOLO box.
    xmin, xmax = max(0, min(xmin, img_w)), max(0, min(xmax, img_w))
    ymin, ymax = max(0, min(ymin, img_h)), max(0, min(ymax, img_h))
    cx = (xmin + xmax) / 2 / img_w
    cy = (ymin + ymax) / 2 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return f"{class_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def find_matching_image(images_dir: Path, xml_stem: str):
    for ext in (".png", ".jpg", ".jpeg"):
        p = images_dir / (xml_stem + ext)
        if p.exists():
            return p
    return None


def main():
    setup_logging("prepare_hmchh")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--annotations-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        from PIL import Image
    except ImportError:
        print("ERROR: Pillow is required (pip install Pillow) to read real "
              "image dimensions rather than trust the XML's <size> block.", file=sys.stderr)
        sys.exit(1)

    images_dir = Path(args.images_dir)
    ann_dir = Path(args.annotations_dir)
    out_dir = Path(args.out_dir)

    xml_files = list(ann_dir.glob("*.xml"))
    print(f"Found {len(xml_files)} annotation files.")

    pairs = []
    unmatched = 0
    empty_boxes = 0
    for xml_path in xml_files:
        img_path = find_matching_image(images_dir, xml_path.stem)
        if img_path is None:
            unmatched += 1
            continue
        boxes = parse_xml(xml_path)
        if not boxes:
            empty_boxes += 1
            continue
        pairs.append((img_path, boxes))

    print(f"Matched {len(pairs)} image/annotation pairs with >=1 box.")
    if unmatched:
        print(f"WARNING: {unmatched} XML files had no matching image file — excluded.")
    if empty_boxes:
        print(f"NOTE: {empty_boxes} XML files parsed with zero boxes — excluded "
              "(if you expected normal/negative images here, they are not present "
              "in this cut of the dataset; see script docstring).")

    if not pairs:
        print("ERROR: zero usable pairs — check --images-dir/--annotations-dir paths.", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)
    random.shuffle(pairs)
    n_val = round(len(pairs) * args.val_fraction)
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    print(f"Split: {len(train_pairs)} train, {len(val_pairs)} valid.")

    for split in ("train", "valid"):
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    def write_split(split_name, split_pairs):
        n_boxes_total = 0
        n_skipped_dim_errors = 0
        n_already_done = 0
        for i, (img_path, boxes) in enumerate(split_pairs):
            dest_img = out_dir / split_name / "images" / img_path.name
            dest_lbl = out_dir / split_name / "labels" / (img_path.stem + ".txt")

            # Resumable: if a previous run already moved this image and wrote
            # its label, skip it rather than fail (source image is gone after
            # a move) or redo work. Lets a run interrupted by e.g. running out
            # of disk space partway through be restarted without starting over.
            if dest_img.exists() and dest_lbl.exists():
                n_already_done += 1
                continue
            if not img_path.exists():
                # Already moved by a prior run but label wasn't written (rare
                # partial-failure case) — nothing we can recover here without
                # the original image; skip and note it.
                n_skipped_dim_errors += 1
                continue

            try:
                with Image.open(img_path) as im:
                    img_w, img_h = im.size
            except Exception:
                n_skipped_dim_errors += 1
                continue

            lines = [to_yolo_line(b, img_w, img_h) for b in boxes]
            n_boxes_total += len(lines)

            # MOVE, not copy — this dataset is ~41GB; copying would need
            # another ~41GB free (confirmed: a copy-based run filled the
            # disk and crashed after 500 images). Moving relocates the file
            # with no duplication, since the flat JPEGImages_extracted
            # source folder isn't needed once every image has a home in
            # train/ or valid/.
            shutil.move(str(img_path), str(dest_img))
            dest_lbl.write_text("\n".join(lines) + "\n")

            if (i + 1) % 500 == 0:
                print(f"  [{split_name}] {i + 1}/{len(split_pairs)} processed...")

        if n_already_done:
            print(f"[{split_name}] {n_already_done} already done from a prior run — skipped.")
        if n_skipped_dim_errors:
            print(f"WARNING: {n_skipped_dim_errors} images in {split_name} could not "
                  "be opened/found and were skipped.")
        print(f"[{split_name}] done: {len(split_pairs) - n_skipped_dim_errors} images, "
              f"{n_boxes_total} total boxes (this run).")

    write_split("train", train_pairs)
    write_split("valid", val_pairs)

    data_yaml = out_dir / "data.yaml"
    import yaml
    with open(data_yaml, "w") as f:
        yaml.safe_dump(
            {"names": CLASS_NAMES, "nc": len(CLASS_NAMES), "train": "train/images", "val": "valid/images"},
            f, sort_keys=False,
        )
    print(f"\ndata.yaml written to {data_yaml}")
    print(f"Use this as --phase2-data in evaluation/cross_dataset_eval.py:\n    {data_yaml}")
    print("\nNOTE: this is a BINARY (Abnormal-only, single-class) dataset — SIPaKMeD's "
          "5-class model output must be rolled up via ROLLUP_5_TO_3 -> ROLLUP_3_TO_2 in "
          "config.py before a fair comparison against this dataset's labels.")


if __name__ == "__main__":
    main()
