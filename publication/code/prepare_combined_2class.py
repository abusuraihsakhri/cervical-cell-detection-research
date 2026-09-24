"""
Builds 2-class (Normal/Abnormal) YOLO datasets for the multi-source training
experiment: SIPaKMeD alone, APCData alone, and the two combined.

Why 2-class: SIPaKMeD's classes are cell morphology and APCData's are Bethesda
diagnostic categories. There is no defensible 1:1 mapping between them (see
config and the status report), so the only level at which the two can be trained
jointly is the Normal/Abnormal roll-up both taxonomies can reach.

Mappings come from config, not from this script:
  SIPaKMeD 5 -> ROLLUP_5_TO_3 -> ROLLUP_3_TO_2
  APCData  6 -> ROLLUP_APCDATA_6_TO_2

Label formats differ and are handled, same as in the evaluation scripts:
SIPaKMeD labels are segmentation polygons (class x1 y1 ... xn yn), APCData
labels are YOLO bboxes (class cx cy w h). Polygons are converted to their
axis-aligned bounding box.

Images are hard-linked where the filesystem allows it and copied otherwise, so
building all three variants does not cost three copies of the data.

Usage:
    python prepare_combined_2class.py
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import config
from _logging_setup import setup_logging

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
CLASSES_2 = config.CLASSES_2  # ["Normal", "Abnormal"]

SOURCES = {
    "sipakmed": {
        "root": config.DATA_DIR / "sipakmed" / "mirror_a",
        "names": ["Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal",
                  "Superficial-Intermediate"],
    },
    "apcdata": {
        "root": config.DATA_DIR / "apcdata" / "APCData_YOLO_prepared",
        "names": ["NILM", "ASCUS", "ASCH", "LSIL", "HSIL", "SCC"],
    },
}


def _norm(name):
    """APCData ships 'ASCUS'/'ASCH'; config uses the Bethesda 'ASC-US'/'ASC-H'."""
    return name.replace("-", "").upper()


def build_index_map(source):
    """Source class index -> 2-class index, via the config roll-ups."""
    apc = {_norm(k): v for k, v in config.ROLLUP_APCDATA_6_TO_2.items()}
    out = {}
    for i, name in enumerate(source["names"]):
        if name in config.ROLLUP_5_TO_3:
            two = config.ROLLUP_3_TO_2[config.ROLLUP_5_TO_3[name]]
        elif _norm(name) in apc:
            two = apc[_norm(name)]
        else:
            raise KeyError(f"No roll-up defined for class {name!r}")
        out[i] = CLASSES_2.index(two)
    return out


def convert_label(src_path, index_map):
    """Returns 2-class YOLO bbox lines. Polygons collapse to their bounding box."""
    lines = []
    for line in src_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        if cls not in index_map:
            continue
        vals = [float(v) for v in parts[1:]]
        if len(vals) == 4:
            cx, cy, w, h = vals
        else:
            xs, ys = vals[0::2], vals[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        lines.append(f"{index_map[cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def place(src, dst):
    """Hard-link if possible, copy otherwise. Avoids N copies of the image data."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def emit(source_keys, out_root, stats):
    for split in ("train", "valid"):
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "labels").mkdir(parents=True, exist_ok=True)

    for key in source_keys:
        src = SOURCES[key]
        index_map = build_index_map(src)
        for split in ("train", "valid"):
            img_dir = src["root"] / split / "images"
            lbl_dir = src["root"] / split / "labels"
            if not img_dir.exists():
                print(f"  [{key}/{split}] missing, skipped")
                continue
            n_img = n_box = 0
            per_class = {c: 0 for c in CLASSES_2}
            for ip in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT):
                lp = lbl_dir / (ip.stem + ".txt")
                if not lp.exists():
                    continue
                lines = convert_label(lp, index_map)
                stem = f"{key}__{ip.stem}"
                place(ip, out_root / split / "images" / f"{stem}{ip.suffix}")
                (out_root / split / "labels" / f"{stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                n_img += 1
                n_box += len(lines)
                for ln in lines:
                    per_class[CLASSES_2[int(ln.split()[0])]] += 1
            print(f"  [{key}/{split}] {n_img} images, {n_box} boxes, {per_class}")
            stats.setdefault(out_root.name, {}).setdefault(split, {})[key] = {
                "images": n_img, "boxes": n_box, "per_class": per_class}

    yaml = (f"names:\n" + "".join(f"- {c}\n" for c in CLASSES_2) +
            f"nc: {len(CLASSES_2)}\ntrain: train/images\nval: valid/images\n")
    (out_root / "data.yaml").write_text(yaml, encoding="utf-8")
    print(f"  wrote {out_root / 'data.yaml'}")


def main():
    setup_logging("prepare_combined_2class")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(config.DATA_DIR / "combined_2class"))
    args = ap.parse_args()

    base = Path(args.out_dir)
    variants = {
        "sipakmed_only": ["sipakmed"],
        "apcdata_only": ["apcdata"],
        "combined": ["sipakmed", "apcdata"],
    }

    stats = {}
    for name, keys in variants.items():
        print(f"\n=== {name} ({' + '.join(keys)}) ===")
        emit(keys, base / name, stats)

    report = config.RESULTS_DIR / "combined_2class_dataset_stats.json"
    report.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nStats written to {report}")


if __name__ == "__main__":
    main()
