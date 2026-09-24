"""
Model-assisted pre-annotation for the dense evaluation set.

Precision is unmeasurable on this project because SIPaKMeD labels only ~3-7%
of the cells visible in a field (see results/annotation_completeness.json).
Fixing that means densely annotating a held-out subset by hand. Drawing ~60
boxes per field from scratch is the expensive way; this script proposes the
boxes with a trained model so the human deletes and corrects instead.

What it does:
  1. Samples N fields from the SIPaKMeD validation split (seeded, reproducible).
  2. Runs a trained detector at a recall-favouring confidence.
  3. Reconciles the proposals against the sparse existing ground truth, so a
     real label is never replaced by a model guess (see reconcile()).
  4. Writes a correction-ready folder: images, YOLO .txt, Pascal VOC .xml,
     Label Studio import JSON, and preview renders that colour GT and
     proposals differently.
  5. Optionally uploads to a Roboflow project instead (--upload). Note the
     free Roboflow tier permits public projects only, which publishes the
     dense annotations as they are made; Label Studio keeps them local.

METHODOLOGICAL WARNING. The proposing model must not then be scored on the
corrected set without a caveat: cells the model missed are cells the annotator
never sees a box for, so uncorrected output inflates that model's recall.
Correcting a field means both fixing the drawn boxes AND sweeping the field for
cells nothing was proposed on. The preview renders exist for that sweep.

Usage:
    python annotation/pre_annotate.py --n 40
    python annotation/pre_annotate.py --n 40 --upload my-workspace/my-project
"""

import argparse
import json
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from _logging_setup import setup_logging

load_dotenv()

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# Recall-favouring default, deliberately below config.DEFAULT_INFERENCE_CONF
# (0.111). That threshold is an operating point for scoring, where a false box
# costs about as much as a missed one. Here it does not: a spurious box costs
# the annotator one keypress, a missed cell costs a box drawn from scratch --
# or, worse, silently stays unlabelled and poisons the precision measurement
# this whole exercise exists to obtain.
PREANNOT_CONF = 0.05


def load_names(split_dir):
    """Class names for a split, from the dataset's own data.yaml."""
    for parent in (split_dir.parent, split_dir):
        y = parent / "data.yaml"
        if y.exists():
            return yaml.safe_load(y.read_text(encoding="utf-8"))["names"]
    raise FileNotFoundError(f"no data.yaml near {split_dir}")


def gt_boxes(lbl_path, names, W, H):
    """Sparse GT -> [(class_name, x1, y1, x2, y2)] in pixels.

    Handles both label formats present in this project: SIPaKMeD ships
    segmentation polygons, APCData ships bboxes. Same convention as
    evaluation/firing_rate_matrix.py.
    """
    out = []
    if not lbl_path.exists():
        return out
    for line in lbl_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = names[int(parts[0])]
        v = [float(x) for x in parts[1:]]
        if len(v) == 4:
            cx, cy, w, h = v
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        else:
            xs, ys = v[0::2], v[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        out.append((cls, x1 * W, y1 * H, x2 * W, y2 * H))
    return out


def to_2class(name):
    """Native class name -> Normal/Abnormal, via the config roll-ups."""
    if name in config.CLASSES_2:
        return name
    return config.ROLLUP_3_TO_2[config.ROLLUP_5_TO_3[name]]


def iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float), np.asarray(b, float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ba = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ba[None, :] - inter + 1e-9)


def reconcile(gt, preds, iou_thr=0.4):
    """Merge sparse human GT with model proposals into one annotation set.

    Returns [(class_name, x1, y1, x2, y2, source)] with source in
    {"gt", "model"}.

    THE POLICY THIS ENCODES -- a judgement call, change it if you disagree.
    A human-drawn box always wins: it is kept verbatim, and any proposal
    overlapping it above iou_thr is dropped rather than merged. That keeps the
    ~3-7% of cells SIPaKMeD's own annotators labelled exactly as they were, so
    the dense set stays comparable to every result already in SESSION_LOG.md.
    The alternative -- letting a higher-confidence model box replace a GT box
    whose extent looks wrong -- gives tighter boxes but silently redefines the
    reference, and no earlier number would still apply to it.

    iou_thr is 0.4 rather than 0.5 because these are duplicate-suppression
    decisions, not matching decisions: a proposal at IoU 0.45 with a GT box is
    the same cell, and keeping both hands the annotator two boxes to reconcile
    on every one of the cells that was already correct.
    """
    keep = [(c, x1, y1, x2, y2, "gt") for c, x1, y1, x2, y2 in gt]
    if not preds:
        return keep
    if gt:
        m = iou_matrix([g[1:] for g in gt], [p[1:] for p in preds])
        dup = m.max(axis=0) >= iou_thr
    else:
        dup = np.zeros(len(preds), bool)
    keep += [(c, x1, y1, x2, y2, "model")
             for (c, x1, y1, x2, y2), d in zip(preds, dup) if not d]
    return keep


def write_yolo(path, boxes, names, W, H):
    lines = []
    for cls, x1, y1, x2, y2, _ in boxes:
        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        w, h = (x2 - x1) / W, (y2 - y1) / H
        lines.append(f"{names.index(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_voc(path, img_name, boxes, W, H):
    """Pascal VOC XML -- the upload format, because it carries class names
    inline and so needs no labelmap agreement with Roboflow."""
    ann = ET.Element("annotation")
    ET.SubElement(ann, "filename").text = img_name
    size = ET.SubElement(ann, "size")
    ET.SubElement(size, "width").text = str(W)
    ET.SubElement(size, "height").text = str(H)
    ET.SubElement(size, "depth").text = "3"
    for cls, x1, y1, x2, y2, _ in boxes:
        obj = ET.SubElement(ann, "object")
        ET.SubElement(obj, "name").text = cls
        ET.SubElement(obj, "difficult").text = "0"
        bb = ET.SubElement(obj, "bndbox")
        for tag, v in zip(("xmin", "ymin", "xmax", "ymax"), (x1, y1, x2, y2)):
            ET.SubElement(bb, tag).text = str(int(round(v)))
    ET.ElementTree(ann).write(path, encoding="utf-8", xml_declaration=True)


def write_preview(path, img_path, boxes):
    """GT green, model proposals amber. The annotator's sweep aid: any cell the
    eye finds outside a box is a cell the model missed."""
    im = Image.open(img_path).convert("RGB")
    d = ImageDraw.Draw(im)
    for cls, x1, y1, x2, y2, src in boxes:
        colour = (0, 220, 0) if src == "gt" else (255, 176, 0)
        d.rectangle([x1, y1, x2, y2], outline=colour, width=3)
        d.text((x1 + 4, max(0, y1 - 12)), cls[:4], fill=colour)
    im.save(path)


LS_CONFIG = """<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <RectangleLabels name="label" toName="image">
    <Label value="Normal" background="#00c800"/>
    <Label value="Abnormal" background="#ffb000"/>
  </RectangleLabels>
</View>
"""


def write_labelstudio(out_dir, records):
    """Label Studio import JSON, one task per field.

    Boxes go in `predictions` rather than `annotations`: Label Studio loads a
    prediction into the canvas as an editable region that the annotator adjusts
    and submits, which is the correction workflow. Put in `annotations` they
    would count as finished human work and the field would look already done.

    Geometry is percent-of-image, which is Label Studio's convention, not the
    normalized 0-1 of the YOLO files written alongside.
    """
    tasks = []
    for name, W, H, boxes in records:
        result = []
        for i, (cls, x1, y1, x2, y2, src) in enumerate(boxes):
            result.append({
                "id": f"b{i}",
                "from_name": "label", "to_name": "image",
                "type": "rectanglelabels",
                "original_width": W, "original_height": H, "image_rotation": 0,
                "value": {"x": 100 * x1 / W, "y": 100 * y1 / H,
                          "width": 100 * (x2 - x1) / W, "height": 100 * (y2 - y1) / H,
                          "rotation": 0, "rectanglelabels": [cls]},
                "meta": {"text": [src]},
            })
        # Path is relative to LOCAL_FILES_DOCUMENT_ROOT, which is the parent of
        # this set's folder (data/dense_eval), not the folder itself -- so a
        # pilot set and a later full set can live in one Label Studio instance.
        tasks.append({
            "data": {"image": f"/data/local-files/?d={out_dir.name}/images/{name}"},
            "predictions": [{"model_version": "pre_annotate", "result": result}],
        })
    (out_dir / "labelstudio_tasks.json").write_text(
        json.dumps(tasks, indent=1), encoding="utf-8")
    (out_dir / "labelstudio_config.xml").write_text(LS_CONFIG, encoding="utf-8")


def upload(out_dir, target, split):
    """Push the pre-annotated fields to a Roboflow project for correction."""
    from roboflow import Roboflow

    try:
        key = config.get_roboflow_key()
    except EnvironmentError as err:
        raise SystemExit(str(err))

    ws, _, proj = target.partition("/")
    if not proj:
        raise SystemExit("--upload wants workspace/project, e.g. my-ws/sipakmed-dense")
    project = Roboflow(api_key=key).workspace(ws).project(proj)

    ok = 0
    for img in sorted((out_dir / "images").iterdir()):
        xml = out_dir / "voc" / f"{img.stem}.xml"
        project.single_upload(image_path=str(img), annotation_path=str(xml),
                              split=split, is_prediction=False)
        ok += 1
        print(f"  uploaded {ok}: {img.name}")
    return ok


def main():
    setup_logging("pre_annotate")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model",
                    default=str(config.RESULTS_DIR / "runs/expC2b_combined_150/weights/best.pt"),
                    help="detector proposing the boxes (default: the replicated combined 2-class run)")
    ap.add_argument("--split", default=str(config.SIPAKMED_DIR / "mirror_a" / "valid"),
                    help="split directory containing images/ and labels/")
    ap.add_argument("--n", type=int, default=40, help="number of fields to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-side", type=int, default=0,
                    help="skip images whose shorter side is below this. The SIPaKMeD "
                         "valid split mixes 2048x1536 native cluster fields with 500px "
                         "and 1400px crops from other sources; --min-side 1536 selects "
                         "the native fields only, so cell density is comparable across "
                         "the dense set")
    ap.add_argument("--conf", type=float, default=PREANNOT_CONF)
    ap.add_argument("--imgsz", type=int, default=config.TRAINING_CONFIG["imgsz"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(config.DATA_DIR / "dense_eval" / "sipakmed_valid"))
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--upload", default=None, metavar="WORKSPACE/PROJECT",
                    help="Roboflow project to upload to; omit to only write files locally")
    ap.add_argument("--upload-split", default="valid")
    args = ap.parse_args()

    from ultralytics import YOLO

    split = Path(args.split)
    src_names = load_names(split)
    out_names = config.CLASSES_2
    out_dir = Path(args.out)
    for sub in ("images", "labels", "voc", "preview"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    imgs = sorted(p for p in (split / "images").iterdir() if p.suffix.lower() in IMG_EXT)
    if args.min_side:
        imgs = [p for p in imgs if min(Image.open(p).size) >= args.min_side]
    if not imgs:
        raise SystemExit(f"no images under {split / 'images'} (min_side={args.min_side})")
    # Shuffle once and slice, rather than random.sample(imgs, n). Sampling
    # returns unrelated sets for different n; slicing a fixed shuffle makes
    # them nested, so a --n 10 pilot is the first 10 fields of the later
    # --n 40 run and the annotation done on it is not thrown away.
    order = imgs[:]
    random.Random(args.seed).shuffle(order)
    sample = order[:min(args.n, len(order))]
    print(f"{len(imgs)} fields available, sampling {len(sample)} (seed {args.seed})")

    model = YOLO(args.model)
    model_names = model.names
    per_image, records, n_gt, n_model = [], [], 0, 0

    for img_path in sample:
        W, H = Image.open(img_path).size
        gt = [(to_2class(c), *b) for c, *b in
              gt_boxes(split / "labels" / f"{img_path.stem}.txt", src_names, W, H)]
        r = model.predict(source=str(img_path), conf=args.conf, imgsz=args.imgsz,
                          device=args.device, verbose=False)[0]
        preds = [(to_2class(model_names[int(c)]), *map(float, xyxy))
                 for c, xyxy in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist())]
        boxes = reconcile(gt, preds)

        shutil.copy2(img_path, out_dir / "images" / img_path.name)
        write_yolo(out_dir / "labels" / f"{img_path.stem}.txt", boxes, out_names, W, H)
        write_voc(out_dir / "voc" / f"{img_path.stem}.xml", img_path.name, boxes, W, H)
        if not args.no_preview:
            write_preview(out_dir / "preview" / f"{img_path.stem}.png", img_path, boxes)

        g = sum(1 for b in boxes if b[5] == "gt")
        m = len(boxes) - g
        n_gt, n_model = n_gt + g, n_model + m
        per_image.append({"image": img_path.name, "gt": g, "model": m, "total": len(boxes)})
        records.append((img_path.name, W, H, boxes))
        print(f"  {img_path.name[:44]:44s} gt={g:3d} proposed={m:3d}")

    write_labelstudio(out_dir, records)

    (out_dir / "data.yaml").write_text(
        yaml.safe_dump({"names": out_names, "nc": len(out_names),
                        "train": "images", "val": "images"}, sort_keys=False),
        encoding="utf-8")

    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "purpose": "dense-annotation seed set; boxes marked 'model' are UNVERIFIED",
        "model": args.model,
        "source_split": str(split),
        "n_fields": len(sample), "seed": args.seed, "min_side": args.min_side,
        "conf": args.conf, "imgsz": args.imgsz,
        "classes": out_names,
        "gt_boxes": n_gt, "model_boxes": n_model,
        "per_image": per_image,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{n_gt} GT boxes kept, {n_model} model boxes proposed "
          f"({(n_gt + n_model) / len(sample):.1f} per field, vs 4.5 in the sparse labels)")
    print(f"wrote {out_dir}")

    if args.upload:
        print(f"\nuploading to Roboflow project {args.upload} (split={args.upload_split})")
        print(f"done: {upload(out_dir, args.upload, args.upload_split)} images")


if __name__ == "__main__":
    main()
