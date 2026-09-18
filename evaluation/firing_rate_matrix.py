"""
Class-agnostic detection comparison across models and datasets.

Answers one question the class-level metrics cannot: given a field of cells,
how many does each model actually fire on, and how many real cells does it
find? Everything here is class-agnostic (IoU>=0.5, any predicted class counts),
because the models being compared have different class counts (SIPaKMeD 5,
APCData 6, roll-up 2) and no defensible mapping between their taxonomies.

Precision is reported but must be read with the annotation-completeness finding
in mind: both datasets label only a fraction of the cells present, so precision
here is a lower bound, not a measurement. Recall and the predicted-to-GT firing
ratio are the trustworthy columns.

Usage:
    python evaluation/firing_rate_matrix.py \
        --model sipakmed=models/best.pt \
        --model apcdata=results/runs/expA_apcdata/weights/best.pt \
        --out results/firing_rate_matrix.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from _logging_setup import setup_logging

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

DATASETS = {
    "SIPaKMeD": config.DATA_DIR / "sipakmed/mirror_a/valid",
    "APCData": config.DATA_DIR / "apcdata/APCData_YOLO_prepared/valid",
}


def gt_boxes(lbl_path, W, H):
    """Normalized labels -> pixel xyxy. Handles bbox (4 values) and polygon (>4)."""
    out = []
    if not lbl_path.exists():
        return out
    for line in lbl_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        v = [float(x) for x in parts[1:]]
        if len(v) == 4:
            cx, cy, w, h = v
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        else:
            xs, ys = v[0::2], v[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        out.append([x1 * W, y1 * H, x2 * W, y2 * H])
    return out


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


def greedy_match(gt, pred, thr=0.5):
    """Greedy one-to-one IoU matching. Returns match count."""
    if not len(gt) or not len(pred):
        return 0
    m = iou_matrix(gt, pred)
    matched = 0
    used_g, used_p = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(-m, axis=None), m.shape))[0]
    for gi, pi in order:
        if m[gi, pi] < thr:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(int(gi)); used_p.add(int(pi)); matched += 1
    return matched


def evaluate(model, split, conf, device=None):
    img_dir, lbl_dir = split / "images", split / "labels"
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    tot_gt = tot_pred = tot_match = 0
    zero_pred = 0
    for ip in images:
        with Image.open(ip) as im:
            W, H = im.size
        r = model.predict(str(ip), conf=conf, verbose=False, device=device)[0]
        pred = [] if r.boxes is None else r.boxes.xyxy.cpu().numpy().tolist()
        g = gt_boxes(lbl_dir / (ip.stem + ".txt"), W, H)
        tot_gt += len(g); tot_pred += len(pred)
        tot_match += greedy_match(g, pred)
        if not pred:
            zero_pred += 1
    return {
        "images": len(images),
        "gt_boxes": tot_gt,
        "pred_boxes": tot_pred,
        "matched": tot_match,
        "firing_ratio_pred_per_gt": round(tot_pred / max(tot_gt, 1), 4),
        "recall": round(tot_match / max(tot_gt, 1), 4),
        "precision_lower_bound": round(tot_match / max(tot_pred, 1), 4),
        "mean_pred_per_image": round(tot_pred / max(len(images), 1), 3),
        "images_with_zero_predictions": zero_pred,
    }


def main():
    setup_logging("firing_rate_matrix")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", required=True,
                    metavar="NAME=PATH", help="repeatable, e.g. sipakmed=models/best.pt")
    ap.add_argument("--conf", type=float, default=config.DEFAULT_INFERENCE_CONF)
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "firing_rate_matrix.json"))
    ap.add_argument("--device", default=None,
                    help="Pass 'cpu' to keep this off the GPU while a training run "
                         "holds VRAM. Default lets ultralytics choose.")
    args = ap.parse_args()

    from ultralytics import YOLO

    report = {"conf": args.conf, "models": {}, "_caveat": (
        "Class-agnostic (IoU>=0.5, class identity ignored) because the models compared "
        "have different class counts and taxonomies. precision_lower_bound is NOT a "
        "precision measurement: both datasets annotate only a fraction of the cells "
        "present (see results/annotation_completeness.json), so unlabelled-but-correct "
        "detections are counted against it. Read recall and firing_ratio_pred_per_gt."
    )}

    for spec in args.model:
        name, path = spec.split("=", 1)
        print(f"\n=== model {name} ({path}) ===")
        model = YOLO(path)
        report["models"][name] = {"weights": path, "classes": model.names, "on": {}}
        for ds, split in DATASETS.items():
            if not (split / "images").exists():
                continue
            res = evaluate(model, split, args.conf, args.device)
            report["models"][name]["on"][ds] = res
            print(f"  {ds:10s} gt={res['gt_boxes']:5d} pred={res['pred_boxes']:6d} "
                  f"fire={res['firing_ratio_pred_per_gt']:6.3f} "
                  f"recall={res['recall']:.3f} prec_lb={res['precision_lower_bound']:.3f}")

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
