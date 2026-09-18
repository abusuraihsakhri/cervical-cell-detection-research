"""
Stain Normalization Benchmark (Reinhard Method).

Evaluates whether digital stain normalization (Reinhard color transfer in CIELAB space)
improves cross-dataset detection accuracy beyond the empirical run-to-run noise floor (+/-0.03).

Tested on:
  1. Exp C1 (single-source SIPaKMeD baseline)
  2. Exp C2b (multi-source combined baseline)
Across:
  - APCData validation split (cross-domain)
  - SIPaKMeD dense verified split

Usage:
    python evaluation/stain_normalization_benchmark.py
"""

import argparse
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from _logging_setup import setup_logging


class ReinhardNormalizer:
    """Reinhard color transfer in CIELAB color space."""

    def __init__(self):
        self.target_means = None
        self.target_stds = None

    def fit(self, reference_images: list):
        """Compute mean and std in Lab space across reference images."""
        l_vals, a_vals, b_vals = [], [], []
        for p in reference_images:
            img = cv2.imread(str(p))
            if img is None:
                continue
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
            # Mask out background (near-white / empty slide glass)
            mask = (lab[:, :, 0] < 240) & (lab[:, :, 0] > 10)
            if np.sum(mask) < 100:
                continue
            l_vals.append(lab[:, :, 0][mask])
            a_vals.append(lab[:, :, 1][mask])
            b_vals.append(lab[:, :, 2][mask])

        all_l = np.concatenate(l_vals) if l_vals else np.array([128.0])
        all_a = np.concatenate(a_vals) if a_vals else np.array([128.0])
        all_b = np.concatenate(b_vals) if b_vals else np.array([128.0])

        self.target_means = np.array([np.mean(all_l), np.mean(all_a), np.mean(all_b)])
        self.target_stds = np.array([np.std(all_l), np.std(all_a), np.std(all_b)]) + 1e-6
        print(f"Fitted reference stain stats -> Means: {self.target_means.round(2)}, Stds: {self.target_stds.round(2)}")

    def transform(self, img_bgr: np.ndarray) -> np.ndarray:
        """Transfer reference color distribution to input image."""
        if self.target_means is None:
            raise ValueError("Normalizer not fitted yet.")

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        mask = (lab[:, :, 0] < 240) & (lab[:, :, 0] > 10)

        for i in range(3):
            ch = lab[:, :, i]
            mean = np.mean(ch[mask]) if np.sum(mask) > 100 else np.mean(ch)
            std = np.std(ch[mask]) + 1e-6 if np.sum(mask) > 100 else np.std(ch) + 1e-6
            ch_norm = (ch - mean) * (self.target_stds[i] / std) + self.target_means[i]
            lab[:, :, i] = np.clip(ch_norm, 0, 255)

        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    return interArea / unionArea if unionArea > 0 else 0.0


def evaluate_folder(model, img_dir: Path, lbl_dir: Path, normalizer=None, conf=0.111, iou_thresh=0.5):
    """Evaluate detection metrics over an image folder with optional on-the-fly normalization."""
    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    total_gt = 0
    total_pred = 0
    tp = 0

    for img_path in img_files:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        gt_boxes = []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    # Handle both bbox (4-values) and polygon (>4-values)
                    coords = [float(x) for x in parts[1:]]
                    if len(coords) == 4:
                        cx, cy, w, h = coords
                        gt_boxes.append((cx - w/2, cy - h/2, cx + w/2, cy + h/2))
                    else:
                        xs, ys = coords[0::2], coords[1::2]
                        gt_boxes.append((min(xs), min(ys), max(xs), max(ys)))
        total_gt += len(gt_boxes)

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        if normalizer is not None:
            img_to_feed = normalizer.transform(img_bgr)
            # YOLO accepts numpy BGR image directly
            res = model.predict(source=img_to_feed, conf=conf, imgsz=640, verbose=False, device=0)[0]
        else:
            res = model.predict(source=str(img_path), conf=conf, imgsz=640, verbose=False, device=0)[0]

        preds = []
        if len(res.boxes) > 0:
            preds = [tuple(b) for b in res.boxes.xyxyn.cpu().numpy()]
        total_pred += len(preds)

        matched = set()
        for p in preds:
            best_iou, best_gi = 0.0, -1
            for gi, g in enumerate(gt_boxes):
                if gi in matched:
                    continue
                iou = calculate_iou(p, g)
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= iou_thresh and best_gi >= 0:
                matched.add(best_gi)
                tp += 1

    prec = tp / total_pred if total_pred > 0 else 1.0
    rec = tp / total_gt if total_gt > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"total_gt": total_gt, "total_pred": total_pred, "tp": tp, "precision": prec, "recall": rec, "f1": f1}


def generate_comparison_figure(normalizer, src_sample: Path, ref_sample: Path, out_fig: Path):
    """Plot visual comparison of raw vs stain normalized images."""
    src_bgr = cv2.imread(str(src_sample))
    ref_bgr = cv2.imread(str(ref_sample))
    norm_bgr = normalizer.transform(src_bgr)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

    axes[0].imshow(cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Source Image (Unnormalized)\n{src_sample.name[:25]}", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Reference Stain Target\n{ref_sample.name[:25]}", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(norm_bgr, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Result: Reinhard Normalized\n(Transferred Color Profile)", fontsize=11, fontweight="bold", color="#1b365d")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_fig, bbox_inches="tight")
    plt.close()
    print(f"Saved visual stain transfer comparison to: {out_fig}")


def main():
    setup_logging("stain_normalization_benchmark")
    from ultralytics import YOLO

    # 1. Fit normalizer on SIPaKMeD reference images
    sipakmed_train_imgs = list((config.DATA_DIR / "sipakmed" / "mirror_a" / "train" / "images").glob("*.jpg"))[:50]
    normalizer = ReinhardNormalizer()
    normalizer.fit(sipakmed_train_imgs)

    # 2. Datasets to benchmark
    apcdata_val_img = config.DATA_DIR / "apcdata" / "APCData_YOLO_prepared" / "valid" / "images"
    apcdata_val_lbl = config.DATA_DIR / "apcdata" / "APCData_YOLO_prepared" / "valid" / "labels"

    dense_val_img = config.DATA_DIR / "dense_eval" / "sipakmed_dense40_verified" / "images"
    dense_val_lbl = config.DATA_DIR / "dense_eval" / "sipakmed_dense40_verified" / "labels"

    models_to_test = [
        ("Exp C1 (SIPaKMeD 2-class)", config.RESULTS_DIR / "runs" / "expC1_sipakmed_2class" / "weights" / "best.pt"),
        ("Exp C2b (Combined 2-class)", config.RESULTS_DIR / "runs" / "expC2b_combined_150" / "weights" / "best.pt"),
    ]

    report = {
        "method": "Reinhard CIELAB Color Transfer",
        "noise_floor_reference": 0.03,
        "results": {},
    }

    print("\n" + "=" * 105)
    print(f"{'Model':<26} | {'Target Dataset':<18} | {'Raw Rec':<8} | {'Norm Rec':<9} | {'Delta Rec':<10} | {'Raw F1':<8} | {'Norm F1':<8}")
    print("=" * 105)

    for m_name, w_path in models_to_test:
        if not w_path.exists():
            continue
        model = YOLO(str(w_path))
        report["results"][m_name] = {}

        # Benchmark on APCData (cross-domain transfer target)
        if apcdata_val_img.exists():
            raw_res = evaluate_folder(model, apcdata_val_img, apcdata_val_lbl, normalizer=None)
            norm_res = evaluate_folder(model, apcdata_val_img, apcdata_val_lbl, normalizer=normalizer)
            d_rec = norm_res["recall"] - raw_res["recall"]
            d_f1 = norm_res["f1"] - raw_res["f1"]
            report["results"][m_name]["APCData"] = {
                "raw": raw_res,
                "normalized": norm_res,
                "delta_recall": d_rec,
                "delta_f1": d_f1,
                "significant_improvement": bool(d_rec > 0.03),
            }
            print(f"{m_name:<26} | {'APCData (Cross)':<18} | {raw_res['recall']:<8.4f} | {norm_res['recall']:<9.4f} | {d_rec:<+10.4f} | {raw_res['f1']:<8.4f} | {norm_res['f1']:<8.4f}")

        # Benchmark on SIPaKMeD Dense Verified
        if dense_val_img.exists():
            raw_res_d = evaluate_folder(model, dense_val_img, dense_val_lbl, normalizer=None)
            norm_res_d = evaluate_folder(model, dense_val_img, dense_val_lbl, normalizer=normalizer)
            d_rec_d = norm_res_d["recall"] - raw_res_d["recall"]
            d_f1_d = norm_res_d["f1"] - raw_res_d["f1"]
            report["results"][m_name]["SIPaKMeD_Dense"] = {
                "raw": raw_res_d,
                "normalized": norm_res_d,
                "delta_recall": d_rec_d,
                "delta_f1": d_f1_d,
                "significant_improvement": bool(d_rec_d > 0.03),
            }
            print(f"{m_name:<26} | {'SIPaKMeD (Dense)':<18} | {raw_res_d['recall']:<8.4f} | {norm_res_d['recall']:<9.4f} | {d_rec_d:<+10.4f} | {raw_res_d['f1']:<8.4f} | {norm_res_d['f1']:<8.4f}")

    print("=" * 105)

    # Save JSON report
    out_json = config.RESULTS_DIR / "stain_normalization_benchmark.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved stain normalization benchmark to: {out_json}")

    # Generate sample figure
    apc_samples = list(apcdata_val_img.glob("*.jpg"))
    if apc_samples and sipakmed_train_imgs:
        out_fig = config.RESULTS_DIR / "stain_normalization_samples.png"
        generate_comparison_figure(normalizer, apc_samples[0], sipakmed_train_imgs[0], out_fig)


if __name__ == "__main__":
    main()
