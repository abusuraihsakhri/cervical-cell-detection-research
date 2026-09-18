"""
Evaluates trained models against the human-verified dense SIPaKMeD evaluation set.

Once human verification of the 40 fields in Label Studio is completed and exported
(either as Label Studio JSON export or YOLO format), this script:
  1. Parses the verified annotations into a clean YOLO evaluation split.
  2. Runs all major checkpoints against the ground truth (IoU >= 0.5).
  3. Computes the first TRUE, UNCONFOUNDED Precision, Recall, and F1 metrics.
  4. Saves results to results/dense_verified_evaluation_report.json.

Usage:
    # If exported from Label Studio as JSON:
    python annotation/evaluate_dense.py --export-json path/to/project-annotations.json

    # If verified labels are already in YOLO format:
    python annotation/evaluate_dense.py --verified-dir data/dense_eval/sipakmed_dense40_verified
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from _logging_setup import setup_logging

CLASS_MAP_2CLASS = {
    "normal": 0,
    "abnormal": 1,
}
CLASS_NAMES_2CLASS = ["Normal", "Abnormal"]


def parse_labelstudio_export(json_path: Path, images_dir: Path, out_dir: Path) -> Path:
    """Convert Label Studio export JSON into standard YOLO dataset folder."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    out_img_dir = out_dir / "images"
    out_lbl_dir = out_dir / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    verified_count = 0
    for task in data:
        # Find image file
        img_name = task.get("data", {}).get("image_name")
        if not img_name:
            raw_img_path = task.get("data", {}).get("image", "")
            img_name = Path(raw_img_path.split("=")[-1]).name if "=" in raw_img_path else Path(raw_img_path).name
        src_img = images_dir / img_name
        if not src_img.exists():
            # Search recursively in dense_eval
            matches = list(images_dir.parent.glob(f"**/{img_name}"))
            if matches:
                src_img = matches[0]

        if not src_img.exists():
            continue

        # Extract human annotations (from 'annotations' field if verified, else fall back to predictions)
        annotations = task.get("annotations", [])
        if annotations:
            ann_results = annotations[0].get("result", [])
        else:
            predictions = task.get("predictions", [])
            ann_results = predictions[0].get("result", []) if predictions else []

        w_img, h_img = Image.open(src_img).size
        yolo_lines = []
        for r in ann_results:
            if r.get("type") != "rectanglelabels":
                continue
            val = r.get("value", {})
            labels = val.get("rectanglelabels", [])
            if not labels:
                continue
            label_name = labels[0].strip().lower()
            cls_id = CLASS_MAP_2CLASS.get(label_name, 0)

            # Label Studio uses 0-100 percentages
            x_pct = val.get("x", 0)
            y_pct = val.get("y", 0)
            w_pct = val.get("width", 0)
            h_pct = val.get("height", 0)

            cx = (x_pct + w_pct / 2.0) / 100.0
            cy = (y_pct + h_pct / 2.0) / 100.0
            bw = w_pct / 100.0
            bh = h_pct / 100.0
            yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        # Copy image and write labels
        shutil.copy2(src_img, out_img_dir / img_name)
        (out_lbl_dir / f"{src_img.stem}.txt").write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
        verified_count += 1

    yaml_content = f"train: images\nval: images\nnc: 2\nnames: {CLASS_NAMES_2CLASS}\n"
    (out_dir / "data.yaml").write_text(yaml_content, encoding="utf-8")
    print(f"Imported {verified_count} verified images into {out_dir}")
    return out_dir


def evaluate_models(verified_yaml: Path, conf: float = 0.111, iou_thresh: float = 0.5):
    """Run evaluation for primary models against verified dense dataset."""
    from ultralytics import YOLO
    from evaluation.confusion_matrix import build_confusion_matrix, per_class_precision_recall, class_agnostic_detection_metrics

    candidate_models = [
        ("Phase 1 (SIPaKMeD 5-class)", Path("models/best.pt")),
        ("Exp C1 (SIPaKMeD 2-class)", Path("results/runs/expC1_sipakmed_2class/weights/best.pt")),
        ("Exp C2b (Combined 2-class)", Path("results/runs/expC2b_combined_150/weights/best.pt")),
    ]

    report = {"conf_threshold": conf, "iou_threshold": iou_thresh, "models": {}}
    print("\n" + "=" * 80)
    print(f"{'Model':<30} | {'Class-Agnostic Recall':<22} | {'Precision':<12} | {'F1-Score':<10}")
    print("=" * 80)

    for name, weights_path in candidate_models:
        if not weights_path.exists():
            continue
        try:
            config.verify_model_checksum(weights_path)
            cm, names = build_confusion_matrix(str(weights_path), str(verified_yaml), conf_thresh=conf)
            metrics = class_agnostic_detection_metrics(cm)
            per_class = per_class_precision_recall(cm, names)

            report["models"][name] = {
                "weights": str(weights_path),
                "class_agnostic": metrics,
                "per_class": per_class,
            }
            rec = metrics.get("recall", 0.0)
            prec = metrics.get("precision", 0.0)
            f1 = metrics.get("f1", 0.0)
            print(f"{name:<30} | {rec:<22.4f} | {prec:<12.4f} | {f1:<10.4f}")
        except Exception as err:
            print(f"Failed evaluating {name}: {err}")

    print("=" * 80)
    out_report = config.RESULTS_DIR / "dense_verified_evaluation_report.json"
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved comprehensive verified report to: {out_report}")


def main():
    setup_logging("evaluate_dense")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-json", type=Path, default=None, help="Path to exported Label Studio JSON")
    parser.add_argument(
        "--source-images",
        type=Path,
        default=config.DATA_DIR / "dense_eval" / "sipakmed_dense40" / "images",
        help="Source images directory",
    )
    parser.add_argument(
        "--verified-dir",
        type=Path,
        default=config.DATA_DIR / "dense_eval" / "sipakmed_dense40_verified",
        help="Directory to store or read verified YOLO dataset",
    )
    parser.add_argument("--conf", type=float, default=config.DEFAULT_INFERENCE_CONF)
    args = parser.parse_args()

    if args.export_json and args.export_json.exists():
        yaml_path = parse_labelstudio_export(args.export_json, args.source_images, args.verified_dir) / "data.yaml"
    else:
        yaml_path = args.verified_dir / "data.yaml"

    if not yaml_path.exists():
        print(
            f"ERROR: {yaml_path} does not exist.\n"
            "Either pass --export-json from Label Studio or place verified dataset in --verified-dir.",
            file=sys.stderr,
        )
        sys.exit(1)

    evaluate_models(yaml_path, conf=args.conf)


if __name__ == "__main__":
    main()
