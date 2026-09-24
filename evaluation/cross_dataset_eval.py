"""Localization-only comparison across native taxonomies; no classification benchmark deltas."""
import argparse
import json
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from evaluation.confusion_matrix import build_confusion_matrix, class_agnostic_detection_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--phase1-data', default=str(config.SIPAKMED_DATA_YAML))
    parser.add_argument('--phase2-data', default=str(config.APCDATA_DATA_YAML))
    parser.add_argument('--phase2-dataset-name', default='APCData')
    parser.add_argument('--conf', type=float, default=config.DEFAULT_INFERENCE_CONF)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    report = {
        'weights': str(Path(args.weights).resolve()),
        'weights_sha256': config.compute_file_sha256(args.weights),
        'conf_threshold': args.conf, 'iou_threshold': 0.5,
        'taxonomy': 'class-agnostic localization only',
        'limitation': 'Annotation completeness and box conventions differ by source. Target-source training exposure must be disclosed; no automatic zero-shot claim.',
    }
    for key, path in [('reference', args.phase1_data), ('target', args.phase2_data)]:
        if not Path(path).is_file():
            report[key] = {'status': 'UNAVAILABLE', 'data': path}
            continue
        matrix, _ = build_confusion_matrix(args.weights, path, args.conf, taxonomy='agnostic')
        report[key] = {'data': path, **class_agnostic_detection_metrics(matrix)}
    slug = re.sub(r'[^a-z0-9]+', '_', args.phase2_dataset_name.lower()).strip('_')
    out = args.out or config.RESULTS_DIR / f'cross_dataset_eval_{slug}.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
