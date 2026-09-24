"""Secondary, post hoc descriptive analysis: the same predictions scored against
sparse (original public) and dense (user-reviewed) references on the frozen v3
dense evaluation fields. Not prespecified in evaluation_protocol.json.

Reuses cached dense predictions and each model's development-selected threshold.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.research_v2 import collect, field_counts, summarize, write_json
from evaluation.research_v3 import OUT, CACHE, load_dense
from evaluation.taxonomy import binary_index_map

SPARSE_LABELS = ROOT / 'data/sipakmed/mirror_a/valid/labels'
SPARSE_NAMES = ['Dyskeratotic', 'Koilocytotic', 'Metaplastic', 'Parabasal', 'Superficial-Intermediate']


def sparse_boxes(stem):
    """Original 5-class polygon or box labels, mapped to binary by class name."""
    mapping = binary_index_map(SPARSE_NAMES)
    boxes = []
    for line in (SPARSE_LABELS / f'{stem}.txt').read_text().splitlines():
        values = line.split()
        if not values:
            continue
        cls, coords = int(values[0]), [float(v) for v in values[1:]]
        if len(coords) == 4:
            cx, cy, w, h = coords
            bbox = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
        else:
            xs, ys = coords[0::2], coords[1::2]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
        boxes.append({'cls': mapping[cls], 'bbox': bbox})
    return boxes


def main():
    evaluation = json.loads((OUT / 'evaluation_summary.json').read_text())
    protocol = evaluation['protocol']
    dense = load_dense(protocol)
    test = {n: dense[n] for n in protocol['dense_evaluation']}
    sparse = {n: {**v, 'boxes': sparse_boxes(Path(n).stem)} for n, v in test.items()}
    report = {
        'status': 'Secondary post hoc descriptive analysis; not prespecified. Same predictions and thresholds; only the reference labels differ.',
        'reference_cells': {
            'sparse': sum(len(v['boxes']) for v in sparse.values()),
            'dense': sum(len(v['boxes']) for v in test.values()),
            'sparse_abnormal': sum(b['cls'] == 1 for v in sparse.values() for b in v['boxes']),
            'dense_abnormal': sum(b['cls'] == 1 for v in test.values() for b in v['boxes']),
        },
        'models': {},
    }
    for name, result in evaluation['models'].items():
        weights = OUT / 'runs' / name / 'weights/best.pt'
        predictions = collect(weights, dense, protocol['inference'], name + '_dense', cache_dir=CACHE)
        threshold = result['threshold_from_development']
        report['models'][name] = {
            'threshold': threshold,
            'sparse_reference': summarize(field_counts(sparse, predictions, threshold)),
            'dense_reference': summarize(field_counts(test, predictions, threshold)),
        }
    write_json(ROOT / 'publication/results/annotation_completeness_secondary.json', report)
    print(json.dumps(report['reference_cells']))


if __name__ == '__main__':
    main()
