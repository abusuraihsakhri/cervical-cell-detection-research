"""Secondary, post hoc descriptive analysis: effect of the earlier HMCHH label
conversion. The same cached predictions and thresholds are scored against the
earlier reference (all rectangles labelled abnormal, polygons dropped) and the
corrected reference (abnormal cells only, polygons included).
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.research_v2 import load_folder, field_counts, summarize, write_json
from evaluation.research_v3 import OUT, CACHE, HMCHH

EARLIER = Path('D:/pap_model/HMCHH_YOLO_prepared/valid')


def cached_predictions(name):
    files = sorted(CACHE.glob(f'{name}_hmchh_*.json'))
    if len(files) != 1:
        raise ValueError(f'Expected one cached HMCHH prediction file for {name}, found {len(files)}')
    return json.loads(files[0].read_text())['predictions']


def main():
    models = json.loads((OUT / 'evaluation_summary.json').read_text())['models']
    corrected = load_folder(HMCHH, abnormal_only=True)
    earlier = load_folder(EARLIER, abnormal_only=True)
    if set(corrected) != set(earlier):
        raise ValueError('References cover different fields')
    report = {'status': 'Secondary post hoc descriptive analysis; same predictions and thresholds, only the reference differs.',
              'reference_cells': {'earlier': sum(len(v['boxes']) for v in earlier.values()),
                                  'corrected': sum(len(v['boxes']) for v in corrected.values())},
              'models': {}}
    for name, result in models.items():
        predictions = cached_predictions(name)
        threshold = result['threshold_from_development']
        report['models'][name] = {'threshold': threshold,
                                  'earlier_reference': summarize(field_counts(earlier, predictions, threshold)),
                                  'corrected_reference': summarize(field_counts(corrected, predictions, threshold))}
    write_json(ROOT / 'publication/results/hmchh_conversion_effect_secondary.json', report)
    print(json.dumps(report['reference_cells']))
    for name, r in report['models'].items():
        e, c = r['earlier_reference']['metrics'], r['corrected_reference']['metrics']
        print(f"{name}: recall {e['abnormal_recall']:.3f} -> {c['abnormal_recall']:.3f}; "
              f"precision {e['abnormal_precision']:.3f} -> {c['abnormal_precision']:.3f}")


if __name__ == '__main__':
    main()
