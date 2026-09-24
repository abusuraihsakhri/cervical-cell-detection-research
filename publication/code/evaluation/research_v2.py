"""Frozen retrospective evaluation with development-only thresholds and held-out calibration."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.taxonomy import binary_index_map
from evaluation.sweep_operating_thresholds import evaluate_at_threshold, calculate_iou
from evaluation.calibration import fit_temperature_scale, apply_temperature, expected_calibration_error
from prepare_research_v2 import group_id

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/research_v2'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def freeze():
    path = OUT / 'evaluation_protocol.json'
    if path.exists():
        raise FileExistsError('Evaluation protocol already frozen')
    images = sorted((ROOT / 'data/dense_eval/sipakmed_dense40_verified/images').glob('*.jpg'))
    canonical = [p.name for p in images if group_id('sipakmed', p.name)]
    other = [p.name for p in images if not group_id('sipakmed', p.name)]
    random.Random(20260920).shuffle(canonical)
    dev = other + canonical[:20-len(other)]
    test = canonical[20-len(other):]
    assert len(dev) == len(test) == 20 and not set(dev) & set(test)
    protocol = {
        'status': 'Retrospective research. Historical images previously examined; no prospective confirmation claim.',
        'split_seed': 20260920,
        'dense_development': dev, 'dense_evaluation': test,
        'dense_split_rule': 'All unresolved-source fields confined to development; seeded canonical fields fill development to 20, leaving 20 canonical fields for evaluation. Deliberately different source compositions.',
        'threshold_selection': 'Maximize abnormal F1 on dense development only; ties favor lower threshold. Thresholds 0.01..0.65 in 0.01 steps.',
        'calibration': 'Binary NLL temperature fit on development predictions; evaluate on separate dense evaluation predictions at fixed floor 0.01. Conditional on detected candidates, not missed-cell calibration.',
        'inference': {'imgsz': 640, 'conf': 0.01, 'iou': 0.7, 'max_det': 300},
        'matching': 'Confidence-ordered greedy IoU >= 0.5; localization class-agnostic; abnormal metric matched separately to abnormal references.',
        'bootstrap': {'iterations': 1000, 'seed': 20260921, 'unit': 'source-field groups; HMCHH patient-prefix proxies'},
        'dense_image_hashes': {p.name: digest(p) for p in images},
        'dense_label_hashes': {p.stem: digest(p.parent.parent/'labels'/(p.stem+'.txt')) for p in images},
        'training_manifest_sha256': digest(ROOT/'data/research_v2/manifest.json'),
        'annotation_attestation_sha256': digest(ROOT/'results/annotation_attestation_20260919.json'),
    }
    write_json(path, protocol)
    print('Frozen 20 development / 20 evaluation fields, without reading model outcomes.')


def load_folder(root, *, abnormal_only=False):
    out = {}
    for p in sorted((root/'images').iterdir()):
        if p.suffix.lower() not in {'.jpg', '.png', '.jpeg'}:
            continue
        label = root/'labels'/(p.stem+'.txt')
        if not label.is_file():
            raise ValueError(f'Missing label {label}')
        boxes = []
        for line in label.read_text().splitlines():
            if not line.strip():
                continue
            cls, cx, cy, w, h = map(float, line.split())
            if not (cls in (0, 1) and w > 0 and h > 0):
                raise ValueError(f'Invalid binary box in {label}')
            boxes.append({'cls': 1 if abnormal_only else int(cls), 'bbox': [cx-w/2, cy-h/2, cx+w/2, cy+h/2]})
        if abnormal_only:
            group = p.stem.split('_')[0]
        else:
            bare = p.name.split('__')[-1]
            group = group_id('apcdata' if p.name.startswith('apcdata__') else 'sipakmed', bare) or 'unresolved_dense_source'
        out[p.name] = {'path': p, 'boxes': boxes, 'group': group, 'image_sha256': digest(p), 'label_sha256': digest(label)}
    if not out:
        raise ValueError(f'Empty evaluation folder {root}')
    return out


def collect(weights, data, settings, tag, normalizer=None, cache_dir=None):
    from ultralytics import YOLO
    import importlib.metadata
    identity = {'weights': digest(weights), 'settings': settings,
                'versions': {name: importlib.metadata.version(name) for name in ("torch", "ultralytics", "numpy")},
                'data': {n: [v['image_sha256'], v['label_sha256']] for n,v in data.items()},
                'transform': None if normalizer is None else [normalizer.target_means.tolist(), normalizer.target_stds.tolist()]}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = (cache_dir or OUT/'prediction_cache')/f'{tag}_{key}.json'
    if cache.exists():
        return json.loads(cache.read_text())['predictions']
    model = YOLO(str(weights))
    mapping = binary_index_map(model.names)
    predictions = {}
    for n, info in data.items():
        source = str(info['path'])
        if normalizer is not None:
            import cv2
            source = normalizer.transform(cv2.imread(source))
        result = model.predict(source, verbose=False, **settings)[0]
        predictions[n] = [{'bbox': [float(x) for x in box], 'conf': float(conf), 'cls': mapping[int(cls)]}
                          for box, conf, cls in zip(result.boxes.xyxyn.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy())]
        predictions[n].sort(key=lambda p: -p['conf'])
    write_json(cache, {'identity': identity, 'predictions': predictions})
    return predictions


def field_counts(data, predictions, threshold):
    out = {}
    for name, info in data.items():
        metrics = evaluate_at_threshold({name: info}, {name: predictions[name]}, threshold)
        out[name] = {'group': info['group'], 'counts': [metrics['tp'], metrics['total_pred'], metrics['total_gt'], metrics['abnormal_tp'], metrics['abnormal_total_pred'], metrics['abnormal_total_gt'], 1]}
    return out


def ratios(counts):
    tp, pred, gt, atp, apred, agt, fields = np.asarray(counts, dtype=float)
    return {'localization_precision': tp/pred if pred else 0., 'localization_recall': tp/gt if gt else 0.,
            'localization_f1': 2*tp/(pred+gt) if pred+gt else 0.,
            'abnormal_precision': atp/apred if apred else 0., 'abnormal_recall': atp/agt if agt else 0.,
            'abnormal_f1': 2*atp/(apred+agt) if apred+agt else 0.,
            'abnormal_false_detections_per_field': (apred-atp)/fields}


def summarize(counts, iterations=1000):
    groups = defaultdict(lambda: np.zeros(7))
    for value in counts.values():
        groups[value['group']] += value['counts']
    values = np.array(list(groups.values()))
    total = values.sum(axis=0)
    rng = np.random.default_rng(20260921)
    samples = [ratios(values[rng.integers(0, len(values), len(values))].sum(axis=0)) for _ in range(iterations)]
    return {'metrics': ratios(total), 'counts': dict(zip(['localization_tp', 'predictions', 'reference_cells', 'abnormal_tp', 'abnormal_predictions', 'abnormal_reference_cells', 'fields'], total.astype(int).tolist())),
            'groups': len(groups), 'conditional_cluster_bootstrap_95ci': {k: np.quantile([s[k] for s in samples], [.025,.975]).tolist() for k in samples[0]}}


def correctness(data, predictions):
    records = []
    for name, info in data.items():
        used = set()
        for p in predictions[name]:
            candidates = [(calculate_iou(p['bbox'], g['bbox']), i) for i,g in enumerate(info['boxes']) if i not in used and p['cls'] == g['cls']]
            overlap, index = max(candidates, default=(0., -1))
            correct = overlap >= .5
            if correct:
                used.add(index)
            records.append((p['conf'], correct))
    return records


def calibration_metrics(records):
    probs = np.clip([r[0] for r in records], 1e-8, 1-1e-8)
    y = np.array([r[1] for r in records], dtype=float)
    ece, bins = expected_calibration_error(records)
    return {'ece': ece, 'brier': float(np.mean((probs-y)**2)), 'nll': float(-np.mean(y*np.log(probs)+(1-y)*np.log(1-probs))), 'bins': bins, 'n_predictions': len(records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze()
        return
    protocol = json.loads((OUT/'evaluation_protocol.json').read_text())
    if digest(ROOT/'data/research_v2/manifest.json') != protocol['training_manifest_sha256']:
        raise ValueError('Frozen training manifest changed')
    if digest(ROOT/'results/annotation_attestation_20260919.json') != protocol['annotation_attestation_sha256']:
        raise ValueError('Frozen annotation attestation changed')
    dense = load_folder(ROOT/'data/dense_eval/sipakmed_dense40_verified')
    for name, info in dense.items():
        if info['image_sha256'] != protocol['dense_image_hashes'][name] or info['label_sha256'] != protocol['dense_label_hashes'][Path(name).stem]:
            raise ValueError('Frozen dense data changed')
    dev = {n:dense[n] for n in protocol['dense_development']}
    test = {n:dense[n] for n in protocol['dense_evaluation']}
    sparse = load_folder(ROOT/'data/research_v2/combined/test')
    external = load_folder(Path('D:/pap_model/HMCHH_YOLO_prepared/valid'), abnormal_only=True)
    outputs = {}
    for seed in (17,43,101):
        for variant in ('sipakmed_only', 'combined'):
            name = f'v2_{variant}_seed{seed}'
            weights = OUT/'runs'/name/'weights/best.pt'
            print(f'Evaluating {name}', flush=True)
            preds = collect(weights, dense, protocol['inference'], name+'_dense')
            sweep = [evaluate_at_threshold(dev, preds, t/100) for t in range(1,66)]
            chosen = max(sweep, key=lambda v:v['abnormal_f1'])
            threshold = chosen['conf']
            counts = field_counts(test, preds, threshold)
            dev_records = correctness(dev, preds)
            test_records = correctness(test, preds)
            temperature = fit_temperature_scale(dev_records)
            external_preds = collect(weights, external, protocol['inference'], name+'_hmchh')
            sparse_preds = collect(weights, sparse, protocol['inference'], name+'_sparse')
            result = {'weights_sha256': digest(weights), 'threshold_from_development': threshold,
                      'development_optimum': chosen, 'development_sweep': sweep,
                      'dense_evaluation': summarize(counts), 'dense_per_field': counts,
                      'calibration': {'fit_fields': len(dev), 'evaluation_fields': len(test), 'temperature': temperature, 'raw': calibration_metrics(test_records), 'scaled': calibration_metrics(apply_temperature(test_records, temperature))},
                      'hmchh': summarize(field_counts(external, external_preds, threshold)),
                      'hmchh_per_field': field_counts(external, external_preds, threshold),
                      'sparse_test': {}}
            for source in ('sipakmed', 'apcdata'):
                subset = {n:v for n,v in sparse.items() if n.startswith(source+'__')}
                result['sparse_test'][source] = {'limitation': 'Incomplete annotations; precision is reference-annotation precision, not definitive true-cell precision', **summarize(field_counts(subset, sparse_preds, threshold))}
            outputs[name] = result
            write_json(OUT/f'{name}_evaluation.json', result)
    write_json(OUT/'evaluation_summary.json', {'protocol': protocol, 'models': outputs})
    print('All six models evaluated; paired comparisons and synthesis remain.', flush=True)


if __name__ == '__main__':
    main()
