"""Frozen retrospective evaluation, version 3 (overlap-aware splits).

Same protocol as research_v2 (development-only threshold, held-out calibration,
cluster bootstrap), with three corrections:
  * dense development/evaluation fields assigned by overlap component;
  * bootstrap resampling units are overlap components;
  * HMCHH reference rebuilt with abnormal cells only (organisms excluded,
    polygon annotations included).

    python evaluation/research_v3.py --freeze   # before any v3 model exists
    python evaluation/research_v3.py            # after all six runs complete
"""
import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.research_v2 import (digest, write_json, load_folder, collect, field_counts, summarize,
                                    correctness, calibration_metrics)
from evaluation.sweep_operating_thresholds import evaluate_at_threshold
from evaluation.calibration import fit_temperature_scale, apply_temperature

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/research_v3'
DATA = ROOT / 'data/research_v3'
DENSE = ROOT / 'data/dense_eval/sipakmed_dense40_verified'
HMCHH = Path('D:/pap_model/HMCHH_abnormal_reference_v3/valid')
CACHE = OUT / 'prediction_cache'
SEEDS, VARIANTS = (17, 43, 101), ('sipakmed_only', 'combined')


def freeze():
    path = OUT / 'evaluation_protocol.json'
    if path.exists():
        raise FileExistsError('Evaluation protocol already frozen')
    if (OUT / 'runs').exists():
        raise RuntimeError('Freeze must precede training')
    split = json.loads((OUT / 'dense_split.json').read_text())
    images = sorted((DENSE / 'images').glob('*.jpg'))
    hmchh = json.loads((OUT / 'hmchh_reference_summary.json').read_text(encoding='utf-8'))
    protocol = {
        'version': 3,
        'status': 'Retrospective research. Historical images previously examined; no prospective confirmation claim.',
        'dense_development': split['dense_development'], 'dense_evaluation': split['dense_evaluation'],
        'dense_components': split['dense_components'], 'dense_split_rule': split['rule'],
        'threshold_selection': 'Maximize abnormal F1 on dense development only; ties favor lower threshold. Thresholds 0.01..0.65 in 0.01 steps.',
        'calibration': 'Binary NLL temperature fit on development predictions; evaluated on dense evaluation predictions at floor 0.01. Conditional on detected candidates.',
        'inference': {'imgsz': 640, 'conf': 0.01, 'iou': 0.7, 'max_det': 300},
        'matching': 'Confidence-ordered greedy IoU >= 0.5; localization class-agnostic; abnormal metric matched separately to abnormal references.',
        'bootstrap': {'iterations': 1000, 'seed': 20260921, 'unit': 'dense: overlap components; HMCHH: filename slide prefix'},
        'external': {'dataset': 'HMCHH validation split, abnormal cells only', 'reference_cells': hmchh['abnormal_cells'],
                     'fields': hmchh['fields'], 'reference_summary_sha256': digest(OUT / 'hmchh_reference_summary.json')},
        'dense_image_hashes': {p.name: digest(p) for p in images},
        'dense_label_hashes': {p.stem: digest(p.parent.parent / 'labels' / (p.stem + '.txt')) for p in images},
        'training_manifest_sha256': digest(DATA / 'manifest.json'),
        'annotation_attestation_sha256': digest(ROOT / 'results/annotation_attestation_20260919.json'),
    }
    write_json(path, protocol)
    # Stain references: three disjoint seeded sets of 50 SIPaKMeD training fields.
    train = sorted((DATA / 'sipakmed_only/train/images').glob('*.jpg'))
    random.Random(20260922).shuffle(train)
    sets = [train[i*50:(i+1)*50] for i in range(3)]
    write_json(OUT / 'analysis_protocol.json', {
        'paired_comparisons': 'combined minus single-source within each matched seed; same fields and cluster draws',
        'seeds': list(SEEDS),
        'stain_references': 'Three disjoint sets of 50 v3 SIPaKMeD training fields after seeded shuffle 20260922',
        'stain_evaluation': 'All six models; dense evaluation fields and APCData sparse test; thresholds not retuned after normalization',
        'claims': 'Descriptive reference and seed sensitivity; no stain-invariance, noise-floor or unadjusted significance claims',
        'uncertainty': 'Percentile cluster bootstrap, 1000 draws, seed 20260921, conditional on fitted models',
        'reference_sets': [[str(p) for p in s] for s in sets],
        'reference_hashes': {str(p): digest(p) for s in sets for p in s},
    })
    print(f'Frozen v3: {len(split["dense_development"])} development / {len(split["dense_evaluation"])} evaluation fields')


def load_dense(protocol):
    dense = load_folder(DENSE)
    for name, info in dense.items():
        if info['image_sha256'] != protocol['dense_image_hashes'][name] or \
                info['label_sha256'] != protocol['dense_label_hashes'][Path(name).stem]:
            raise ValueError('Frozen dense data changed')
        info['group'] = 'component:' + protocol['dense_components'][name]
    return dense


def load_sparse_test():
    """Sparse held-out test; bootstrap unit is the overlap component from the manifest."""
    manifest = json.loads((DATA / 'manifest.json').read_text())
    component = {r['source'] + '__' + Path(r['image']).stem: r['group'] for r in manifest['records']}
    sparse = load_folder(DATA / 'combined/test')
    for name, info in sparse.items():
        info['group'] = component[Path(name).stem]
    return sparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze()
        return
    protocol = json.loads((OUT / 'evaluation_protocol.json').read_text())
    if digest(DATA / 'manifest.json') != protocol['training_manifest_sha256']:
        raise ValueError('Frozen training manifest changed')
    if digest(OUT / 'hmchh_reference_summary.json') != protocol['external']['reference_summary_sha256']:
        raise ValueError('Frozen external reference changed')
    dense = load_dense(protocol)
    dev = {n: dense[n] for n in protocol['dense_development']}
    test = {n: dense[n] for n in protocol['dense_evaluation']}
    sparse = load_sparse_test()
    external = load_folder(HMCHH, abnormal_only=True)
    if sum(len(v['boxes']) for v in external.values()) != protocol['external']['reference_cells']:
        raise ValueError('External reference count differs from frozen summary')
    outputs = {}
    for seed in SEEDS:
        for variant in VARIANTS:
            name = f'v3_{variant}_seed{seed}'
            weights = OUT / 'runs' / name / 'weights/best.pt'
            print(f'Evaluating {name}', flush=True)
            preds = collect(weights, dense, protocol['inference'], name + '_dense', cache_dir=CACHE)
            sweep = [evaluate_at_threshold(dev, preds, t / 100) for t in range(1, 66)]
            chosen = max(sweep, key=lambda v: v['abnormal_f1'])   # first maximum = lowest threshold
            threshold = chosen['conf']
            counts = field_counts(test, preds, threshold)
            dev_records, test_records = correctness(dev, preds), correctness(test, preds)
            temperature = fit_temperature_scale(dev_records)
            external_preds = collect(weights, external, protocol['inference'], name + '_hmchh', cache_dir=CACHE)
            sparse_preds = collect(weights, sparse, protocol['inference'], name + '_sparse', cache_dir=CACHE)
            hmchh_counts = field_counts(external, external_preds, threshold)
            result = {'weights_sha256': digest(weights), 'threshold_from_development': threshold,
                      'development_optimum': chosen, 'development_sweep': sweep,
                      'dense_evaluation': summarize(counts), 'dense_per_field': counts,
                      'calibration': {'fit_fields': len(dev), 'evaluation_fields': len(test), 'temperature': temperature,
                                      'raw': calibration_metrics(test_records),
                                      'scaled': calibration_metrics(apply_temperature(test_records, temperature))},
                      'hmchh': summarize(hmchh_counts), 'hmchh_per_field': hmchh_counts, 'sparse_test': {}}
            for source in ('sipakmed', 'apcdata'):
                subset = {n: v for n, v in sparse.items() if n.startswith(source + '__')}
                result['sparse_test'][source] = {
                    'limitation': 'Incomplete annotations; precision is reference-annotation precision, not true-cell precision',
                    **summarize(field_counts(subset, sparse_preds, threshold))}
            outputs[name] = result
            write_json(OUT / f'{name}_evaluation.json', result)
    write_json(OUT / 'evaluation_summary.json', {'protocol': protocol, 'models': outputs})
    print('All six v3 models evaluated.', flush=True)


if __name__ == '__main__':
    main()
