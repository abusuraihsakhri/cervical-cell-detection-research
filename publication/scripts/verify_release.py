"""Release verification: data manifest, split independence, dense references,
checkpoints and result provenance. Exits nonzero on any failure.

Usage:
    python publication/scripts/verify_release.py            # everything
    python publication/scripts/verify_release.py --data     # data checks only
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.taxonomy import binary_index_map

V2_DATA = ROOT / 'data/research_v3'          # names kept short; these are the v3 release paths
V2_RESULTS = ROOT / 'results/research_v3'
HMCHH = Path('D:/pap_model/HMCHH_abnormal_reference_v3/valid')
HISTORICAL = {'phase1_sipakmed_5class': ROOT / 'models/best.pt',
              'expA_apcdata': ROOT / 'results/runs/expA_apcdata/weights/best.pt',
              'expC1_sipakmed_2class': ROOT / 'results/runs/expC1_sipakmed_2class/weights/best.pt',
              'expC2b_combined': ROOT / 'results/runs/expC2b_combined_150/weights/best.pt'}
SOURCE_ROOTS = {'sipakmed': ROOT / 'data/sipakmed/mirror_a', 'apcdata': ROOT / 'data/apcdata/APCData_YOLO_prepared'}
SEEDS, VARIANTS = (17, 43, 101), ('sipakmed_only', 'combined')
report = {'checks': [], 'failures': []}


if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(name, ok, detail=None):
    report['checks'].append({'check': name, 'passed': bool(ok), 'detail': detail})
    if not ok:
        report['failures'].append(name)
    detail_str = f': {detail}' if detail is not None else ''
    try:
        print(f'[{"PASS" if ok else "FAIL"}] {name}{detail_str}', flush=True)
    except UnicodeEncodeError:
        safe_str = f'[{"PASS" if ok else "FAIL"}] {name}{detail_str}'.encode('ascii', errors='backslashreplace').decode('ascii')
        print(safe_str, flush=True)


def label_boxes(label, mapping):
    """Return (converted lines, source line count) without silently dropping anything."""
    converted, source = [], 0
    for line in label.read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if not parts:
            continue
        source += 1
        values = [float(v) for v in parts[1:]]
        if len(values) == 4:
            cx, cy, w, h = values
        else:
            xs, ys = values[0::2], values[1::2]
            cx, cy, w, h = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2, max(xs)-min(xs), max(ys)-min(ys)
        converted.append((mapping[int(float(parts[0]))], cx, cy, w, h))
    return converted, source


def dhash(path, size=8):
    image = Image.open(path).convert('L').resize((size + 1, size), Image.LANCZOS)
    pixels = np.asarray(image, dtype=np.int16)
    return np.packbits((pixels[:, 1:] > pixels[:, :-1]).flatten())


def verify_data():
    manifest = json.loads((V2_DATA / 'manifest.json').read_text())
    check('tracked data manifest matches data/research_v3 manifest',
          json.loads((V2_RESULTS / 'data_manifest.json').read_text()) == manifest)
    mappings = {s: binary_index_map(yaml.safe_load((r / 'data.yaml').read_text())['names']) for s, r in SOURCE_ROOTS.items()}
    hash_errors, dropped, mismatch, geometry = [], [], [], []
    expected = defaultdict(set)
    for record in manifest['records']:
        image, label = ROOT / record['image'], ROOT / record['label']
        if sha(image) != record['sha256'] or sha(label) != record['label_sha256']:
            hash_errors.append(record['image'])
            continue
        boxes, source_lines = label_boxes(label, mappings[record['source']])
        if len(boxes) != source_lines:
            dropped.append(record['image'])
        if any(not (0 < w <= 1 and 0 < h <= 1 and 0 <= cx <= 1 and 0 <= cy <= 1) for _, cx, cy, w, h in boxes):
            geometry.append(record['image'])
        if record['split'] == 'dense_reserved':
            continue
        for variant in VARIANTS:
            if variant == 'sipakmed_only' and record['source'] != 'sipakmed':
                continue
            stem = record['source'] + '__' + image.stem
            out_image = V2_DATA / variant / record['split'] / 'images' / (stem + image.suffix)
            out_label = V2_DATA / variant / record['split'] / 'labels' / (stem + '.txt')
            expected[(variant, record['split'])].add(out_image.name)
            written = [tuple(float(v) for v in l.split()) for l in out_label.read_text().splitlines() if l.strip()]
            if not out_image.is_file() or sha(out_image) != record['sha256'] or \
                    len(written) != len(boxes) or any(int(w[0]) != b[0] or max(abs(x - y) for x, y in zip(w[1:], b[1:])) > 2e-6 for w, b in zip(written, boxes)):
                mismatch.append(str(out_label.relative_to(ROOT)))
    check('source image and label hashes match manifest', not hash_errors, f'{len(manifest["records"])} records; {len(hash_errors)} mismatched')
    check('no source boxes dropped during binary conversion', not dropped, f'{len(dropped)} files with dropped lines')
    check('all converted boxes have valid normalized geometry', not geometry, f'{len(geometry)} files')
    check('split files equal a fresh re-conversion of the manifest', not mismatch, f'{len(mismatch)} mismatches')
    extras = []
    for (variant, split), names in expected.items():
        on_disk = {p.name for p in (V2_DATA / variant / split / 'images').iterdir() if p.suffix.lower() != '.npy'}
        extras += sorted(on_disk ^ names)
    check('split folders contain exactly the manifest records', not extras, f'{len(extras)} unexpected or missing files')
    split_of_group = defaultdict(set)
    for record in manifest['records']:
        split_of_group[record['group']].add(record['split'])
    check('every overlap component lies in exactly one split', all(len(s) == 1 for s in split_of_group.values()),
          f'{len(split_of_group)} groups')
    by_hash = defaultdict(set)
    for record in manifest['records']:
        by_hash[record['sha256']].add(record['split'])
    check('no exact duplicate image crosses splits', all(len(s) == 1 for s in by_hash.values()))
    overlap_p = V2_RESULTS / 'field_overlap.json'
    if not overlap_p.exists() and (V2_RESULTS / 'field_overlap.json.gz').exists():
        import gzip
        raw_overlap_bytes = gzip.decompress((V2_RESULTS / 'field_overlap.json.gz').read_bytes())
        overlap = json.loads(raw_overlap_bytes.decode('utf-8'))
        check('overlap evidence matches manifest hash', hashlib.sha256(raw_overlap_bytes).hexdigest() == manifest['overlap_evidence_sha256'])
    else:
        overlap = json.loads(overlap_p.read_text(encoding='utf-8'))
        check('overlap evidence matches manifest hash', sha(overlap_p) == manifest['overlap_evidence_sha256'])
    split_of = {r['image']: r['split'] for r in manifest['records']}
    crossing = [p for p in overlap['pairs'] if p['overlap'] and p['a'] in split_of and p['b'] in split_of
                and split_of[p['a']] != split_of[p['b']]]
    check('no confirmed overlapping field pair crosses splits', not crossing,
          f'{sum(p["overlap"] for p in overlap["pairs"])} confirmed pairs; {len(crossing)} crossing')
    photometric = json.loads((V2_RESULTS / 'overlap_photometric.json').read_text())['pairs']
    scored = [p['region_ncc'] for p in photometric if p['region_ncc'] is not None]
    check('confirmed overlaps agree photometrically (shared-region NCC >= 0.9)',
          len(photometric) == sum(p['overlap'] for p in overlap['pairs']) and min(scored) >= 0.9,
          f'{len(scored)}/{len(photometric)} scored; minimum NCC {min(scored):.3f}')

    # Near-duplicate screen: 64-bit difference hash, Hamming distance <= 4 across different splits.
    hashes = []
    for record in manifest['records']:
        hashes.append((record['split'], record['group'], record['image'], dhash(ROOT / record['image'])))
    bits = np.unpackbits(np.stack([h[3] for h in hashes]), axis=1).astype(np.uint8)
    near = []
    for i in range(len(hashes)):
        distance = np.count_nonzero(bits[i+1:] != bits[i], axis=1)
        for j in np.nonzero(distance <= 4)[0] + i + 1:
            if hashes[i][0] != hashes[j][0]:
                near.append({'a': hashes[i][2], 'b': hashes[j][2], 'splits': [hashes[i][0], hashes[j][0]], 'hamming': int(distance[j-i-1])})
    report['near_duplicates_across_splits'] = near
    check('no perceptual near-duplicate (dHash Hamming <= 4) crosses splits', not near, f'{len(near)} pairs')

    protocol = json.loads((V2_RESULTS / 'evaluation_protocol.json').read_text())
    dense = ROOT / 'data/dense_eval/sipakmed_dense40_verified'
    changed = [n for n, h in protocol['dense_image_hashes'].items() if sha(dense / 'images' / n) != h]
    changed += [s for s, h in protocol['dense_label_hashes'].items() if sha(dense / 'labels' / f'{s}.txt') != h]
    check('dense images and labels match frozen protocol hashes', not changed, f'{len(changed)} changed')
    check('dense development and evaluation fields are disjoint and cover all 40',
          not set(protocol['dense_development']) & set(protocol['dense_evaluation'])
          and len(set(protocol['dense_development']) | set(protocol['dense_evaluation'])) == 40,
          f'{len(protocol["dense_development"])} development / {len(protocol["dense_evaluation"])} evaluation')
    dense_hashes = set(protocol['dense_image_hashes'].values())
    trained = {r['sha256'] for r in manifest['records'] if r['split'] != 'dense_reserved'}
    dense_groups = {'component:' + c for c in protocol['dense_components'].values()}
    trained_groups = {r['group'] for r in manifest['records'] if r['split'] != 'dense_reserved'}
    check('no dense field image or overlap component used in training/validation/test',
          not (dense_hashes & trained) and not (dense_groups & trained_groups))
    comp_dev = {protocol['dense_components'][n] for n in protocol['dense_development']}
    comp_eval = {protocol['dense_components'][n] for n in protocol['dense_evaluation']}
    check('no overlap component spans dense development and evaluation', not comp_dev & comp_eval)
    dense_bits = np.unpackbits(np.stack([dhash(dense / 'images' / n) for n in protocol['dense_image_hashes']]), axis=1)
    trained_bits = bits[[i for i, h in enumerate(hashes) if h[0] != 'dense_reserved']]
    near_dense = int(sum((np.count_nonzero(trained_bits != row, axis=1) <= 4).any() for row in dense_bits))
    check('no dense field has a perceptual near-duplicate in split data', near_dense == 0, f'{near_dense} fields')
    export = json.loads((ROOT / 'results/annotation_attestation_20260919.json').read_text())['export_sha256']
    check('dense Label Studio export matches attested hash', sha(dense / 'labelstudio_verified_export.json') == export)
    boxes = Counter()
    for label in (dense / 'labels').glob('*.txt'):
        for line in label.read_text().splitlines():
            if line.strip():
                boxes[int(line.split()[0])] += 1
    check('dense reference totals 1,067 cells (706 normal, 361 abnormal)', boxes == Counter({0: 706, 1: 361}), dict(boxes))

    summary = json.loads((V2_RESULTS / 'hmchh_reference_summary.json').read_text(encoding='utf-8'))
    bad = [n for n, v in summary['per_field'].items() if sha(HMCHH / 'images' / n) != v['image_sha256']
           or sha(HMCHH / 'labels' / (Path(n).stem + '.txt')) != v['label_sha256']]
    total = sum(len([l for l in (HMCHH / 'labels' / (Path(n).stem + '.txt')).read_text().splitlines() if l.strip()])
                for n in summary['per_field'])
    check('HMCHH reference files match frozen summary', not bad and total == summary['abnormal_cells'] == protocol['external']['reference_cells'],
          f'{summary["fields"]} fields, {total} abnormal cells')
    check('HMCHH reference excludes organism and flora categories', set(summary['kept_by_name']) <= {'异常', '异常，菌群失调'},
          {'excluded': summary['excluded_by_name']})


def verify_checkpoints():
    import torch
    from ultralytics import YOLO
    manifest_hash = sha(V2_DATA / 'manifest.json')
    status = json.loads((V2_RESULTS / 'training_status.json').read_text())
    check('training protocol bound to current data manifest', status['protocol']['manifest_sha256'] == manifest_hash)
    report['checkpoints'] = {}
    for seed in SEEDS:
        for variant in VARIANTS:
            name = f'v3_{variant}_seed{seed}'
            run = V2_RESULTS / 'runs' / name
            weights = run / 'weights/best.pt'
            if not weights.is_file():
                check(f'{name}: checkpoint exists', False)
                continue
            if not (run / 'weights/last.pt').is_file():
                check(f'{name}: last.pt present (training finished normally)', False)
            args = yaml.safe_load((run / 'args.yaml').read_text())
            expected_data = V2_DATA / variant / 'data.yaml'
            model = YOLO(str(weights))
            names = dict(model.names)
            ckpt = getattr(model, 'ckpt', {}) or {}
            rows = [l.split(',') for l in (run / 'results.csv').read_text().splitlines()]
            header = [h.strip() for h in rows[0]]
            data = np.array([[float(v) for v in r] for r in rows[1:]])
            # Ultralytics 8.4.31 Metric.fitness weights [P, R, mAP50, mAP50-95] = [0, 0, 0, 1]
            # (source saved in results/research_v2/training_fitness_source.txt).
            fitness = data[:, header.index('metrics/mAP50-95(B)')]
            best_epoch = int(data[int(np.argmax(fitness)), header.index('epoch')])
            summary = json.loads((ROOT / 'results' / f'{name}_training_summary.json').read_text())
            ok = (args['seed'] == seed and Path(args['data']).resolve() == expected_data.resolve()
                  and args['deterministic'] and args['epochs'] == 150 and args['patience'] == 30
                  and names == {0: 'Normal', 1: 'Abnormal'} and ckpt.get('epoch') is not None
                  and summary['seed'] == seed and Path(summary['save_dir']).resolve() == run.resolve())
            info = {'sha256': sha(weights), 'size_bytes': weights.stat().st_size, 'seed': args['seed'],
                    'data': str(expected_data.relative_to(ROOT)), 'epochs_run': len(data), 'best_epoch_by_fitness': best_epoch,
                    'best_mAP50': float(data[int(np.argmax(fitness)), header.index('metrics/mAP50(B)')]),
                    'best_mAP50_95': float(data[int(np.argmax(fitness)), header.index('metrics/mAP50-95(B)')]),
                    'class_names': names, 'initial_weights_sha256': summary['initial_weights_sha256'],
                    'batch': summary['batch_used'], 'wall_clock_minutes': round(summary['total_wall_clock_seconds'] / 60, 1)}
            report['checkpoints'][name] = info
            check(f'{name}: seed, data split, class names, protocol arguments and summary consistent', ok,
                  f'best epoch {best_epoch}/{len(data)}, sha256 {info["sha256"][:12]}')
    report['historical_checkpoints'] = {}
    for name, path in HISTORICAL.items():
        if path.is_file():
            model = YOLO(str(path))
            report['historical_checkpoints'][name] = {'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
                                                      'class_names': dict(model.names), 'status': 'superseded; not released as a result'}
    evaluation = V2_RESULTS / 'evaluation_summary.json'
    if evaluation.exists():
        models = json.loads(evaluation.read_text())['models']
        stale = [n for n, m in models.items() if m['weights_sha256'] != report['checkpoints'].get(n, {}).get('sha256')]
        check('every evaluation result was produced from the released checkpoint', not stale and len(models) == 6, stale)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', action='store_true', help='Data checks only')
    args = parser.parse_args()
    verify_data()
    if not args.data:
        verify_checkpoints()
    out = ROOT / 'publication/reproducibility/release_verification.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    print(f'\n{len(report["checks"]) - len(report["failures"])}/{len(report["checks"])} checks passed')
    sys.exit(1 if report['failures'] else 0)


if __name__ == '__main__':
    main()
