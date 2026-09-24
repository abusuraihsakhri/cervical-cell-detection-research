"""Independent audit: read existing artifacts; write only separate audit evidence."""
import sys
from pathlib import Path
import json
import hashlib
import re
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ultralytics import YOLO
from evaluation.sweep_operating_thresholds import load_ground_truth, evaluate_at_threshold


def main():
    model = YOLO(str(ROOT / 'models/best.pt'))
    gt, _ = load_ground_truth(ROOT / 'data/dense_eval/sipakmed_dense40_verified/data.yaml')
    predictions = {}
    for name, info in gt.items():
        res = model.predict(str(info['path']), conf=0.01, imgsz=640, verbose=False)[0]
        predictions[name] = [
            {'bbox': tuple(box), 'conf': float(conf),
             'cls': int(model.names[int(cls)] in ('Dyskeratotic', 'Koilocytotic'))}
            for box, conf, cls in zip(res.boxes.xyxyn.cpu().numpy(),
                                      res.boxes.conf.cpu().numpy(),
                                      res.boxes.cls.cpu().numpy())
        ]
    scores = [evaluate_at_threshold(gt, predictions, t / 100) for t in range(1, 66)]
    evidence = {
        'scope': 'Audit evidence only; original code and reports unchanged. Corrected baseline uses existing sweep matcher and existing labels, not independent ground truth.',
        'phase1_checkpoint_sha256': hashlib.sha256((ROOT / 'models/best.pt').read_bytes()).hexdigest(),
        'phase1_model_names': model.names,
        'phase1_corrected_at_004': evaluate_at_threshold(gt, predictions, 0.04),
        'phase1_corrected_at_005': evaluate_at_threshold(gt, predictions, 0.05),
        'phase1_corrected_at_0111': evaluate_at_threshold(gt, predictions, 0.111),
        'phase1_corrected_best_abnormal_f1_exploratory': max(scores, key=lambda x: x['abnormal_f1']),
    }
    root = ROOT / 'data/sipakmed/mirror_a'
    images = {s: sorted((root / s / 'images').glob('*.jpg')) for s in ('train', 'valid')}
    evidence['split_image_counts'] = {s: len(v) for s, v in images.items()}
    hashes = {s: {hashlib.sha256(p.read_bytes()).hexdigest() for p in v} for s, v in images.items()}
    evidence['exact_file_hash_overlap'] = len(hashes['train'] & hashes['valid'])
    def tiles(paths):
        out = []
        for p in paths:
            m = re.match(r'(.+?)_(\d+)_(\d+)_(\d+)_(\d+)_jpg\.rf\.', p.name)
            if m:
                out.append((p.name, m.group(1), tuple(map(int, m.groups()[1:]))))
        return out
    pairs = []
    for vn, vf, v in tiles(images['valid']):
        for tn, tf, t in tiles(images['train']):
            if vf != tf:
                continue
            inter = max(0, min(v[2], t[2]) - max(v[0], t[0])) * max(0, min(v[3], t[3]) - max(v[1], t[1]))
            if inter:
                pairs.append({'validation': vn, 'train': tn, 'fraction_validation_area': inter / ((v[2]-v[0])*(v[3]-v[1]))})
    evidence['filename_coordinate_overlap_note'] = 'Inferred from shared source prefixes and coordinate suffixes; verify against original source images.'
    evidence['filename_coordinate_overlap_pairs'] = pairs
    evidence['validation_tiles_with_coordinate_overlap'] = len({p['validation'] for p in pairs})
    evidence['dense_fields_with_coordinate_overlap'] = len(set(gt) & {p['validation'] for p in pairs})
    export = json.loads((ROOT / 'data/dense_eval/sipakmed_dense40_verified/labelstudio_verified_export.json').read_text())
    evidence['annotation_origin_counts'] = dict(Counter(x.get('origin', 'missing') for t in export for a in t['annotations'] for x in a['result']))
    evidence['annotation_metadata_keys_except_result'] = sorted({k for t in export for a in t['annotations'] for k in a if k != 'result'})
    path = ROOT / 'results/critical_audit_evidence_20260919.json'
    path.write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in evidence.items() if k != 'filename_coordinate_overlap_pairs'}, indent=2))


if __name__ == '__main__':
    main()
