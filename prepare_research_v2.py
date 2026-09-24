"""Build conservative, immutable source-grouped binary research splits.

Quarantine mirror images without canonical SIPaKMeD field filenames and APCData
images without recoverable specimen identifiers. Reserve all 40 dense fields.
These are retrospective field/specimen-proxy groups, not verified patient IDs.
"""
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
import yaml
from prepare_combined_2class import convert_label, place
from evaluation.taxonomy import binary_index_map

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data/research_v2'


def group_id(source, name):
    if source == 'sipakmed':
        m = re.match(r'((?:dysk|koil|meta|para|supe)_\d+)_bmp', name, re.I)
        return source + ':' + m.group(1).lower() if m else None
    stem = re.sub(r'^\d+\s+', '', Path(name).stem)
    m = re.search(r'\b([CK])\s*(\d+)', stem, re.I)
    if m:
        return f'{source}:{m.group(1).upper()}{int(m.group(2))}'
    m = re.search(r'\b(\d{3,})\s+\d{2}\b', stem)
    return f'{source}:bare{int(m.group(1))}' if m else None


def main():
    if OUT.exists():
        raise FileExistsError(f'{OUT} exists; do not silently change a frozen research split')
    dense_root = ROOT / 'data/dense_eval/sipakmed_dense40_verified'
    dense_names = {p.name for p in (dense_root / 'images').glob('*.jpg')}
    reserved = {group_id('sipakmed', n) for n in dense_names} - {None}
    records, quarantine = [], []
    for source, relative in [('sipakmed', 'sipakmed/mirror_a'), ('apcdata', 'apcdata/APCData_YOLO_prepared')]:
        root = ROOT / 'data' / relative
        mapping = binary_index_map(yaml.safe_load((root / 'data.yaml').read_text())['names'])
        for old_split in ('train', 'valid'):
            for image in sorted((root / old_split / 'images').glob('*.jpg')):
                group = group_id(source, image.name)
                label = image.parent.parent / 'labels' / (image.stem + '.txt')
                if not group:
                    quarantine.append({'source': source, 'image': str(image.relative_to(ROOT)), 'reason': 'Unresolved source/specimen provenance'})
                    continue
                lines = convert_label(label, mapping)
                records.append({'source': source, 'image': str(image.relative_to(ROOT)),
                                'label': str(label.relative_to(ROOT)), 'group': group,
                                'sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
                                'label_sha256': hashlib.sha256(label.read_bytes()).hexdigest(),
                                'lines': lines, 'stratum': Counter(x.split()[0] for x in lines).most_common(1)[0][0] if lines else 'empty'})
    # Merge exact duplicates into common groups, including cross-source duplicates.
    by_hash = defaultdict(list)
    for item in records:
        by_hash[item['sha256']].append(item)
    for items in by_hash.values():
        groups = {x['group'] for x in items}
        if len(groups) > 1:
            raise ValueError(f'Cross-group duplicate needs adjudication: {groups}')
    grouped = defaultdict(list)
    for item in records:
        grouped[item['group']].append(item)
    strata = defaultdict(list)
    assignment = {g: 'dense_reserved' for g in reserved}
    for group, items in grouped.items():
        if group in reserved:
            continue
        dominant = Counter(x['stratum'] for x in items).most_common(1)[0][0]
        strata[(items[0]['source'], dominant)].append(group)
    rng = random.Random(20260919)
    for _, groups in sorted(strata.items()):
        groups.sort()
        rng.shuffle(groups)
        count = max(1, round(.15 * len(groups)))
        for i, group in enumerate(groups):
            assignment[group] = 'test' if i < count else 'valid' if i < 2 * count else 'train'
    for item in records:
        item['split'] = assignment[item['group']]
    manifest = {
        'split_seed': 20260919,
        'status': 'Retrospective source-grouped experiment; not a prospective or patient-independent clinical test',
        'grouping': 'SIPaKMeD canonical field stem; APCData specimen-code proxy ignoring year conservatively. Patient identities unavailable.',
        'dense_evaluation': 'All 40 user-reviewed fields reserved from new training; noncanonical sources excluded from new training entirely',
        'quarantined': quarantine,
        'records': [{k:v for k,v in x.items() if k != 'lines'} for x in records],
    }
    OUT.mkdir(parents=True)
    for variant, sources in [('sipakmed_only', {'sipakmed'}), ('combined', {'sipakmed', 'apcdata'})]:
        root = OUT / variant
        for split in ('train', 'valid', 'test'):
            (root / split / 'images').mkdir(parents=True)
            (root / split / 'labels').mkdir(parents=True)
        for item in records:
            if item['source'] not in sources or item['split'] == 'dense_reserved':
                continue
            image = ROOT / item['image']
            stem = item['source'] + '__' + image.stem
            place(image, root / item['split'] / 'images' / (stem + image.suffix))
            (root / item['split'] / 'labels' / (stem + '.txt')).write_text('\n'.join(item['lines']) + '\n')
        (root / 'data.yaml').write_text(yaml.safe_dump({'path': str(root), 'train': 'train/images', 'val': 'valid/images', 'test': 'test/images', 'names': ['Normal', 'Abnormal']}))
    manifest['counts'] = dict(Counter(x['source'] + '/' + x['split'] for x in records))
    manifest['quarantine_counts'] = dict(Counter(x['source'] for x in quarantine))
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    tracked = ROOT / 'results/research_v2'
    tracked.mkdir(exist_ok=True)
    (tracked / 'data_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('records', 'quarantined')}, indent=2))


if __name__ == '__main__':
    main()
