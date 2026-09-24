"""Build overlap-aware, source-grouped binary research splits (version 3).

Supersedes research_v2, whose filename grouping missed re-captured fields: the
SIPaKMeD derivative contains shifted/refocused captures of one physical field
under different image numbers (see publication/scripts/detect_field_overlap.py).

Grouping unit: connected components of (filename source group) joined by
confirmed image-overlap edges. Components containing any dense reference field
are withheld from training, validation and test. The dense development and
evaluation fields are re-frozen so that no component crosses them. Everything
is written before any v3 model exists.
"""
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from prepare_combined_2class import convert_label, place
from prepare_research_v2 import group_id
from evaluation.taxonomy import binary_index_map

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data/research_v3'
RESULTS = ROOT / 'results/research_v3'
DENSE = ROOT / 'data/dense_eval/sipakmed_dense40_verified'
SOURCES = [('sipakmed', 'data/sipakmed/mirror_a'), ('apcdata', 'data/apcdata/APCData_YOLO_prepared')]
SPLIT_SEED, DENSE_SEED = 20260919, 20260920


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Components:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def main():
    if OUT.exists():
        raise FileExistsError(f'{OUT} exists; do not silently change a frozen research split')
    overlap = json.loads((RESULTS / 'field_overlap.json').read_text())
    comps = Components()
    fields = []   # every field on disk, including quarantined ones, joins the overlap graph
    for source, relative in SOURCES:
        root = ROOT / relative
        for split in ('train', 'valid'):
            for image in sorted((root / split / 'images').glob('*.jpg')):
                key = str(image.relative_to(ROOT))
                group = group_id(source, image.name)
                comps.find(key)
                if group:
                    comps.union(key, 'name:' + group)
                fields.append((source, root, image, group))
    for pair in overlap['pairs']:
        if pair['overlap']:
            comps.union(pair['a'], pair['b'])
    # Exact duplicate bytes are always one component.
    by_hash = defaultdict(list)
    for _, _, image, _ in fields:
        by_hash[sha(image)].append(str(image.relative_to(ROOT)))
    for keys in by_hash.values():
        for key in keys[1:]:
            comps.union(keys[0], key)

    dense_names = sorted(p.name for p in (DENSE / 'images').glob('*.jpg'))
    mirror_valid = ROOT / 'data/sipakmed/mirror_a/valid/images'
    dense_key = {}
    for name in dense_names:
        if sha(DENSE / 'images' / name) != sha(mirror_valid / name):
            raise ValueError(f'Dense field {name} differs from its mirror source image')
        dense_key[name] = str((mirror_valid / name).relative_to(ROOT))
    dense_components = {comps.find(k) for k in dense_key.values()}

    records, quarantine = [], []
    for source, root, image, group in fields:
        key = str(image.relative_to(ROOT))
        if not group:
            quarantine.append({'source': source, 'image': key, 'reason': 'Unresolved source/specimen provenance'})
            continue
        mapping = binary_index_map(yaml.safe_load((root / 'data.yaml').read_text())['names'])
        label = image.parent.parent / 'labels' / (image.stem + '.txt')
        lines = convert_label(label, mapping)
        records.append({'source': source, 'image': key, 'label': str(label.relative_to(ROOT)),
                        'filename_group': group, 'group': 'component:' + comps.find(key),
                        'sha256': sha(image), 'label_sha256': sha(label), 'lines': lines,
                        'stratum': Counter(x.split()[0] for x in lines).most_common(1)[0][0] if lines else 'empty'})

    grouped = defaultdict(list)
    for item in records:
        grouped[item['group']].append(item)
    assignment = {}
    strata = defaultdict(list)
    for group, items in grouped.items():
        if group.removeprefix('component:') in dense_components:
            assignment[group] = 'dense_reserved'
            continue
        sources = sorted({x['source'] for x in items})
        dominant = Counter(x['stratum'] for x in items).most_common(1)[0][0]
        strata[('+'.join(sources), dominant)].append(group)
    rng = random.Random(SPLIT_SEED)
    for _, groups in sorted(strata.items()):
        groups.sort()
        rng.shuffle(groups)
        count = max(1, round(.15 * len(groups)))
        for i, group in enumerate(groups):
            assignment[group] = 'test' if i < count else 'valid' if i < 2 * count else 'train'
    for item in records:
        item['split'] = assignment[item['group']]

    # Re-freeze dense development/evaluation at component level. Unresolved-source
    # dense fields stay in development, as in v2; canonical components fill the rest.
    dense_comp = {name: comps.find(dense_key[name]) for name in dense_names}
    unresolved = {dense_comp[n] for n in dense_names if not group_id('sipakmed', n)}
    other = sorted({dense_comp[n] for n in dense_names} - unresolved)
    random.Random(DENSE_SEED).shuffle(other)
    development_comps = set(unresolved)
    for comp in other:
        if sum(dense_comp[n] in development_comps for n in dense_names) >= 20:
            break
        development_comps.add(comp)
    development = [n for n in dense_names if dense_comp[n] in development_comps]
    evaluation = [n for n in dense_names if dense_comp[n] not in development_comps]

    manifest = {
        'version': 3,
        'split_seed': SPLIT_SEED,
        'status': 'Retrospective overlap-aware source-grouped experiment; not a prospective or patient-independent clinical test',
        'grouping': 'Connected components of filename source groups (SIPaKMeD canonical field stem; APCData specimen code) joined by confirmed image overlap and exact duplicates. Patient identities unavailable.',
        'overlap_evidence_sha256': sha(RESULTS / 'field_overlap.json'),
        'dense_evaluation': 'Every component containing a dense reference field is withheld from training, validation and test',
        'quarantined': quarantine,
        'records': [{k: v for k, v in x.items() if k != 'lines'} for x in records],
        'counts': dict(Counter(x['source'] + '/' + x['split'] for x in records)),
        'quarantine_counts': dict(Counter(x['source'] for x in quarantine)),
        'components': {'total': len(grouped), 'multi_field': sum(len(v) > 1 for v in grouped.values()),
                       'largest': max(len(v) for v in grouped.values()),
                       'filename_groups_merged_by_overlap': len({x['filename_group'] for x in records}) - len(grouped)},
    }
    dense_protocol = {
        'split_seed': DENSE_SEED,
        'rule': 'Dense fields assigned by overlap component. Unresolved-source components in development; seeded canonical components added until development holds >= 20 fields; remainder is evaluation.',
        'dense_development': development, 'dense_evaluation': evaluation,
        'dense_components': {n: dense_comp[n] for n in dense_names},
    }
    assert not {dense_comp[n] for n in development} & {dense_comp[n] for n in evaluation}

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
        (root / 'data.yaml').write_text(yaml.safe_dump({'path': str(root), 'train': 'train/images', 'val': 'valid/images',
                                                        'test': 'test/images', 'names': ['Normal', 'Abnormal']}))
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'data_manifest.json').write_text(json.dumps(manifest, indent=2))
    (RESULTS / 'dense_split.json').write_text(json.dumps(dense_protocol, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k not in ('records', 'quarantined')}, indent=2))
    print(f'Dense development {len(development)} / evaluation {len(evaluation)} fields')


if __name__ == '__main__':
    main()
