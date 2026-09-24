"""Detect microscope fields that show the same physical specimen region.

Filename grouping misses re-captures of one field (shifted, refocused or
renumbered). Every pair of fields from the same source is matched exhaustively;
a thumbnail correlation screen was tried first but partial overlaps kept
appearing at its lower bound. Cross-source pairs are matched only when their
shift-tolerant thumbnail correlation exceeds SCREEN_CORR (all cross-source
candidates found at 0.45 were artifacts). Confirmation uses ORB keypoints and a
RANSAC similarity transform; a pair overlaps when enough geometrically
consistent inliers exist at the same magnification (scale 0.8-1.25).

Output: results/research_v3/field_overlap.json (pairs, scores, inliers).
"""
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/research_v3/field_overlap.json'
SCREEN_CORR = 0.35          # cross-source pairs only; within-source pairs are matched exhaustively
RECORD_MIN_INLIERS = 10     # pairs below this and below SCREEN_CORR are counted, not stored
MIN_INLIERS = 20            # stage 2: geometric confirmation. Calibrated on 7 visually
MIN_INLIER_RATIO = 0.70     # confirmed overlaps (26-1768 inliers, ratio >= 0.89) versus
                            # 60 random pairs (max 17 inliers, ratio <= 0.55).
NEAR_IDENTICAL_CORR = 0.97  # refocused re-captures can yield few ORB features
SCALE_RANGE = (0.8, 1.25)   # same magnification; degenerate fits (scale ~0) come from static
                            # sensor/optics artifacts and matched fields across unrelated sources


def images():
    """All candidate fields: SIPaKMeD mirror (train+valid), APCData, dense set."""
    paths = []
    for root in ('data/sipakmed/mirror_a', 'data/apcdata/APCData_YOLO_prepared'):
        for split in ('train', 'valid'):
            paths += sorted((ROOT / root / split / 'images').glob('*.jpg'))
    return paths


def thumbnail(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(image, (64, 48), interpolation=cv2.INTER_AREA).astype(np.float64)
    return (small - small.mean()) / (small.std() + 1e-6)


def source(path):
    return 'apcdata' if 'apcdata' in str(path).lower() else 'sipakmed'


def screen(paths):
    spectra = np.stack([np.fft.rfft2(thumbnail(p)) for p in paths])
    sources = np.array([source(p) for p in paths])
    candidates = []
    for i in range(len(paths) - 1):
        corr = np.fft.irfft2(spectra[i+1:] * np.conj(spectra[i]), s=(48, 64)) / (48 * 64)
        best = corr.reshape(len(corr), -1).max(axis=1)
        keep = (sources[i+1:] == sources[i]) | (best > SCREEN_CORR)
        for j in np.nonzero(keep)[0]:
            candidates.append((i, i + 1 + int(j), float(best[j])))
    return candidates


def features(path, orb):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (1024, int(1024 * image.shape[0] / image.shape[1])), interpolation=cv2.INTER_AREA)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    # Keep coordinates only; cv2.KeyPoint objects are large when cached for every field.
    return np.float32([k.pt for k in keypoints]).reshape(-1, 2), descriptors


def confirm(fa, fb, matcher):
    (ka, da), (kb, db) = fa, fb
    if da is None or db is None or len(ka) < 10 or len(kb) < 10:
        return 0, 0.0, 0.0
    pairs = matcher.knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return 0, 0.0, 0.0
    src = ka[[m.queryIdx for m in good]]
    dst = kb[[m.trainIdx for m in good]]
    transform, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=6.0)
    if transform is None:
        return 0, 0.0, 0.0
    inliers = int(mask.sum())
    return inliers, inliers / len(good), float(np.hypot(transform[0, 0], transform[1, 0]))


def confirm_block(args):
    """Worker: confirm a contiguous block of candidates with a local feature cache.

    ORB detection is deterministic, so results do not depend on the partition.
    """
    paths, block = args
    cv2.setNumThreads(1)
    orb = cv2.ORB_create(nfeatures=3000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    cache, out = {}, []
    for i, j, corr in block:
        for k in (i, j):
            if k not in cache:
                cache[k] = features(paths[k], orb)
        inliers, ratio, scale = confirm(cache[i], cache[j], matcher)
        geometric = inliers >= MIN_INLIERS and ratio >= MIN_INLIER_RATIO and SCALE_RANGE[0] <= scale <= SCALE_RANGE[1]
        overlap = geometric or corr >= NEAR_IDENTICAL_CORR
        if not (overlap or inliers >= RECORD_MIN_INLIERS or corr >= SCREEN_CORR):
            continue
        out.append({'a': str(paths[i].relative_to(ROOT)), 'b': str(paths[j].relative_to(ROOT)),
                    'correlation': round(corr, 4), 'inliers': inliers, 'inlier_ratio': round(ratio, 3),
                    'scale': round(scale, 4), 'overlap': overlap})
    return out


def main():
    from multiprocessing import Pool, cpu_count
    paths = images()
    print(f'{len(paths)} fields; screening pairs', flush=True)
    candidates = screen(paths)
    print(f'{len(candidates)} pairs to match (all within-source; cross-source above {SCREEN_CORR})', flush=True)
    blocks = [candidates[k:k + 10000] for k in range(0, len(candidates), 10000)]
    results = []
    with Pool(max(1, cpu_count() - 2)) as pool:
        for n, block in enumerate(pool.imap(confirm_block, [(paths, b) for b in blocks])):
            results += block
            print(f'  matched {min((n + 1) * 10000, len(candidates))}/{len(candidates)}', flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({'method': __doc__, 'screen_correlation': SCREEN_CORR, 'min_inliers': MIN_INLIERS,
                               'min_inlier_ratio': MIN_INLIER_RATIO, 'near_identical_corr': NEAR_IDENTICAL_CORR, 'scale_range': SCALE_RANGE, 'n_fields': len(paths), 'pairs_matched': len(candidates), 'record_min_inliers': RECORD_MIN_INLIERS,
                               'pairs': results}, indent=1), encoding='utf-8')
    print(f'{sum(r["overlap"] for r in results)} confirmed overlapping pairs', flush=True)


if __name__ == '__main__':
    sys.exit(main())
