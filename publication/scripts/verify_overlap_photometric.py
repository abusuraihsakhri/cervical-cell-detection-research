"""Photometric check of confirmed overlaps: warp field B onto field A with the
fitted similarity transform and measure normalized cross-correlation inside the
shared region. Keypoint agreement by chance leaves the warped pixels uncorrelated.

Output: results/research_v3/overlap_photometric.json
"""
import json
from multiprocessing import Pool
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_field_overlap import ROOT, features

OUT = ROOT / 'results/research_v3/overlap_photometric.json'


def gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(image, (1024, int(1024 * image.shape[0] / image.shape[1])), interpolation=cv2.INTER_AREA)


def check(pair):
    cv2.setNumThreads(1)
    orb, matcher = cv2.ORB_create(nfeatures=3000), cv2.BFMatcher(cv2.NORM_HAMMING)
    (ka, da), (kb, db) = features(ROOT / pair['a'], orb), features(ROOT / pair['b'], orb)
    good = [m for m, n in (p for p in matcher.knnMatch(db, da, k=2) if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return {**pair, 'overlap_fraction': 0.0, 'region_ncc': None}
    transform, _ = cv2.estimateAffinePartial2D(kb[[m.queryIdx for m in good]], ka[[m.trainIdx for m in good]],
                                               method=cv2.RANSAC, ransacReprojThreshold=6.0)
    a, b = gray(ROOT / pair['a']).astype(np.float32), gray(ROOT / pair['b']).astype(np.float32)
    size = (a.shape[1], a.shape[0])
    warped = cv2.warpAffine(b, transform, size)
    mask = cv2.warpAffine(np.ones_like(b), transform, size, flags=cv2.INTER_NEAREST) > 0
    mask = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    fraction = float(mask.mean())
    if mask.sum() < 5000:
        return {**pair, 'overlap_fraction': fraction, 'region_ncc': None}
    x, y = a[mask] - a[mask].mean(), warped[mask] - warped[mask].mean()
    ncc = float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-9))
    return {**pair, 'overlap_fraction': round(fraction, 4), 'region_ncc': round(ncc, 4)}


def main():
    pairs = [p for p in json.loads((ROOT / 'results/research_v3/field_overlap.json').read_text())['pairs'] if p['overlap']]
    with Pool(14) as pool:
        results = pool.map(check, pairs, chunksize=8)
    OUT.write_text(json.dumps({'method': __doc__, 'pairs': results}, indent=1))
    values = np.array([r['region_ncc'] if r['region_ncc'] is not None else -1 for r in results])
    print(np.histogram(values, bins=[-1.01, 0, .2, .4, .5, .6, .7, .8, .9, 1.01]))


if __name__ == '__main__':
    main()
