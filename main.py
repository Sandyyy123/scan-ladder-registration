"""
main.py — end-to-end Stage-2 demo.

Builds a synthetic "Retracted" reconstruction: 2-8 Scan Ladder cylinders at
known poses, embedded in noisy face/background clutter with partial occlusion,
then runs the full detect -> register -> confidence -> accept/reject pipeline
and prints per-marker accuracy against ground truth.

Run:  python main.py            (default 5 markers, seed 42)
      python main.py 3 7        (n_markers seed)
"""

from __future__ import annotations

import sys
import numpy as np

from scan_ladder import (sample_cylinder, detect_markers, register_reference,
                         REF_RADIUS_MM, REF_HEIGHT_MM)


def build_scene(n_markers=5, seed=42):
    rng = np.random.default_rng(seed)
    truth = []
    clouds = []
    # place markers along a shallow arc, each tilted differently
    for i in range(n_markers):
        center = np.array([(-i + n_markers / 2) * 9.0,
                           rng.uniform(-3, 3),
                           rng.uniform(-2, 2)])
        axis = np.array([rng.uniform(-0.25, 0.25),
                         rng.uniform(-0.25, 0.25), 1.0])
        axis /= np.linalg.norm(axis)
        pts = sample_cylinder(center, axis, n=1000, rng=rng)
        # partial occlusion: drop 30-60% of one side (noisy/incomplete recon)
        keep = pts @ axis * 0 + (pts[:, 0] > center[0] - 4)  # simple half-cut
        frac = rng.uniform(0.4, 0.7)
        mask = rng.random(len(pts)) < frac
        pts = pts[mask | keep.astype(bool)]
        # measurement noise ~0.15 mm
        pts = pts + rng.normal(0, 0.15, pts.shape)
        clouds.append(pts)
        truth.append(dict(center=center, axis=axis))
    # background clutter = the "face" reconstruction (irregular blob)
    clutter = rng.normal(0, 18, (4000, 3))
    clutter[:, 2] -= 25  # push behind the markers
    clouds.append(clutter)
    scene = np.vstack(clouds)
    return scene, truth


def main():
    n_markers = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    scene, truth = build_scene(n_markers, seed)
    print(f"Scene: {len(scene):,} points | {n_markers} true markers "
          f"+ background clutter\n"
          f"Reference cylinder: r={REF_RADIUS_MM} mm, h={REF_HEIGHT_MM} mm\n"
          + "-" * 68)

    candidates = detect_markers(scene)
    print(f"Primitive detection: {len(candidates)} cylinder candidate(s) "
          f"passed the radius + cylindricity gate\n" + "-" * 68)

    results = []
    for cand in candidates:
        res = register_reference(cand)
        results.append(res)

    # match each accepted result to nearest ground-truth marker for accuracy
    print(f"{'#':>2}  {'pos error':>9}  {'axis err':>8}  {'RMSE':>6}  "
          f"{'fitness':>7}  {'conf':>5}  verdict")
    accepted = 0
    for k, res in enumerate(results):
        d = [np.linalg.norm(res["position"] - t["center"]) for t in truth]
        j = int(np.argmin(d))
        pos_err = d[j]
        ax_err = np.degrees(np.arccos(
            np.clip(abs(res["axis"] @ truth[j]["axis"]), 0, 1)))
        verdict = "ACCEPT" if res["accepted"] else "reject"
        accepted += int(res["accepted"])
        print(f"{k:>2}  {pos_err:>7.3f}mm  {ax_err:>6.2f}°  "
              f"{res['rmse_mm']:>5.3f}  {res['fitness']:>7.2f}  "
              f"{res['confidence']:>5.2f}  {verdict}")

    acc_pos = np.mean([min(np.linalg.norm(r['position'] - t['center'])
                           for t in truth)
                       for r in results if r['accepted']]) if accepted else float('nan')
    print("-" * 68)
    print(f"Accepted {accepted}/{len(results)} candidate(s) "
          f"({len(truth)} true markers present).")
    if accepted:
        print(f"Mean position error on accepted markers: {acc_pos:.3f} mm "
              f"(target v1: ~1 mm)")
    print("\nEach accepted marker yields: XYZ position, 3D axis/orientation,\n"
          "inlier RMSE, ICP fitness and a 0-1 confidence. Low-confidence or\n"
          "high-RMSE candidates are rejected, never returned as a visual guess.")


if __name__ == "__main__":
    main()
