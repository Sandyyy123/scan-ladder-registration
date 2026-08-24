# Scan Ladder Recognition & Registration — Stage-2 demo

A runnable demonstration of the geometric-registration approach for detecting
dental **Scan Ladder** cylinders in a noisy photogrammetry reconstruction and
placing the exact reference STL at each detected pose — with numerical
confidence and reject-bad-fit logic, not a silent visual best-fit.

Built as a working reference for an iPhone dental 3D workflow (Stage 2 of a
photogrammetry → marker-registration → smile-simulation pipeline).

## What it does

```
noisy reconstruction (OBJ/PLY point cloud, 2-8 partial cylinders + face clutter)
        │
        ▼
1. Primitive detection      DBSCAN clustering → PCA axis + robust radius,
                            gated by a radius band + cylindricity test
                            (rejects face/background clutter)
        │
        ▼
2. Constrained registration primitive estimate seeds a rigid transform;
                            point-to-plane ICP refines it against the cluster
        │
        ▼
3. Confidence + gating      inlier RMSE, ICP fitness, 0-1 confidence;
                            ACCEPT only if RMSE ≤ 0.5 mm and fitness ≥ 0.8,
                            otherwise REJECT (never a visual guess)
        │
        ▼
per-marker output: XYZ position, 3D axis/orientation, RMSE, fitness, confidence
```

## Run

```bash
pip install -r requirements.txt
python main.py            # 5 markers, seed 42
python main.py 8 99       # 8 markers, seed 99
```

The demo builds synthetic markers at **known** ground-truth poses (with partial
occlusion + measurement noise + a background "face" blob), so the reported
accuracy is checkable against truth rather than asserted. Typical result:
**mean position error ~0.1–0.25 mm** across 2–8 markers, comfortably inside the
v1 ~1 mm target.

Example (`python main.py`):

```
 #  pos error  axis err    RMSE  fitness   conf  verdict
 0    0.058mm    0.66°     0.162     1.00   0.72  ACCEPT
 1    0.585mm    0.99°     0.184     1.00   0.69  ACCEPT
 ...
Mean position error on accepted markers: 0.220 mm (target v1: ~1 mm)
```

## From demo to production

On the real job the pipeline swaps the synthetic scene for:

- **Input:** the Retracted photogrammetry reconstruction (colour OBJ/PLY, true
  XYZ scale) produced in Stage 1.
- **Reference:** `library_main.stl` (the exact Scan Ladder geometry) instead of
  the parametric cylinder used here.
- **Scale-up path toward sub-0.1 mm:** multi-start ICP, cylinder-axis
  constraints from the known ladder layout, and generalised-ICP / point-to-plane
  with normal weighting — investigated once v1 (~1 mm) is validated on real
  scans.

## Files

| File | Role |
|------|------|
| `scan_ladder.py` | detection + registration + confidence core |
| `main.py` | synthetic scene builder + end-to-end run + accuracy report |
| `requirements.txt` | numpy, scipy, open3d (open3d optional) |

Illustrative demo — synthetic data. Not a medical device.

Dr. Sandeep Grover
