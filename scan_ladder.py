"""
scan_ladder.py — Stage-2 core: detect dental Scan Ladder cylinders in a noisy
photogrammetry reconstruction and register reference geometry to each detected
marker, with numerical confidence and reject-bad-fit logic.

This is a self-contained, verifiable demonstration of the geometric-registration
approach proposed for the "Scan Ladder Recognition" stage:

    candidate/primitive detection  ->  robust axis/radius estimate
    ->  constrained ICP refinement  ->  confidence (fitness + inlier RMSE)
    ->  accept / reject vs threshold

It runs on synthetic data with KNOWN ground-truth poses so the reported accuracy
can be checked against truth (no black box). On real data the same pipeline
consumes the OBJ/PLY reconstruction and the supplied library_main.stl reference.

Author: Dr. Sandeep Grover
"""

from __future__ import annotations

import numpy as np

try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:  # pragma: no cover - demo still runs in fallback mode
    _HAS_O3D = False


# --------------------------------------------------------------------------- #
# Reference geometry (a single Scan Ladder cylinder). On real jobs this comes
# from library_main.stl; here we parametrise it so the demo is dependency-free.
# --------------------------------------------------------------------------- #
REF_RADIUS_MM = 1.25      # cylinder radius (mm)
REF_HEIGHT_MM = 6.0       # cylinder height (mm)


def _rotation_from_axis(axis: np.ndarray) -> np.ndarray:
    """Rotation mapping +Z onto the given unit axis (Rodrigues)."""
    z = np.array([0.0, 0.0, 1.0])
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    v = np.cross(z, axis)
    s = np.linalg.norm(v)
    c = float(np.dot(z, axis))
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def sample_cylinder(center, axis, radius=REF_RADIUS_MM, height=REF_HEIGHT_MM,
                    n=1200, rng=None):
    """Sample surface points of a cylinder at a given pose."""
    rng = rng or np.random.default_rng(0)
    R = _rotation_from_axis(np.asarray(axis, float))
    theta = rng.uniform(0, 2 * np.pi, n)
    h = rng.uniform(-height / 2, height / 2, n)
    local = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), h])
    return (local @ R.T) + np.asarray(center, float)


# --------------------------------------------------------------------------- #
# Robust primitive (cylinder) detection
# --------------------------------------------------------------------------- #
def estimate_cylinder_axis(points: np.ndarray):
    """
    Estimate the axis, center and radius of a set of points sampled from a
    cylinder. The axis is the dominant eigenvector of the centred covariance
    (points spread most along the cylinder axis); radius is the robust median
    distance to that axis. Returns (center, axis, radius).
    """
    c = points.mean(axis=0)
    X = points - c
    # PCA: cylinder axis = eigenvector of largest eigenvalue
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    axis = vt[0]
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    # radial distance of each point to the axis line through c
    proj = X - np.outer(X @ axis, axis)
    radius = float(np.median(np.linalg.norm(proj, axis=1)))
    return c, axis, radius


def detect_markers(scene: np.ndarray, eps=2.0, min_points=80,
                   radius_lo=0.6, radius_hi=2.5):
    """
    Cluster the scene point cloud and keep clusters whose fitted radius is
    consistent with a Scan Ladder cylinder. Returns a list of candidate dicts.

    eps / min_points: DBSCAN parameters (mm). radius_lo/hi: acceptance band
    around the reference radius to reject face/background clutter.
    """
    labels = _cluster(scene, eps, min_points)
    candidates = []
    for lab in sorted(set(labels)):
        if lab < 0:
            continue
        pts = scene[labels == lab]
        if len(pts) < min_points:
            continue
        center, axis, radius = estimate_cylinder_axis(pts)
        # cylindricity check: residual of radial distances should be small
        proj = (pts - center) - np.outer((pts - center) @ axis, axis)
        radial = np.linalg.norm(proj, axis=1)
        cylindricity = float(np.std(radial))  # mm; low = clean cylinder
        if radius_lo <= radius <= radius_hi and cylindricity < 0.6:
            candidates.append(dict(points=pts, center=center, axis=axis,
                                   radius=radius, cylindricity=cylindricity))
    return candidates


def _cluster(scene, eps, min_points):
    """DBSCAN via open3d if present, else a light numpy grid-region fallback."""
    if _HAS_O3D:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(scene)
        return np.asarray(pc.cluster_dbscan(eps=eps, min_points=min_points))
    # Fallback: union-find on a KD-tree radius graph (kept simple for the demo)
    from scipy.spatial import cKDTree
    tree = cKDTree(scene)
    parent = np.arange(len(scene))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, nbrs in enumerate(tree.query_ball_tree(tree, eps)):
        for j in nbrs:
            parent[find(i)] = find(j)
    roots = np.array([find(i) for i in range(len(scene))])
    labels = -np.ones(len(scene), dtype=int)
    for lab, r in enumerate(np.unique(roots)):
        idx = np.where(roots == r)[0]
        if len(idx) >= min_points:
            labels[idx] = lab
    return labels


# --------------------------------------------------------------------------- #
# Constrained registration + confidence
# --------------------------------------------------------------------------- #
def register_reference(candidate, ref_radius=REF_RADIUS_MM,
                       ref_height=REF_HEIGHT_MM, rmse_reject=0.5,
                       fitness_reject=0.8):
    """
    Register the reference cylinder onto a detected candidate and return a pose
    with numerical confidence. The primitive estimate seeds the transform; ICP
    (point-to-plane) refines it. A match is REJECTED when inlier RMSE is too
    high or ICP fitness (inlier fraction) is too low — i.e. we never return a
    silent visual best-fit.
    """
    center = candidate["center"]
    axis = candidate["axis"]

    # Seed pose from the primitive estimate.
    R = _rotation_from_axis(axis)
    T_init = np.eye(4)
    T_init[:3, :3] = R
    T_init[:3, 3] = center

    ref_pts = sample_cylinder([0, 0, 0], [0, 0, 1], ref_radius, ref_height,
                              n=len(candidate["points"]),
                              rng=np.random.default_rng(7))

    if _HAS_O3D:
        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(ref_pts)
        tgt = o3d.geometry.PointCloud()
        tgt.points = o3d.utility.Vector3dVector(candidate["points"])
        tgt.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
        reg = o3d.pipelines.registration.registration_icp(
            src, tgt, max_correspondence_distance=1.0, init=T_init,
            estimation_method=o3d.pipelines.registration.
            TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=60))
        T = reg.transformation
        rmse = float(reg.inlier_rmse)
        fitness = float(reg.fitness)
    else:  # numpy point-to-point ICP fallback
        T, rmse, fitness = _icp_numpy(ref_pts, candidate["points"], T_init)

    accepted = (rmse <= rmse_reject) and (fitness >= fitness_reject)
    confidence = float(fitness * np.exp(-rmse / rmse_reject))
    return dict(transform=T, position=T[:3, 3], axis=T[:3, :3] @ np.array([0, 0, 1.0]),
                rmse_mm=rmse, fitness=fitness, confidence=confidence,
                accepted=accepted, radius_mm=candidate["radius"])


def _icp_numpy(src, tgt, T_init, iters=40, max_dist=1.0):
    from scipy.spatial import cKDTree
    T = T_init.copy()
    tree = cKDTree(tgt)
    rmse = fitness = 0.0
    for _ in range(iters):
        s = (src @ T[:3, :3].T) + T[:3, 3]
        dist, idx = tree.query(s)
        m = dist < max_dist
        if m.sum() < 10:
            break
        P, Q = s[m], tgt[idx[m]]
        pc, qc = P.mean(0), Q.mean(0)
        H = (P - pc).T @ (Q - qc)
        U, _, Vt = np.linalg.svd(H)
        Rk = Vt.T @ U.T
        if np.linalg.det(Rk) < 0:
            Vt[-1] *= -1
            Rk = Vt.T @ U.T
        tk = qc - Rk @ pc
        step = np.eye(4)
        step[:3, :3] = Rk
        step[:3, 3] = tk
        T = step @ T
        rmse = float(np.sqrt((dist[m] ** 2).mean()))
        fitness = float(m.mean())
    return T, rmse, fitness
