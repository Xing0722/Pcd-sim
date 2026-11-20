"""
main.py

Entry point for the virtual scanning pipeline.

Full pipeline:
    load mesh
        ↓
    PoseGenerator  — stable pose, bounding cylinder, camera poses
        ↓
    Scanner        — ray casting per pose → list of point clouds
        ↓
    Registration   — merge frames, FPFH + FGR + ICP → alignment matrix
        ↓
    Visualizer     — show and save intermediate results

Usage:
    python main.py --mesh data/bunny.obj
    python main.py --mesh data/bunny.obj --thetas 8 --phis 2 --no-show
"""

import argparse
import json
import os
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

from poses        import PoseGenerator
from scanner      import Scanner
from registration import Registration
from visualizer   import Visualizer
from logger import RoboLogger

logger = RoboLogger()

def parse_args():
    parser = argparse.ArgumentParser(description="Virtual 3D scanning simulation")
    parser.add_argument("--mesh",    type=str, default="data/bunny.obj",
                        help="Path to input mesh file")
    parser.add_argument("--thetas",  type=int, default=8,
                        help="Number of horizontal scan positions (default: 8)")
    parser.add_argument("--phis",    type=int, default=2,
                        help="Number of zenith angle levels (default: 2)")
    parser.add_argument("--out-dir", type=str, default="results",
                        help="Output directory for saved visualizations")
    parser.add_argument("--no-show", action="store_true",
                        help="Skip interactive open3d windows (save PNG only)")
    parser.add_argument("--perturb", action="store_true",
                        help="Apply a known random rigid transform to the scans "
                             "before registration and report how well it is recovered")
    parser.add_argument("--perturb-deg", type=float, default=30.0,
                        help="Rotation applied by --perturb, in degrees (default: 30)")
    parser.add_argument("--perturb-trans", type=float, default=0.2,
                        help="Translation applied by --perturb, as a fraction of the "
                             "object's bounding-box diagonal (default: 0.2)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for --perturb (default: 0)")
    return parser.parse_args()


def random_rigid_transform(rng, deg: float, trans_frac: float, scale: float) -> np.ndarray:
    """Rotation of `deg` degrees about a random axis plus a random translation."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    T = trimesh.transformations.rotation_matrix(np.deg2rad(deg), axis)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    T[:3, 3] = direction * trans_frac * scale
    return T


def pose_error(T_est: np.ndarray, T_true: np.ndarray):
    """Rotation error (deg) and translation error (mesh units) of T_est vs T_true."""
    D = T_est @ np.linalg.inv(T_true)
    cos = np.clip((np.trace(D[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos))), float(np.linalg.norm(D[:3, 3]))


def main():
    args   = parse_args()
    show   = not args.no_show

    # ------------------------------------------------------------------
    # Load mesh
    # ------------------------------------------------------------------
    logger.info(f"\n[main] loading mesh: {args.mesh}")
    mesh = trimesh.load(args.mesh, force="mesh")
    logger.info(f"[main] vertices={len(mesh.vertices)}  faces={len(mesh.faces)}")

    # ------------------------------------------------------------------
    # Step 1: generate camera poses
    # ------------------------------------------------------------------
    logger.info("\n[main] step 1 — pose generation")
    thetas = np.linspace(0, 2 * np.pi, args.thetas, endpoint=False)
    phis   = np.linspace(np.pi / 6, np.pi / 3, args.phis)

    gen   = PoseGenerator(mesh, thetas, phis)
    poses = gen.run()

    # ------------------------------------------------------------------
    # Step 2: virtual scanning
    # ------------------------------------------------------------------
    logger.info("\n[main] step 2 — virtual scanning")
    scanner = Scanner(gen.stabilized_mesh)
    pcds    = scanner.start_scanning(poses)

    # ------------------------------------------------------------------
    # Optional: knock the scans out of alignment by a known transform.
    # The scans are generated in the model's own frame, so without this the
    # registration only has to return the identity.
    # ------------------------------------------------------------------
    diag   = float(np.linalg.norm(gen.stabilized_mesh.extents))
    T_pert = np.eye(4)
    if args.perturb:
        rng    = np.random.default_rng(args.seed)
        T_pert = random_rigid_transform(rng, args.perturb_deg,
                                        args.perturb_trans, diag)
        pcds   = [o3d.geometry.PointCloud(p).transform(T_pert) for p in pcds]
        logger.info(f"\n[main] perturbing scans: {args.perturb_deg:.0f} deg, "
                    f"{args.perturb_trans * diag:.4f} units")

    # ------------------------------------------------------------------
    # Step 3: registration
    # ------------------------------------------------------------------
    logger.info("\n[main] step 3 — registration")
    reg = Registration(pcds, gen.stabilized_mesh)
    T   = reg.run()
    logger.info(f"\n[main] alignment matrix:\n{T}")

    # ------------------------------------------------------------------
    # Evaluation against ground truth
    # ------------------------------------------------------------------
    rot_err, trans_err = pose_error(T, np.linalg.inv(T_pert))
    dists, _ = cKDTree(np.asarray(gen.stabilized_mesh.vertices)).query(
        np.asarray(reg.aligned_pcd.points))
    metrics = {
        "mesh":                 args.mesh,
        "num_poses":            len(poses),
        "num_points_merged":    int(len(reg.aligned_pcd.points)),
        "perturbed":            bool(args.perturb),
        "perturb_deg":          args.perturb_deg if args.perturb else 0.0,
        "rotation_error_deg":   rot_err,
        "translation_error":    trans_err,
        "translation_error_pct_of_diag": 100.0 * trans_err / diag,
        "mean_dist_to_mesh":    float(dists.mean()),
        "rms_dist_to_mesh":     float(np.sqrt((dists ** 2).mean())),
        "bbox_diagonal":        diag,
    }
    logger.info(f"[main] rotation error {rot_err:.4f} deg, "
                f"translation error {trans_err:.5f} units "
                f"({metrics['translation_error_pct_of_diag']:.3f}% of bbox diagonal), "
                f"RMS distance to mesh {metrics['rms_dist_to_mesh']:.5f}")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ------------------------------------------------------------------
    # Step 4: visualize
    # ------------------------------------------------------------------
    logger.info("\n[main] step 4 — visualization")
    vis = Visualizer(out_dir=args.out_dir)

    vis.cameras(
        gen.stabilized_mesh, poses,
        show=show, save=True)

    vis.single_scan(
        gen.stabilized_mesh, pcds[0], poses[0],
        show=show, save=True)

    vis.merged_scans(
        gen.stabilized_mesh, pcds,
        show=show, save=True)

    vis.aligned(
        gen.stabilized_mesh, reg.aligned_pcd,
        show=show, save=True)

    logger.info(f"\n[main] done. results saved to '{args.out_dir}/'")


if __name__ == "__main__":
    main()