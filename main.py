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
import numpy as np
import trimesh

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
    return parser.parse_args()


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
    # Step 3: registration
    # ------------------------------------------------------------------
    logger.info("\n[main] step 3 — registration")
    reg = Registration(pcds, gen.stabilized_mesh)
    T   = reg.run()
    logger.info(f"\n[main] alignment matrix:\n{T}")

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