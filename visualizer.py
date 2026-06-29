"""
visualizer.py

Visualize intermediate results of the virtual scanning pipeline.

Two output modes:
    show() → open3d interactive window (local debugging)
    save() → matplotlib PNG (for README)

Visualization nodes:
    1. stabilized mesh + camera positions
    2. single frame scan result
    3. all frames merged (before registration)
    4. merged scan aligned to original mesh (after registration)
"""

import numpy as np
import trimesh
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from logger import RoboLogger
# ---------------------------------------------------------------------------
# Color palette (one color per node for consistency)
# ---------------------------------------------------------------------------

COLORS = {
    "mesh":         [0.8, 0.8, 0.8],   # light grey
    "camera":       [1.0, 0.2, 0.2],   # red
    "scan_single":  [0.2, 0.6, 1.0],   # blue
    "scan_merged":  [0.3, 0.8, 0.3],   # green
    "scan_aligned": [0.2, 0.6, 1.0],   # blue
    "target_mesh":  [0.8, 0.4, 0.1],   # orange
}


class Visualizer:
    def __init__(self, out_dir: str = "results"):
        self.out_dir = out_dir
        import os
        os.makedirs(out_dir, exist_ok=True)
        self.logger = RoboLogger()
    # -----------------------------------------------------------------------
    # Node 1: stabilized mesh + camera positions
    # -----------------------------------------------------------------------

    def cameras(self, mesh: trimesh.Trimesh, poses: list[np.ndarray],
                show: bool = True, save: bool = True):
        """Stabilized mesh with all camera positions overlaid."""
        mesh_pcd = self._mesh_to_pcd(mesh, COLORS["mesh"])
        cam_pcd  = self._poses_to_pcd(poses, COLORS["camera"])

        if show:
            self._show_o3d([mesh_pcd, cam_pcd],
                           title="Node 1: Stabilized Mesh + Camera Positions")
        if save:
            self._save_mpl(
                point_sets=[
                    (np.asarray(mesh_pcd.points), COLORS["mesh"],    "mesh", 0.5),
                    (np.asarray(cam_pcd.points),  COLORS["camera"],  "cameras", 30),
                ],
                title="Node 1: Stabilized Mesh + Camera Positions",
                filename="node1_cameras.png")

    # -----------------------------------------------------------------------
    # Node 2: single frame scan
    # -----------------------------------------------------------------------

    def single_scan(self, mesh: trimesh.Trimesh,
                    pcd: o3d.geometry.PointCloud, pose: np.ndarray,
                    show: bool = True, save: bool = True):
        """One camera pose and its resulting scan point cloud."""
        mesh_pcd  = self._mesh_to_pcd(mesh, COLORS["mesh"])
        scan_pcd  = self._colorize(pcd, COLORS["scan_single"])
        cam_pcd   = self._poses_to_pcd([pose], COLORS["camera"])

        if show:
            self._show_o3d([mesh_pcd, scan_pcd, cam_pcd],
                           title="Node 2: Single Frame Scan")
        if save:
            self._save_mpl(
                point_sets=[
                    (np.asarray(mesh_pcd.points),  COLORS["mesh"],        "mesh", 0.5),
                    (np.asarray(scan_pcd.points),  COLORS["scan_single"], "scan",    5),
                    (np.asarray(cam_pcd.points),   COLORS["camera"],      "camera", 50),
                ],
                title="Node 2: Single Frame Scan",
                filename="node2_single_scan.png")

    # -----------------------------------------------------------------------
    # Node 3: all frames merged (before registration)
    # -----------------------------------------------------------------------

    def merged_scans(self, mesh: trimesh.Trimesh,
                     point_clouds: list[o3d.geometry.PointCloud],
                     show: bool = True, save: bool = True):
        """All scan frames overlaid before registration."""
        mesh_pcd   = self._mesh_to_pcd(mesh, COLORS["mesh"])
        merged     = o3d.geometry.PointCloud()
        for pcd in point_clouds:
            merged += pcd
        scan_pcd = self._colorize(merged, COLORS["scan_merged"])

        if show:
            self._show_o3d([mesh_pcd, scan_pcd],
                           title="Node 3: All Frames Merged (before registration)")
        if save:
            self._save_mpl(
                point_sets=[
                    (np.asarray(mesh_pcd.points), COLORS["mesh"],        "mesh", 0.5),
                    (np.asarray(scan_pcd.points), COLORS["scan_merged"], "scan",    3),
                ],
                title="Node 3: All Frames Merged (before registration)",
                filename="node3_merged.png")

    # -----------------------------------------------------------------------
    # Node 4: aligned scan vs original mesh (after registration)
    # -----------------------------------------------------------------------

    def aligned(self, mesh: trimesh.Trimesh,
                aligned_pcd: o3d.geometry.PointCloud,
                show: bool = True, save: bool = True):
        """Registered scan overlaid on original mesh."""
        mesh_pcd = self._mesh_to_pcd(mesh, COLORS["target_mesh"])
        scan_pcd = self._colorize(aligned_pcd, COLORS["scan_aligned"])

        if show:
            self._show_o3d([mesh_pcd, scan_pcd],
                           title="Node 4: Aligned Scan vs Original Mesh")
        if save:
            self._save_mpl(
                point_sets=[
                    (np.asarray(mesh_pcd.points), COLORS["target_mesh"],  "mesh", 0.5),
                    (np.asarray(scan_pcd.points), COLORS["scan_aligned"], "aligned scan", 3),
                ],
                title="Node 4: Aligned Scan vs Original Mesh",
                filename="node4_aligned.png")

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _mesh_to_pcd(self, mesh: trimesh.Trimesh,
                     color: list) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        pcd.paint_uniform_color(color)
        return pcd

    def _colorize(self, pcd: o3d.geometry.PointCloud,
                  color: list) -> o3d.geometry.PointCloud:
        out = o3d.geometry.PointCloud(pcd)
        out.paint_uniform_color(color)
        return out

    def _poses_to_pcd(self, poses: list[np.ndarray],
                      color: list) -> o3d.geometry.PointCloud:
        """Extract camera positions from 4x4 poses → point cloud."""
        positions = np.array([p[:3, 3] for p in poses])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions)
        pcd.paint_uniform_color(color)
        return pcd

    def _show_o3d(self, geometries: list, title: str = ""):
        """Open3D interactive window (requires display)."""
        try:
            o3d.visualization.draw_geometries(geometries, window_name=title)
        except Exception as e:
            self.logger.error(f"[visualizer] open3d display unavailable: {e}")

    def _save_mpl(self, point_sets: list, title: str, filename: str):
        """
        Save a 3D scatter plot as PNG.

        point_sets: list of (points_array, rgb_color, label, marker_size)
        """
        fig = plt.figure(figsize=(8, 6))
        ax  = fig.add_subplot(111, projection="3d")

        for pts, color, label, size in point_sets:
            if len(pts) == 0:
                continue
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c=[color], s=size, label=label, alpha=0.6)

        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

        out_path = f"{self.out_dir}/{filename}"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        self.logger.info(f"[visualizer] saved → {out_path}")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from poses import PoseGenerator
    from scanner import Scanner
    from registration import Registration

    mesh_path = sys.argv[1] if len(sys.argv) > 1 else "data/bunny.obj"
    if not os.path.exists(mesh_path):
        self.logger.error(f"mesh not found: {mesh_path}")
        sys.exit(1)

    mesh   = trimesh.load(mesh_path, force="mesh")
    thetas = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    phis   = np.array([np.pi / 3])

    gen   = PoseGenerator(mesh, thetas, phis)
    poses = gen.run()

    scanner = Scanner(gen.stabilized_mesh)
    pcds    = scanner.run(poses)

    reg = Registration(pcds, gen.stabilized_mesh)
    reg.run()

    vis = Visualizer(out_dir="results")
    vis.cameras(gen.stabilized_mesh, poses,     show=False, save=True)
    vis.single_scan(gen.stabilized_mesh, pcds[0], poses[0], show=False, save=True)
    vis.merged_scans(gen.stabilized_mesh, pcds,              show=False, save=True)
    vis.aligned(gen.stabilized_mesh, reg.aligned_pcd,        show=False, save=True)