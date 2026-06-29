"""
registration.py

Register the merged scan point cloud against the original mesh.
点云配准：将合并的扫描点云与原始网格进行配准。
Pipeline:
    merge all scan frames → preprocess (downsample + normals + FPFH) →
    FGR coarse registration → ICP fine registration →
    output 4x4 alignment matrix + aligned point cloud
"""

import numpy as np
import trimesh
import open3d as o3d
from logger import RoboLogger

class Registration:
    def __init__(self, point_clouds: list[o3d.geometry.PointCloud],
                 mesh: trimesh.Trimesh,
                 voxel_size: float = 0.005):
        """
        Parameters
        ----------
        point_clouds : list of point clouds from Scanner.run()
        mesh         : stabilized mesh from PoseGenerator (used as target)
        voxel_size   : voxel size for downsampling, relative to mesh scale
        """
        self.point_clouds = point_clouds
        self.mesh         = mesh
        self.voxel_size   = voxel_size

        self.aligned_pcd  = None
        self.transform    = None
        self.logger       = RoboLogger()
    def run(self) -> np.ndarray:
        """
        Full pipeline:
            merge → preprocess → FGR → ICP
        Returns 4x4 alignment matrix (source → target).
        """
        merged            = self._merge_point_clouds()
        target            = self._mesh_to_pcd()
        src_down, src_fpfh = self._preprocess(merged)
        tgt_down, tgt_fpfh = self._preprocess(target)
        T_coarse          = self._fgr(src_down, tgt_down, src_fpfh, tgt_fpfh)
        T_fine            = self._icp(merged, target, T_coarse)

        self.transform    = T_fine
        self.aligned_pcd  = merged.transform(T_fine)
        return self.transform

    def _merge_point_clouds(self) -> o3d.geometry.PointCloud:
        """
        Concatenate all scan frames into a single point cloud.
        Each frame is already in world coordinates (ray casting outputs
        world-space hit points directly), so no per-frame transform needed.
        """
        merged = o3d.geometry.PointCloud()
        for pcd in self.point_clouds:
            merged += pcd
        self.logger.info(f"[registration] merged {len(self.point_clouds)} frames → "
                          f"{len(merged.points)} points")
        return merged

    def _mesh_to_pcd(self) -> o3d.geometry.PointCloud:
        """
        Convert mesh vertices to an open3d point cloud (no sampling needed).
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(self.mesh.vertices))
        return pcd

    def _preprocess(self, pcd: o3d.geometry.PointCloud):
        """
        Voxel downsample → estimate normals → compute FPFH features.
        """
        pcd_down = pcd.voxel_down_sample(self.voxel_size)

        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size * 2, max_nn=30))

        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size * 5, max_nn=100))

        return pcd_down, fpfh

    def _fgr(self, src_down, tgt_down, src_fpfh, tgt_fpfh) -> np.ndarray:
        """
        Fast Global Registration: coarse alignment using FPFH correspondences
        and Geman-McClure robust optimization. No initial guess required.
        """
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            src_down, tgt_down, src_fpfh, tgt_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=self.voxel_size * 0.5))

        self.logger.info(f"[registration] FGR fitness={result.fitness:.4f}  "
                          f"inlier_rmse={result.inlier_rmse:.5f}")
        return result.transformation

    def _icp(self, src: o3d.geometry.PointCloud,
             tgt: o3d.geometry.PointCloud,
             T_init: np.ndarray) -> np.ndarray:
        """
        Point-to-plane ICP fine registration, initialized with FGR result.
        """
        if not tgt.has_normals():
            tgt.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.voxel_size * 2, max_nn=30))

        result = o3d.pipelines.registration.registration_icp(
            src, tgt,
            max_correspondence_distance=self.voxel_size * 0.4,
            init=T_init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane())

        self.logger.info(f"[registration] ICP  fitness={result.fitness:.4f}  "
                          f"inlier_rmse={result.inlier_rmse:.5f}")
        return result.transformation


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from poses import PoseGenerator
    from scanner import Scanner, ScannerConfig

    mesh_path = sys.argv[1] if len(sys.argv) > 1 else "data/bunny.obj"
    if not os.path.exists(mesh_path):
        self.logger.error(f"mesh not found: {mesh_path}")
        sys.exit(1)

    mesh   = trimesh.load(mesh_path, force="mesh")
    thetas = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    phis   = np.array([np.pi / 3])

    gen     = PoseGenerator(mesh, thetas, phis)
    poses   = gen.run()

    scanner = Scanner(gen.stabilized_mesh)
    pcds    = scanner.run(poses)

    reg     = Registration(pcds, gen.stabilized_mesh)
    T       = reg.run()
    self.logger.info(f"\nFinal alignment matrix:\n{T}")