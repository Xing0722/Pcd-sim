"""
scanner.py

Virtual scanner: simulate a structured-light camera scanning a mesh
from multiple poses using ray casting.

Camera parameters are hardcoded based on publicly available specs of
the Shining3D OptimScan 5M series (a common industrial structured-light
scanner). FOV and resolution determine the ray grid; depth_range filters
out hits that fall outside the sensor's valid working distance.
"""

import numpy as np
import trimesh
import open3d as o3d
from logger import RoboLogger

class Scanner:
    def __init__(self, mesh: trimesh.Trimesh):
        """
        Parameters
        ----------
        mesh : stabilized and centered mesh from PoseGenerator.stabilized_mesh
        """
        self.mesh = mesh

        # --- hardcoded sensor parameters (Shining3D OptimScan 5M series) ---
        self.fov_h         = 62.0          # horizontal FOV in degrees
        self.fov_v         = 48.0          # vertical FOV in degrees
        self.resolution    = (64, 48)   # (width, height) in pixels
        self.depth_range   = (0.01, 1.0)   # valid depth range in meters (min, max)            # cast 1 in every N rays (1 = full res)
                                           # set to 1 on machines with sufficient RAM
                                           # 64x48 full res = ~300K rays per pose

        self.point_clouds  = []
        self.logger = RoboLogger()

    def start_scanning(self, poses: list[np.ndarray]) -> list[o3d.geometry.PointCloud]:
        """
        Full scanning pipeline for all poses:
            compute FOV params → generate ray directions → cast rays per pose

        FOV params and ray directions are computed once and reused across
        all poses — only the origin and rotation change per pose.

        Returns list of point clouds, one per pose.
        """
        fov_params = self._compute_fov_params()
        ray_dirs   = self._generate_ray_directions(fov_params)

        self.point_clouds = []
        for i, pose in enumerate(poses):
            pcd = self._cast_rays(pose, ray_dirs)
            self.point_clouds.append(pcd)
            self.logger.info(f"[scanner] pose {i+1:02d}/{len(poses)}  "
                              f"points={len(pcd.points)}")

        return self.point_clouds

    def _compute_fov_params(self) -> dict:
        """
        Convert FOV angles and resolution into per-pixel angular step sizes.

        Returns
        -------
        dict with:
            fov_h_rad, fov_v_rad : FOV in radians
            step_h, step_v       : angular step per pixel (radians)
            width, height        : resolution
        """
        fov_h_rad = np.deg2rad(self.fov_h)
        fov_v_rad = np.deg2rad(self.fov_v)
        width, height = self.resolution
        return {
            "fov_h_rad" : fov_h_rad,
            "fov_v_rad" : fov_v_rad,
            "step_h"    : fov_h_rad / width,
            "step_v"    : fov_v_rad / height,
            "width"     : width,
            "height"    : height,
        }

    def _generate_ray_directions(self, fov_params: dict) -> np.ndarray:
        """
        Generate a grid of ray directions in camera coordinate space.
        Camera looks along -Z (OpenCV convention): X right, Y up.

        One ray per pixel, uniformly sampled across the FOV:
            horizontal : [-fov_h/2, fov_h/2]
            vertical   : [-fov_v/2, fov_v/2]

        Returns
        -------
        ray_dirs : (width * height, 3) unit vectors in camera space
        """
        angles_h = np.linspace(
            -fov_params["fov_h_rad"] / 2,
             fov_params["fov_h_rad"] / 2,
             fov_params["width"])
        angles_v = np.linspace(
            -fov_params["fov_v_rad"] / 2,
             fov_params["fov_v_rad"] / 2,
             fov_params["height"])

        ah, av = np.meshgrid(angles_h, angles_v)
        ah, av = ah.ravel(), av.ravel()

        # camera looks along -Z
        dx =  np.tan(ah)
        dy =  np.tan(av)
        dz = -np.ones_like(dx)

        dirs = np.stack([dx, dy, dz], axis=1)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)


        self.logger.info(f"[scanner] ray grid: {fov_params['width']}x{fov_params['height']} "
                          f"→ {len(dirs)} rays per pose")
        return dirs

    def _cast_rays(self, pose: np.ndarray,
                   ray_dirs: np.ndarray) -> o3d.geometry.PointCloud:
        """
        Cast rays for one camera pose and return valid hit points.

        Steps:
            1. Rotate ray directions from camera space to world space (R = pose[:3,:3])
            2. Set all ray origins to camera position (pose[:3,3])
            3. trimesh ray casting → hit locations + distances
            4. Filter hits outside depth_range
            5. Return as open3d PointCloud

        Parameters
        ----------
        pose     : (4, 4) camera-to-world matrix from PoseGenerator
        ray_dirs : (N, 3) unit ray directions in camera space
        """
        R          = pose[:3, :3]
        cam_origin = pose[:3, 3]

        # camera space → world space
        dirs_world = (R @ ray_dirs.T).T                          # (N, 3)
        origins    = np.tile(cam_origin, (len(dirs_world), 1))   # (N, 3)

        locations, ray_indices, _ = self.mesh.ray.intersects_location(
            ray_origins=origins,
            ray_directions=dirs_world,
            multiple_hits=False)

        pcd = o3d.geometry.PointCloud()
        if len(locations) == 0:
            return pcd

        # filter by depth range: distance from camera to hit point
        depths = np.linalg.norm(locations - cam_origin, axis=1)
        valid  = (depths >= self.depth_range[0]) & (depths <= self.depth_range[1])
        locations = locations[valid]

        if len(locations) > 0:
            pcd.points = o3d.utility.Vector3dVector(locations)

        return pcd


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from poses import PoseGenerator

    mesh_path = sys.argv[1] if len(sys.argv) > 1 else "data/bunny.obj"
    if not os.path.exists(mesh_path):
        logger.error(f"mesh not found: {mesh_path}")
        sys.exit(1)

    mesh   = trimesh.load(mesh_path, force="mesh")
    thetas = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    phis   = np.array([np.pi / 3])

    gen     = PoseGenerator(mesh, thetas, phis)
    poses   = gen.run()

    scanner = Scanner(gen.stabilized_mesh)
    pcds    = scanner.start_scanning(poses)

    total = sum(len(p.points) for p in pcds)
    print(f"\nTotal points across {len(pcds)} scans: {total}")