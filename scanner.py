import numpy as np
import trimesh
import open3d as o3d
from logger import RoboLogger

class Scanner:
    def __init__(self, mesh: trimesh.Trimesh):
        """
        Parameters
        """
        self.mesh = mesh
        self.logger = RoboLogger()
        #camera parameters, these can be adjusted to simulate different scanners
        self.fov_h      = 62.0        # horizontal FOV in degrees
        self.fov_v      = 48.0        # vertical FOV in degrees
        self.depth      = (0.01, 1.0) # valid depth distance range in meters
                                      
        self.resolution = (64, 48)    # downsampled from 4096x3000

        self.point_clouds = []

    def start_scanning(self, poses: list[np.ndarray]) -> list[o3d.geometry.PointCloud]:
        """
        entry point for scanning the mesh from a list of camera poses.
        """
        ray_endpoints = self._generate_ray_lines()

        self.point_clouds = []
        for i, pose in enumerate(poses):
            hits            = self._cast_rays(pose, ray_endpoints)
            hits_filtered   = self._filter_points(hits, pose)

            pcd = o3d.geometry.PointCloud()
            if len(hits_filtered) > 0:
                pcd.points = o3d.utility.Vector3dVector(hits_filtered)
            self.point_clouds.append(pcd)

            self.logger.info(f"[scanner] pose {i+1:02d}/{len(poses)}  "
                              f"hits={len(hits)}  after_filter={len(hits_filtered)}")

        return self.point_clouds

    def _generate_ray_lines(self) -> np.ndarray:
        """
        Generate ray endpoint positions in camera coordinate space.

        Camera convention: looks along -Z, X right, Y up.
        Each ray goes from the origin (camera position) toward a point
        on the far plane at depth_max, spread across the FOV grid.
        """
        fov_h_rad = np.deg2rad(self.fov_h)
        fov_v_rad = np.deg2rad(self.fov_v)
        w, h      = self.resolution
        depth_max = self.depth[1]

        angles_h = np.linspace(-fov_h_rad / 2, fov_h_rad / 2, w)
        angles_v = np.linspace(-fov_v_rad / 2, fov_v_rad / 2, h)

        ah, av = np.meshgrid(angles_h, angles_v)
        ah, av = ah.ravel(), av.ravel()

        # direction in camera space (camera looks along -Z)
        dx =  np.tan(ah)
        dy =  np.tan(av)
        dz = -np.ones_like(dx)

        dirs = np.stack([dx, dy, dz], axis=1)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

        # endpoint = origin + direction * depth_max
        endpoints = dirs * depth_max

        self.logger.info(f"[scanner] generated {len(endpoints)} ray lines "
                          f"({w}x{h}, fov={self.fov_h}x{self.fov_v}deg, "
                          f"depth_max={depth_max})")
        return endpoints

    def _cast_rays(self, pose: np.ndarray,
                   ray_endpoints: np.ndarray) -> np.ndarray:
        """
        Transform ray endpoints from camera space to world space,
        then cast rays against the mesh.
        """
        R          = pose[:3, :3]
        cam_origin = pose[:3, 3]

        # transform endpoints to world space
        endpoints_world = (R @ ray_endpoints.T).T + cam_origin  # (N, 3)

        # ray direction = endpoint - origin (already unit length from _generate_ray_lines)
        dirs_world = endpoints_world - cam_origin
        dirs_world /= np.linalg.norm(dirs_world, axis=1, keepdims=True)

        origins = np.tile(cam_origin, (len(dirs_world), 1))

        locations, _, _ = self.mesh.ray.intersects_location(
            ray_origins=origins,
            ray_directions=dirs_world,
            multiple_hits=False)

        return locations

    def _filter_points(self, points: np.ndarray,
                       pose: np.ndarray) -> np.ndarray:
        """
        Double-check hit points against depth range and FOV angle.
        Ray casting should already respect these bounds, but this
        filters any edge cases.
        """
        if len(points) == 0:
            return points

        cam_origin = pose[:3, 3]
        cam_axis   = -pose[:3, 2]   # camera looks along -Z in camera space → world forward

        vecs   = points - cam_origin
        depths = np.linalg.norm(vecs, axis=1)

        # depth filter
        depth_mask = (depths >= self.depth[0]) & (depths <= self.depth[1])

        # FOV angle filter
        vecs_norm  = vecs / depths[:, np.newaxis]
        cos_angles = np.clip(vecs_norm @ cam_axis, -1.0, 1.0)
        angles_deg = np.degrees(np.arccos(cos_angles))
        fov_half   = max(self.fov_h, self.fov_v) / 2.0
        angle_mask = angles_deg <= fov_half

        return points[depth_mask & angle_mask]

# single entry point for testing the scanner module
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from poses import PoseGenerator

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
    pcds    = scanner.start_scanning(poses)

    total = sum(len(p.points) for p in pcds)
    self.logger.info(f"\nTotal points: {total}")