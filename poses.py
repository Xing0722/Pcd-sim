"""
poses.py

sample camera poses around a mesh for synthetic scanning.
生成相机位姿

Pipeline:
    stable pose (trimesh) → center mesh to world origin → 
    bounding cylinder → camera poses (theta x phi grid)

No coordinate system alignment to a turntable origin is performed here,
since this simulation does not use RoboDK or physical hardware.
"""

import numpy as np
import trimesh
from logger import RoboLogger


class PoseGenerator:
    def __init__(self, mesh,thetas, phis):
        """
        Parameters
        ----------
        mesh   : input mesh
        thetas : horizontal angles in radians
                 e.g. np.linspace(0, 2*pi, 8, endpoint=False)
        phis   : zenith angles in radians
                 0 = directly above, pi/2 = equator level
        """
        self.mesh = mesh
        self.thetas = thetas
        self.phis = phis
        self.poses = None
        self.stabilized_mesh = None
        self.logger = RoboLogger()

    def run(self):
        """
        Full pipeline:
            stable pose → center to world origin →
            bounding cylinder → camera poses
            计算稳定位姿 -> 将网格中心对齐到世界原点 -> 计算最小包围圆柱体 -> 生成相机位姿
        Returns list of 4x4 camera-to-world matrices.
        """
        stable_transform     = self._compute_stable_pose()
        stabilized_mesh      = self.mesh.copy().apply_transform(stable_transform)
        self.stabilized_mesh = self._center_mesh(stabilized_mesh)
        centroid             = self._compute_centroid(self.stabilized_mesh)
        bbox                 = self._compute_bounding_box(self.stabilized_mesh)
        cylinder             = self._compute_bounding_cylinder(bbox)
        self.poses           = self._generate(centroid, cylinder)
        return self.poses

    def _compute_stable_pose(self):
        """
        Compute the most stable resting pose using trimesh's built-in
        stable pose estimation:
            convex hull → face enumeration →
            support polygon test → probability weighting
        Returns the 4x4 transform for the highest-probability stable pose.
        """
        transforms, probs = trimesh.poses.compute_stable_poses(self.mesh)
        best_idx = np.argmax(probs)
        self.logger.info(f"[poses] stable pose selected, probability: {probs[best_idx]:.3f}")
        return transforms[best_idx]

    def _center_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Translate mesh so that:
            - XY centroid is at world origin (0, 0)
            - Z_min is at Z=0 (object rests on the XY plane)
        """
        v  = np.asarray(mesh.vertices)
        cx = v[:, 0].mean()
        cy = v[:, 1].mean()
        cz = v[:, 2].min()
        mesh = mesh.copy()
        mesh.apply_translation([-cx, -cy, -cz])
        return mesh

    def _compute_centroid(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Centroid of the centered mesh:
            XY at (0, 0), Z at half the mesh height.
        """
        v  = np.asarray(mesh.vertices)
        cz = (v[:, 2].max() + v[:, 2].min()) / 2.0
        return np.array([0.0, 0.0, cz])

    def _compute_bounding_box(self, mesh: trimesh.Trimesh) -> dict:
        """
        Axis-aligned bounding box in the stable pose coordinate frame.
        mesh.bounds returns [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        computed on the already-stabilized and centered mesh.
        """
        bounds = mesh.bounds
        return {
            "x_min": bounds[0][0], "x_max": bounds[1][0],
            "y_min": bounds[0][1], "y_max": bounds[1][1],
            "z_min": bounds[0][2], "z_max": bounds[1][2],
        }

    def _compute_bounding_cylinder(self, bbox: dict) -> dict:
        """
        Minimum enclosing cylinder aligned to Z axis, derived from bounding box.
        Radius: circumradius of the XY bounding rectangle
                = sqrt((x_range/2)^2 + (y_range/2)^2)
        Height: z_max - z_min
        """
        x_range = bbox["x_max"] - bbox["x_min"]
        y_range = bbox["y_max"] - bbox["y_min"]
        radius  = np.sqrt((x_range / 2) ** 2 + (y_range / 2) ** 2)
        height  = bbox["z_max"] - bbox["z_min"]
        self.logger.info(f"[poses] bounding box: x={x_range:.4f}  y={y_range:.4f}  z={height:.4f}")
        self.logger.info(f"[poses] bounding cylinder: radius={radius:.4f}  height={height:.4f}")
        return {
            "radius": radius,
            "height": height,
            "z_min":  bbox["z_min"],
            "z_max":  bbox["z_max"],
        }

    def _look_at(self, camera_pos: np.ndarray, centroid: np.ndarray,
                 world_up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
        """
        Build a 4x4 camera-to-world matrix from position and centroid target.
        """
        forward = camera_pos - centroid
        forward /= np.linalg.norm(forward)

        # degenerate case: camera directly above/below
        if abs(np.dot(forward, world_up)) > 0.999:
            world_up = np.array([0.0, 1.0, 0.0])

        right = np.cross(world_up, forward)
        right /= np.linalg.norm(right)

        up = np.cross(forward, right)
        up /= np.linalg.norm(up)

        T = np.eye(4)
        T[:3, 0] = right
        T[:3, 1] = up
        T[:3, 2] = forward
        T[:3, 3] = camera_pos
        return T

    def _generate(self, centroid: np.ndarray, cylinder: dict) -> list[np.ndarray]:
        """
        Sample camera positions on a sphere around the centered mesh
        for every (theta, phi) combination.
        Distance is set internally to 2x the bounding cylinder radius —
        large enough to guarantee the camera is outside the mesh.
        Actual scan distance is not a parameter here; it is determined
        by the ray length in scanner.py.
        """
        r     = cylinder["radius"] * 2.0
        poses = []

        for phi in self.phis:
            for theta in self.thetas:
                cam_pos = np.array([
                    r * np.sin(phi) * np.cos(theta),
                    r * np.sin(phi) * np.sin(theta),
                    centroid[2] + r * np.cos(phi),
                ])
                poses.append(self._look_at(cam_pos, centroid))

        self.logger.info(f"[poses] generated {len(poses)} camera poses "
              f"({len(self.thetas)} theta x {len(self.phis)} phi)")
        return poses


if __name__ == "__main__":
    import sys, os
    mesh_path = sys.argv[1] if len(sys.argv) > 1 else "data/bunny.obj"
    if not os.path.exists(mesh_path):
        self.logger.error(f"mesh not found: {mesh_path}")
        sys.exit(1)

    mesh   = trimesh.load(mesh_path, force="mesh")
    thetas = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    phis   = np.array([np.pi / 4, np.pi / 3])

    gen   = PoseGenerator(mesh, thetas, phis)
    poses = gen.run()
    self.logger.info(f"First pose:\n{poses[0]}")