import numpy as np


class RangeProjection(object):
    '''
    Project the 3D point cloud to 2D data with range projection

    Adapted from Z. Zhuang et al. https://github.com/ICEORY/PMF
    '''

    def __init__(self, fov_up, fov_down, proj_w, proj_h, fov_left=-180, fov_right=180):

        # check params
        assert fov_up >= 0 and fov_down <= 0, 'require fov_up >= 0 and fov_down <= 0, while fov_up/fov_down is {}/{}'.format(
            fov_up, fov_down)
        assert fov_right >= 0 and fov_left <= 0, 'require fov_right >= 0 and fov_left <= 0, while fov_right/fov_left is {}/{}'.format(
            fov_right, fov_left)

        # params of fov angeles
        self.fov_up = fov_up / 180.0 * np.pi
        self.fov_down = fov_down / 180.0 * np.pi
        self.fov_v = abs(self.fov_up) + abs(self.fov_down)

        self.fov_left = fov_left / 180.0 * np.pi
        self.fov_right = fov_right / 180.0 * np.pi
        self.fov_h = abs(self.fov_left) + abs(self.fov_right)

        # params of proj img size
        self.proj_w = proj_w
        self.proj_h = proj_h

        self.cached_data = {}

    def doProjection(self, pointcloud: np.ndarray):
        self.cached_data = {}
        # get depth of all points
        depth = np.linalg.norm(pointcloud[:, :3], 2, axis=1) + 1e-5
        # get point cloud components
        x = pointcloud[:, 0]
        y = pointcloud[:, 1]
        z = pointcloud[:, 2]

        # get angles of all points
        yaw = -np.arctan2(y, x)
        pitch = np.arcsin(z / depth)

        # get projection in image coords
        proj_x = (yaw + abs(self.fov_left)) / self.fov_h
        # proj_x = 0.5 * (yaw / np.pi + 1.0) # normalized in [0, 1]
        proj_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov_v  # normalized in [0, 1]

        # scale to image size using angular resolution
        proj_x *= self.proj_w
        proj_y *= self.proj_h

        px = np.maximum(np.minimum(self.proj_w, proj_x), 0) # or proj_x.copy()
        py = np.maximum(np.minimum(self.proj_h, proj_y), 0) # or proj_y.copy()

        # round and clamp for use as index
        proj_x = np.maximum(np.minimum(
            self.proj_w - 1, np.floor(proj_x)), 0).astype(np.int32)

        proj_y = np.maximum(np.minimum(
            self.proj_h - 1, np.floor(proj_y)), 0).astype(np.int32)

        self.cached_data['uproj_x_idx'] = proj_x.copy()
        self.cached_data['uproj_y_idx'] = proj_y.copy()
        self.cached_data['uproj_depth'] = depth.copy()
        self.cached_data['px'] = px
        self.cached_data['py'] = py

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        pointcloud = pointcloud[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # get projection result
        proj_range = np.full((self.proj_h, self.proj_w), -1, dtype=np.float32)
        proj_range[proj_y, proj_x] = depth

        proj_pointcloud = np.full(
            (self.proj_h, self.proj_w, pointcloud.shape[1]), -1, dtype=np.float32)
        proj_pointcloud[proj_y, proj_x] = pointcloud

        proj_idx = np.full((self.proj_h, self.proj_w), -1, dtype=np.int32)
        proj_idx[proj_y, proj_x] = indices

        proj_mask = (proj_idx > 0).astype(np.int32)

        return proj_pointcloud, proj_range, proj_idx, proj_mask

    def back_project_range(self, range_depth: np.ndarray) -> np.ndarray:
        """Back-project a depth/range map to a 3-D point cloud.

        Inverts ``doProjection`` using the stored FOV parameters.  For each
        pixel ``(u, v)`` with valid depth ``r > 0`` the 3-D point is:

            yaw   = (u / proj_w) * fov_h  - |fov_left|
            pitch = (1 - v / proj_h) * fov_v - |fov_down|
            x =  r * cos(pitch) * cos(yaw)
            y = -r * cos(pitch) * sin(yaw)   # sign from yaw = -atan2(y, x)
            z =  r * sin(pitch)

        This is the inverse of the spherical projection used in DiffLoc and
        adapted here from ``doProjection``.

        Args:
            range_depth: ``(proj_h, proj_w)`` float array of per-pixel depth
                         values in metres.  Non-positive values are invalid.

        Returns:
            ``(N, 3)`` float32 array of valid 3-D points ``(x, y, z)``.
        """
        H, W = range_depth.shape
        assert H == self.proj_h and W == self.proj_w, (
            f"Depth map shape {range_depth.shape} does not match "
            f"projection dimensions ({self.proj_h}, {self.proj_w})"
        )

        # Pixel coordinate grids
        u = np.arange(W, dtype=np.float32)
        v = np.arange(H, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)                          # (H, W) each

        # Recover spherical angles (inverse of doProjection)
        yaw   = (uu / W) * self.fov_h - abs(self.fov_left)   # (H, W)
        pitch = (1.0 - vv / H) * self.fov_v - abs(self.fov_down)  # (H, W)

        valid = range_depth > 0.0
        r = range_depth[valid]
        p = pitch[valid]
        y = yaw[valid]

        x_pts =  r * np.cos(p) * np.cos(y)
        y_pts = -r * np.cos(p) * np.sin(y)    # negative: yaw = -atan2(y, x)
        z_pts =  r * np.sin(p)

        return np.stack([x_pts, y_pts, z_pts], axis=1).astype(np.float32)


class ScanProjection(object):
    '''
    Project the 3D point cloud to 2D data with range projection

    Adapted from A. Milioto et al. https://github.com/PRBonn/lidar-bonnetal
    '''

    def __init__(self, proj_w, proj_h):
        # params of proj img size
        self.proj_w = proj_w
        self.proj_h = proj_h

        self.cached_data = {}

    def doProjection(self, pointcloud: np.ndarray):
        self.cached_data = {}
        # get depth of all points
        depth = np.linalg.norm(pointcloud[:, :3], 2, axis=1)
        # get point cloud components
        x = pointcloud[:, 0]
        y = pointcloud[:, 1]
        z = pointcloud[:, 2]

        # get angles of all points
        yaw = -np.arctan2(y, -x)
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        #breakpoint()
        new_raw = np.nonzero((proj_x[1:] < 0.2) * (proj_x[:-1] > 0.8))[0] + 1
        proj_y = np.zeros_like(proj_x)
        proj_y[new_raw] = 1
        proj_y = np.cumsum(proj_y)
        # scale to image size using angular resolution
        proj_x = proj_x * self.proj_w - 0.001

        # print(f'proj_y: [{proj_y.min()} - {proj_y.max()}] - ({(proj_y < self.proj_h).astype(np.int32).sum()} - {(proj_y >= self.proj_h).astype(np.int32).sum()})')

        px = proj_x.copy()
        py = proj_y.copy()

        # round and clamp for use as index
        proj_x = np.maximum(np.minimum(
            self.proj_w - 1, np.floor(proj_x)), 0).astype(np.int32)

        proj_y = np.maximum(np.minimum(
            self.proj_h - 1, np.floor(proj_y)), 0).astype(np.int32)

        self.cached_data['uproj_x_idx'] = proj_x.copy()
        self.cached_data['uproj_y_idx'] = proj_y.copy()
        self.cached_data['uproj_depth'] = depth.copy()
        self.cached_data['px'] = px
        self.cached_data['py'] = py

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        pointcloud = pointcloud[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # get projection result
        proj_range = np.full((self.proj_h, self.proj_w), -1, dtype=np.float32)
        proj_range[proj_y, proj_x] = depth

        proj_pointcloud = np.full(
            (self.proj_h, self.proj_w, pointcloud.shape[1]), -1, dtype=np.float32)
        proj_pointcloud[proj_y, proj_x] = pointcloud

        proj_idx = np.full((self.proj_h, self.proj_w), -1, dtype=np.int32)
        proj_idx[proj_y, proj_x] = indices

        proj_mask = (proj_idx > 0).astype(np.int32)

        return proj_pointcloud, proj_range, proj_idx, proj_mask