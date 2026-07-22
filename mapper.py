from pathlib import Path

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    njit = None


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _mark_cache_numba(ix, iy, iz, occ, count_hit, count_miss, touched, touched_count):
        if count_hit[ix, iy, iz] == 0 and count_miss[ix, iy, iz] == 0:
            sy = count_hit.shape[1]
            sz = count_hit.shape[2]
            touched[touched_count] = (ix * sy + iy) * sz + iz
            touched_count += 1

        if occ == 0:
            count_miss[ix, iy, iz] = 1
        else:
            count_hit[ix, iy, iz] += 1
        return touched_count


    @njit(cache=True)
    def _intbound_numba(s, ds):
        if ds < 0.0:
            return _intbound_numba(-s, -ds)
        if ds == 0.0:
            return 1.0e30
        s = s - np.floor(s)
        return (1.0 - s) / ds


    @njit(cache=True)
    def _integrate_point_cloud_numba(
        points_map,
        endpoint_hits,
        camera_pos_map,
        occupancy_buffer,
        grid,
        count_hit,
        count_miss,
        flag_rayend,
        raycast_num,
        touched,
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
        resolution,
        max_ray_length,
        prob_hit_log,
        prob_miss_log,
        clamp_min_log,
        clamp_max_log,
        min_occupancy_log,
    ):
        sx = occupancy_buffer.shape[0]
        sy = occupancy_buffer.shape[1]
        sz = occupancy_buffer.shape[2]
        touched_count = 0
        updated = 0

        update_min0 = camera_pos_map[0]
        update_min1 = camera_pos_map[1]
        update_min2 = camera_pos_map[2]
        update_max0 = camera_pos_map[0]
        update_max1 = camera_pos_map[1]
        update_max2 = camera_pos_map[2]

        cam_ix = int(np.floor((camera_pos_map[0] - x_min) / resolution))
        cam_iy = int(np.floor((camera_pos_map[1] - y_min) / resolution))
        cam_iz = int(np.floor((camera_pos_map[2] - z_min) / resolution))

        for i in range(points_map.shape[0]):
            px = points_map[i, 0]
            py = points_map[i, 1]
            pz = points_map[i, 2]
            point_is_hit = endpoint_hits[i]

            in_map = (
                px >= x_min and px <= x_max and
                py >= y_min and py <= y_max and
                pz >= z_min and pz <= z_max
            )

            if not in_map:
                dx = px - camera_pos_map[0]
                dy = py - camera_pos_map[1]
                dz = pz - camera_pos_map[2]
                min_t = 1.0e30

                if abs(dx) > 1.0e-9:
                    t = (x_min - camera_pos_map[0]) / dx
                    if t > 0.0 and t < min_t:
                        cy = camera_pos_map[1] + t * dy
                        cz = camera_pos_map[2] + t * dz
                        if cy >= y_min - 1.0e-6 and cy <= y_max + 1.0e-6 and cz >= z_min - 1.0e-6 and cz <= z_max + 1.0e-6:
                            min_t = t
                    t = (x_max - camera_pos_map[0]) / dx
                    if t > 0.0 and t < min_t:
                        cy = camera_pos_map[1] + t * dy
                        cz = camera_pos_map[2] + t * dz
                        if cy >= y_min - 1.0e-6 and cy <= y_max + 1.0e-6 and cz >= z_min - 1.0e-6 and cz <= z_max + 1.0e-6:
                            min_t = t

                if abs(dy) > 1.0e-9:
                    t = (y_min - camera_pos_map[1]) / dy
                    if t > 0.0 and t < min_t:
                        cx = camera_pos_map[0] + t * dx
                        cz = camera_pos_map[2] + t * dz
                        if cx >= x_min - 1.0e-6 and cx <= x_max + 1.0e-6 and cz >= z_min - 1.0e-6 and cz <= z_max + 1.0e-6:
                            min_t = t
                    t = (y_max - camera_pos_map[1]) / dy
                    if t > 0.0 and t < min_t:
                        cx = camera_pos_map[0] + t * dx
                        cz = camera_pos_map[2] + t * dz
                        if cx >= x_min - 1.0e-6 and cx <= x_max + 1.0e-6 and cz >= z_min - 1.0e-6 and cz <= z_max + 1.0e-6:
                            min_t = t

                if abs(dz) > 1.0e-9:
                    t = (z_min - camera_pos_map[2]) / dz
                    if t > 0.0 and t < min_t:
                        cx = camera_pos_map[0] + t * dx
                        cy = camera_pos_map[1] + t * dy
                        if cx >= x_min - 1.0e-6 and cx <= x_max + 1.0e-6 and cy >= y_min - 1.0e-6 and cy <= y_max + 1.0e-6:
                            min_t = t
                    t = (z_max - camera_pos_map[2]) / dz
                    if t > 0.0 and t < min_t:
                        cx = camera_pos_map[0] + t * dx
                        cy = camera_pos_map[1] + t * dy
                        if cx >= x_min - 1.0e-6 and cx <= x_max + 1.0e-6 and cy >= y_min - 1.0e-6 and cy <= y_max + 1.0e-6:
                            min_t = t

                if min_t < 1.0e29:
                    t_clip = min_t - 1.0e-3
                    if t_clip < 0.0:
                        t_clip = 0.0
                    px = camera_pos_map[0] + t_clip * dx
                    py = camera_pos_map[1] + t_clip * dy
                    pz = camera_pos_map[2] + t_clip * dz
                else:
                    if px < x_min:
                        px = x_min
                    elif px > x_max:
                        px = x_max
                    if py < y_min:
                        py = y_min
                    elif py > y_max:
                        py = y_max
                    if pz < z_min:
                        pz = z_min
                    elif pz > z_max:
                        pz = z_max
                point_is_hit = False

            dx = px - camera_pos_map[0]
            dy = py - camera_pos_map[1]
            dz = pz - camera_pos_map[2]
            length = np.sqrt(dx * dx + dy * dy + dz * dz)
            if length > max_ray_length and length > 1.0e-9:
                scale = max_ray_length / length
                px = camera_pos_map[0] + dx * scale
                py = camera_pos_map[1] + dy * scale
                pz = camera_pos_map[2] + dz * scale
                point_is_hit = False

            if not (
                px >= x_min and px <= x_max and
                py >= y_min and py <= y_max and
                pz >= z_min and pz <= z_max
            ):
                continue

            ix = int(np.floor((px - x_min) / resolution))
            iy = int(np.floor((py - y_min) / resolution))
            iz = int(np.floor((pz - z_min) / resolution))
            if ix < 0 or ix >= sx or iy < 0 or iy >= sy or iz < 0 or iz >= sz:
                continue

            if point_is_hit:
                touched_count = _mark_cache_numba(ix, iy, iz, 1, count_hit, count_miss, touched, touched_count)
            else:
                touched_count = _mark_cache_numba(ix, iy, iz, 0, count_hit, count_miss, touched, touched_count)

            if px < update_min0:
                update_min0 = px
            if py < update_min1:
                update_min1 = py
            if pz < update_min2:
                update_min2 = pz
            if px > update_max0:
                update_max0 = px
            if py > update_max1:
                update_max1 = py
            if pz > update_max2:
                update_max2 = pz

            if flag_rayend[ix, iy, iz] == raycast_num:
                continue
            flag_rayend[ix, iy, iz] = raycast_num

            ray_dx = px - camera_pos_map[0]
            ray_dy = py - camera_pos_map[1]
            ray_dz = pz - camera_pos_map[2]
            ray_distance = np.sqrt(ray_dx * ray_dx + ray_dy * ray_dy + ray_dz * ray_dz)
            if ray_distance <= resolution:
                continue
            steps = int(ray_distance / resolution)
            if steps <= 1:
                continue
            for k in range(1, steps):
                alpha = k / steps
                qx = camera_pos_map[0] + alpha * ray_dx
                qy = camera_pos_map[1] + alpha * ray_dy
                qz = camera_pos_map[2] + alpha * ray_dz
                qix = int(np.floor((qx - x_min) / resolution))
                qiy = int(np.floor((qy - y_min) / resolution))
                qiz = int(np.floor((qz - z_min) / resolution))
                if qix < 0 or qix >= sx or qiy < 0 or qiy >= sy or qiz < 0 or qiz >= sz:
                    continue
                touched_count = _mark_cache_numba(qix, qiy, qiz, 0, count_hit, count_miss, touched, touched_count)

        for i in range(touched_count):
            adr = touched[i]
            ix = adr // (sy * sz)
            rem = adr - ix * sy * sz
            iy = rem // sz
            iz = rem - iy * sz

            if count_hit[ix, iy, iz] == 0 and count_miss[ix, iy, iz] == 0:
                continue
            if count_hit[ix, iy, iz] >= count_miss[ix, iy, iz]:
                log_update = prob_hit_log
            else:
                log_update = prob_miss_log

            if occupancy_buffer[ix, iy, iz] < clamp_min_log - 1.0e-3:
                occupancy_buffer[ix, iy, iz] = min_occupancy_log
            value = occupancy_buffer[ix, iy, iz] + log_update
            if value < clamp_min_log:
                value = clamp_min_log
            elif value > clamp_max_log:
                value = clamp_max_log
            occupancy_buffer[ix, iy, iz] = value

            count_hit[ix, iy, iz] = 0
            count_miss[ix, iy, iz] = 0
            if value < clamp_min_log - 1.0e-3:
                grid[ix, iy, iz] = -1
            elif value > min_occupancy_log:
                grid[ix, iy, iz] = 1
            else:
                grid[ix, iy, iz] = 0
            updated += 1

        return updated, update_min0, update_min1, update_min2, update_max0, update_max1, update_max2


class OccupancyMap:
    UNKNOWN = -1
    FREE = 0
    OCCUPIED = 1

    def __init__(self, resolution=0.1, volume_size=(10.0, 10.0, 5.0), z_min=0.0):
        self.resolution = float(resolution)
        self.voxel_size = self.resolution
        self.volume_size = np.asarray(volume_size, dtype=float)

        self.x_min = -0.5 * self.volume_size[0] # coord of local map
        self.x_max = 0.5 * self.volume_size[0]
        self.y_min = 0.0
        self.y_max = self.volume_size[1]
        self.z_min = float(z_min)
        self.z_max = self.volume_size[2]

        self.shape = ( # shape of the grdi 100x100x50
            int(np.ceil((self.x_max - self.x_min) / self.resolution)), # 100
            int(np.ceil((self.y_max - self.y_min) / self.resolution)), # 100
            int(np.ceil((self.z_max - self.z_min) / self.resolution)),
        )
        self.p_hit = .7
        self.p_miss = .35
        self.p_min = .12
        self.p_max = .97
        self.p_occ = .8
        self.default_dist = 5.0
        self.unknown_flag = 0.01
        self.max_ray_length = 5 
        self.depth_skip = 2

        self.prob_hit_log = self.logit(self.p_hit)
        self.prob_miss_log = self.logit(self.p_miss)
        self.clamp_min_log = self.logit(self.p_min)
        self.clamp_max_log = self.logit(self.p_max)
        self.min_occupancy_log = self.logit(self.p_occ)

        self.occupancy_buffer = np.full(
            self.shape,
            self.clamp_min_log - self.unknown_flag,
            dtype=np.float32,
        )
        self.occupancy_buffer_inflate = np.zeros(self.shape, dtype=np.int8) # posterior/current belief of the state
        self.distance_buffer_neg = np.full(self.shape, self.default_dist, dtype=np.float32)
        self.distance_buffer = np.full(self.shape, self.default_dist, dtype=np.float32)
        self.count_hit_and_miss = np.zeros(self.shape, dtype=np.int16)
        self.count_hit = np.zeros(self.shape, dtype=np.int16)
        self.count_miss = np.zeros(self.shape, dtype=np.int16)
        self.flag_rayend = np.full(self.shape, -1, dtype=np.int32)
        self.flag_visited = np.full(self.shape, -1, dtype=np.int32)
        self.tmp_buffer1 = np.zeros(self.shape, dtype=np.float32) # temporary
        self.tmp_buffer2 = np.zeros(self.shape, dtype=np.float32)
        self.raycast_num = 0
        self.reset_updated_box = True
        self.cache_voxels = [] # only the voxels from the current 200x200 frame
        self.numba_touched_voxels = np.empty(int(np.prod(self.shape)), dtype=np.int64)
        self.update_min = np.zeros(3, dtype=float) # NEW: min local coordinate touched by updates
        self.update_max = np.zeros(3, dtype=float) # NEW: max local coordinate touched by updates
        self.local_bound_min = np.zeros(3, dtype=int) # NEW: min voxel index touched by local update
        self.local_bound_max = np.asarray(self.shape, dtype=int) - 1 # NEW: max voxel index touched by local update

        self.grid = np.full(self.shape, self.UNKNOWN, dtype=np.int8)

        self.map_origin_world = None
        self.world_to_map_R = None

    @staticmethod
    def logit(p):
        return np.log(p / (1.0 - p))


    def set_map_frame(self, drone_pos_world, drone_yaw):
        origin = np.asarray(drone_pos_world, dtype=float).reshape(3)
        yaw = float(drone_yaw)

        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=float)
        right = np.array([np.sin(yaw), -np.cos(yaw), 0.0], dtype=float)
        up = np.array([0.0, 0.0, 1.0], dtype=float)

        self.map_origin_world = origin
        self.world_to_map_R = np.vstack([right, forward, up])

    def world_to_map(self, points_world): # NEW: convert Isaac/world xyz points into this local map frame
        points_world = np.asarray(points_world, dtype=float).reshape((-1, 3))
        return (self.world_to_map_R @ (points_world - self.map_origin_world).T).T

    def map_to_world(self, points_map): # NEW: convert local map xyz points back into Isaac/world frame
        points_map = np.asarray(points_map, dtype=float).reshape((-1, 3))
        if self.map_origin_world is None or self.world_to_map_R is None:
            return points_map
        return self.map_origin_world + (self.world_to_map_R.T @ points_map.T).T

    def depth_to_world_points(self, depth, K, T_camera_world, min_depth=0.05):
        depth = np.asarray(depth, dtype=np.float32)
        K = np.asarray(K, dtype=float)
        T_camera_world = np.asarray(T_camera_world, dtype=float).copy()

        for col in range(3):
            norm = np.linalg.norm(T_camera_world[:3, col])
            if norm > 0.0:
                T_camera_world[:3, col] /= norm

        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        height, width = depth.shape
        depth_skip = max(1, int(self.depth_skip))
        rows = np.arange(0, height, depth_skip)
        cols = np.arange(0, width, depth_skip)
        depth = depth[rows[:, None], cols]
        us, vs = np.meshgrid(cols, rows)

        finite = np.isfinite(depth)
        too_near = finite & (depth > 0.0) & (depth < min_depth)
        ray_mask = ~too_near
        camera_pos_world = T_camera_world[:3, 3]
        if not np.any(ray_mask):
            return np.empty((0, 3), dtype=float), np.empty(0, dtype=bool), camera_pos_world

        endpoint_hits = (
            finite[ray_mask]
            & (depth[ray_mask] >= min_depth)
            & (depth[ray_mask] <= self.max_ray_length)
        )
        z = np.where(endpoint_hits, depth[ray_mask], self.max_ray_length)
        u = us[ray_mask].astype(np.float32)
        v = vs[ray_mask].astype(np.float32)

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        points_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)
        points_world = (T_camera_world @ points_cam.T).T[:, :3]

        return points_world, endpoint_hits.astype(bool), camera_pos_world

    def map_to_grid(self, points_map): # NEW: convert local map xyz meters into voxel indices
        points_map = np.asarray(points_map, dtype=float).reshape((-1, 3))
        ix = np.floor((points_map[:, 0] - self.x_min) / self.resolution).astype(int)
        iy = np.floor((points_map[:, 1] - self.y_min) / self.resolution).astype(int)
        iz = np.floor((points_map[:, 2] - self.z_min) / self.resolution).astype(int)
        return np.stack([ix, iy, iz], axis=1)

    def grid_to_map(self, indices): # NEW: convert voxel indices into local map xyz voxel centers
        indices = np.asarray(indices, dtype=int).reshape((-1, 3))
        x = (indices[:, 0] + 0.5) * self.resolution + self.x_min
        y = (indices[:, 1] + 0.5) * self.resolution + self.y_min
        z = (indices[:, 2] + 0.5) * self.resolution + self.z_min
        return np.stack([x, y, z], axis=1)

    def in_bounds_indices(self, indices): # NEW: test whether voxel indices are inside the grid
        indices = np.asarray(indices, dtype=int).reshape((-1, 3))
        return (
            (indices[:, 0] >= 0) & (indices[:, 0] < self.shape[0]) &
            (indices[:, 1] >= 0) & (indices[:, 1] < self.shape[1]) &
            (indices[:, 2] >= 0) & (indices[:, 2] < self.shape[2])
        )

    def is_in_map(self, points_map): # NEW: test whether local map xyz points are inside the map box
        points_map = np.asarray(points_map, dtype=float).reshape((-1, 3))
        return (
            (points_map[:, 0] >= self.x_min) & (points_map[:, 0] < self.x_max) &
            (points_map[:, 1] >= self.y_min) & (points_map[:, 1] < self.y_max) &
            (points_map[:, 2] >= self.z_min) & (points_map[:, 2] < self.z_max)
        )

    def is_occupied(self, point_map, unknown_is_occupied=False): # NEW: collision check used by trajectory planner
        point_map = np.asarray(point_map, dtype=float).reshape(3)
        if not self.is_in_map(point_map.reshape(1, 3))[0]:
            return True

        index = self.map_to_grid(point_map.reshape(1, 3))[0]
        if not self.in_bounds_indices(index.reshape(1, 3))[0]:
            return True

        state = self.grid[tuple(index)]
        if state == self.OCCUPIED:
            return True
        if unknown_is_occupied and state == self.UNKNOWN:
            return True
        return False

    def closest_point_in_map(self, point_map, camera_pos_map): # NEW: clip an out-of-bounds point to the map box along the camera ray
        point_map = np.asarray(point_map, dtype=float).reshape(3)
        camera_pos_map = np.asarray(camera_pos_map, dtype=float).reshape(3)
        direction = point_map - camera_pos_map
        bounds_min = np.array([self.x_min, self.y_min, self.z_min], dtype=float)
        bounds_max = np.array([self.x_max, self.y_max, self.z_max], dtype=float)
        min_t = np.inf

        for axis in range(3):
            if abs(direction[axis]) <= 1e-9:
                continue
            for bound in (bounds_min[axis], bounds_max[axis]):
                t = (bound - camera_pos_map[axis]) / direction[axis]
                if 0.0 < t < min_t:
                    candidate = camera_pos_map + t * direction
                    if np.all(candidate >= bounds_min - 1e-6) and np.all(candidate <= bounds_max + 1e-6):
                        min_t = t

        if not np.isfinite(min_t):
            return np.clip(point_map, bounds_min, bounds_max)
        return camera_pos_map + max(0.0, min_t - 1e-3) * direction

    def raycast_indices(self, start_map, end_map): # NEW: return voxel indices along a ray, excluding the endpoint
        start_map = np.asarray(start_map, dtype=float).reshape(3)
        end_map = np.asarray(end_map, dtype=float).reshape(3)
        ray = end_map - start_map
        distance = float(np.linalg.norm(ray))
        if distance <= self.resolution:
            return np.empty((0, 3), dtype=int)

        steps = int(distance / self.resolution)
        if steps <= 1:
            return np.empty((0, 3), dtype=int)

        alphas = np.arange(1, steps, dtype=float) / steps
        points = start_map + alphas[:, None] * ray
        indices = self.map_to_grid(points)
        keep = self.in_bounds_indices(indices)
        indices = indices[keep]
        if len(indices) == 0:
            return indices
        return np.unique(indices, axis=0)

    def set_cache_occupancy(self, indices, occ): #??????????????????????
        indices = np.asarray(indices, dtype=int).reshape((-1, 3)) #  

        keep = self.in_bounds_indices(indices)
        indices = indices[keep]

        if len(indices) == 0:
            return 0

        new_mask = self.count_hit[indices[:, 0], indices[:, 1], indices[:, 2]] == 0
        new_mask &= self.count_miss[indices[:, 0], indices[:, 1], indices[:, 2]] == 0

        new_indices = indices[new_mask]
        if len(new_indices):
            self.cache_voxels.extend(map(tuple, new_indices))

        if occ == 0:
            self.count_miss[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
        elif occ == 1:
            self.count_hit[indices[:, 0], indices[:, 1], indices[:, 2]] += 1

        return len(indices)

    def apply_cached_updates(self): # NEW: apply cached hit/miss evidence to occupancy_buffer using FUEL-style log odds
        if not self.cache_voxels:
            return 0

        updated = 0
        for voxel in self.cache_voxels:
            ix, iy, iz = voxel
            hit_count = self.count_hit[ix, iy, iz]
            miss_count = self.count_miss[ix, iy, iz]

            if hit_count == 0 and miss_count == 0:
                continue

            log_odds_update = self.prob_hit_log if hit_count >= miss_count else self.prob_miss_log

            if self.occupancy_buffer[ix, iy, iz] < self.clamp_min_log - 1e-3:
                self.occupancy_buffer[ix, iy, iz] = self.min_occupancy_log

            self.occupancy_buffer[ix, iy, iz] = np.clip(
                self.occupancy_buffer[ix, iy, iz] + log_odds_update,
                self.clamp_min_log,
                self.clamp_max_log,
            )

            self.count_hit[ix, iy, iz] = 0
            self.count_miss[ix, iy, iz] = 0
            updated += 1

        self.cache_voxels.clear()
        self.refresh_grid_from_occupancy() # NEW: keep the simple UNKNOWN/FREE/OCCUPIED grid synced
        return updated

    def refresh_grid_from_occupancy(self): # NEW: derive hard voxel states from log-odds occupancy_buffer
        self.grid.fill(self.FREE)
        self.grid[self.occupancy_buffer < self.clamp_min_log - 1e-3] = self.UNKNOWN
        self.grid[self.occupancy_buffer > self.min_occupancy_log] = self.OCCUPIED

    def map_stats(self):
        total = int(self.grid.size)
        unknown = int(np.count_nonzero(self.grid == self.UNKNOWN))
        occupied = int(np.count_nonzero(self.grid == self.OCCUPIED))
        free = total - unknown - occupied
        known = total - unknown
        return {
            "known": known,
            "unknown": unknown,
            "free": free,
            "occupied": occupied,
            "total": total,
            "known_pct": round(100.0 * known / max(total, 1), 2),
            "unknown_pct": round(100.0 * unknown / max(total, 1), 2),
        }

    def input_point_cloud(self, points_map, camera_pos_map, endpoint_hits=None):
        # takes point cloud and stores hits and misses, and all values on the ray into the voxel map
        #
        points_map = np.asarray(points_map, dtype=float).reshape((-1, 3))
        camera_pos_map = np.asarray(camera_pos_map, dtype=float).reshape(3)
        if len(points_map) == 0:
            return 0
        if endpoint_hits is None:
            endpoint_hits = np.ones(len(points_map), dtype=bool)
        else:
            endpoint_hits = np.asarray(endpoint_hits, dtype=bool).reshape(-1)
            if len(endpoint_hits) != len(points_map):
                raise ValueError("endpoint_hits must have the same length as points_map")

        if NUMBA_AVAILABLE:
            self.raycast_num += 1
            if self.reset_updated_box:
                self.update_min = camera_pos_map.copy()
                self.update_max = camera_pos_map.copy()
                self.reset_updated_box = False

            updated, min_x, min_y, min_z, max_x, max_y, max_z = _integrate_point_cloud_numba(
                points_map,
                endpoint_hits,
                camera_pos_map,
                self.occupancy_buffer,
                self.grid,
                self.count_hit,
                self.count_miss,
                self.flag_rayend,
                int(self.raycast_num),
                self.numba_touched_voxels,
                float(self.x_min),
                float(self.x_max),
                float(self.y_min),
                float(self.y_max),
                float(self.z_min),
                float(self.z_max),
                float(self.resolution),
                float(self.max_ray_length),
                float(self.prob_hit_log),
                float(self.prob_miss_log),
                float(self.clamp_min_log),
                float(self.clamp_max_log),
                float(self.min_occupancy_log),
            )
            update_min = np.array([min_x, min_y, min_z], dtype=float)
            update_max = np.array([max_x, max_y, max_z], dtype=float)
            self.local_bound_min = self.map_to_grid(update_min.reshape(1, 3))[0]
            self.local_bound_max = self.map_to_grid(update_max.reshape(1, 3))[0]
            self.local_bound_min = np.maximum(self.local_bound_min, 0)
            self.local_bound_max = np.minimum(self.local_bound_max, np.asarray(self.shape, dtype=int) - 1)
            self.update_min = np.minimum(self.update_min, update_min)
            self.update_max = np.maximum(self.update_max, update_max)
            return int(updated)

        self.raycast_num += 1
        self.cache_voxels.clear()
        update_min = camera_pos_map.copy()
        update_max = camera_pos_map.copy()

        if self.reset_updated_box:
            self.update_min = camera_pos_map.copy()
            self.update_max = camera_pos_map.copy()
            self.reset_updated_box = False

        for point_map, endpoint_hit in zip(points_map, endpoint_hits):
            point_map = point_map.reshape(3)
            point_is_hit = bool(endpoint_hit)

            if not self.is_in_map(point_map)[0]:
                point_map = self.closest_point_in_map(point_map, camera_pos_map)
                point_is_hit = False

            length = float(np.linalg.norm(point_map - camera_pos_map))
            if length > self.max_ray_length:
                point_map = (point_map - camera_pos_map) / length * self.max_ray_length + camera_pos_map
                point_is_hit = False

            if not self.is_in_map(point_map)[0]:
                continue

            endpoint_index = self.map_to_grid(point_map)[0]
            if not self.in_bounds_indices(endpoint_index.reshape(1, 3))[0]:
                continue

            self.set_cache_occupancy(endpoint_index.reshape(1, 3), 1 if point_is_hit else 0)

            update_min = np.minimum(update_min, point_map)
            update_max = np.maximum(update_max, point_map)

            ix, iy, iz = endpoint_index
            if self.flag_rayend[ix, iy, iz] == self.raycast_num:
                continue
            self.flag_rayend[ix, iy, iz] = self.raycast_num

            ray_indices = self.raycast_indices(camera_pos_map, point_map)
            self.set_cache_occupancy(ray_indices, 0)

        self.local_bound_min = self.map_to_grid(update_min.reshape(1, 3))[0]
        self.local_bound_max = self.map_to_grid(update_max.reshape(1, 3))[0]
        self.local_bound_min = np.maximum(self.local_bound_min, 0)
        self.local_bound_max = np.minimum(self.local_bound_max, np.asarray(self.shape, dtype=int) - 1)

        self.update_min = np.minimum(self.update_min, update_min)
        self.update_max = np.maximum(self.update_max, update_max)

        return self.apply_cached_updates()

    def input_point_cloud_world(self, points_world, camera_pos_world, endpoint_hits=None): # NEW: convenience wrapper for Isaac/world-frame point clouds
        points_map = self.world_to_map(points_world)
        camera_pos_map = self.world_to_map(np.asarray(camera_pos_world, dtype=float).reshape(1, 3))[0]
        return self.input_point_cloud(points_map, camera_pos_map, endpoint_hits=endpoint_hits)

    def init_tsdf(self): # NEW: initialize TSDF storage without requiring a live viewer window
        import open3d as o3d

        self.o3d = o3d
        self.o3d.utility.set_verbosity_level(self.o3d.utility.VerbosityLevel.Error)
        tsdf_voxel_length = min(0.10, self.resolution)
        self.tsdf_depth_trunc = float(self.max_ray_length)
        self.tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=tsdf_voxel_length,
            sdf_trunc=max(0.25, 3.0 * tsdf_voxel_length),
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        self.viewer_tsdf_frames = 0
        self.tsdf_mesh_update_interval = 10 # viewer updates; driver updates viewer every 60 sim steps, so 10 ~= 600 sim steps

    def start_tsdf_recorder(self): # NEW: record TSDF frames for offline inspection, no live Open3D window
        self.init_tsdf()

    def start_viewer(self): # NEW: start Open3D viewer for this occupancy map
        self.init_tsdf()

        self.viewer = self.o3d.visualization.Visualizer()
        self.viewer.create_window("TSDF + Frontier Map", width=1280, height=720)

        self.tsdf_mesh = self.o3d.geometry.TriangleMesh()
        self.occupied_cloud = self.o3d.geometry.PointCloud()
        self.frontier_cloud = self.o3d.geometry.PointCloud()
        self.frontier_blob = self.o3d.geometry.LineSet()
        self.centroid_cloud = self.o3d.geometry.PointCloud() # NEW: frontier centroid overlay
        self.best_viewpoint_cloud = self.o3d.geometry.PointCloud() # NEW: selected viewpoint overlay
        self.best_viewpoint_heading = self.o3d.geometry.LineSet() # NEW: selected viewpoint heading line


        self.viewer.add_geometry(self.tsdf_mesh)
        self.viewer.add_geometry(self.occupied_cloud)
        self.viewer.add_geometry(self.frontier_cloud)
        self.viewer.add_geometry(self.frontier_blob)
        self.viewer.add_geometry(self.centroid_cloud)
        self.viewer.add_geometry(self.best_viewpoint_cloud) # NEW: magenta selected viewpoint point
        self.viewer.add_geometry(self.best_viewpoint_heading) # NEW: magenta selected viewpoint heading
        self.viewer_needs_reset = True # NEW: reset view once after real points are available

    def save_tsdf_mesh(self, path): # NEW: save reconstructed surface for offline viewing after the run
        if not hasattr(self, "tsdf_volume"):
            return None

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        mesh = self.tsdf_volume.extract_triangle_mesh()
        if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            return None

        mesh.compute_vertex_normals()
        ok = self.o3d.io.write_triangle_mesh(str(path), mesh)
        if not ok:
            return None
        return {
            "path": str(path),
            "vertices": int(len(mesh.vertices)),
            "triangles": int(len(mesh.triangles)),
        }

    def save_point_cloud(self, path, points, colors=None): # NEW: save lightweight overlays alongside offline TSDF snapshots
        if not hasattr(self, "o3d"):
            return None

        points = np.asarray(points, dtype=float).reshape((-1, 3))
        if len(points) == 0:
            return None

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cloud = self.o3d.geometry.PointCloud()
        cloud.points = self.o3d.utility.Vector3dVector(points)
        if colors is not None:
            colors = np.asarray(colors, dtype=float).reshape((-1, 3))
            if len(colors) == len(points):
                cloud.colors = self.o3d.utility.Vector3dVector(colors)
        ok = self.o3d.io.write_point_cloud(str(path), cloud)
        if not ok:
            return None
        return {
            "path": str(path),
            "points": int(len(points)),
        }

    def save_line_set(self, path, points, lines, colors=None): # NEW: save heading/yaw overlays for offline snapshots
        if not hasattr(self, "o3d"):
            return None

        points = np.asarray(points, dtype=float).reshape((-1, 3))
        lines = np.asarray(lines, dtype=np.int32).reshape((-1, 2))
        if len(points) == 0 or len(lines) == 0:
            return None

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        line_set = self.o3d.geometry.LineSet()
        line_set.points = self.o3d.utility.Vector3dVector(points)
        line_set.lines = self.o3d.utility.Vector2iVector(lines)
        if colors is not None:
            colors = np.asarray(colors, dtype=float).reshape((-1, 3))
            if len(colors) == len(lines):
                line_set.colors = self.o3d.utility.Vector3dVector(colors)
        ok = self.o3d.io.write_line_set(str(path), line_set)
        if not ok:
            return None
        return {
            "path": str(path),
            "lines": int(len(lines)),
        }

    def save_sphere_markers(self, path, points, colors=None, radius=0.12): # NEW: save centroids as visible 3D markers
        if not hasattr(self, "o3d"):
            return None

        points = np.asarray(points, dtype=float).reshape((-1, 3))
        if len(points) == 0:
            return None

        colors = np.asarray(colors, dtype=float).reshape((-1, 3)) if colors is not None else None
        if colors is None or len(colors) != len(points):
            colors = np.tile(np.array([[0.0, 0.0, 0.0]], dtype=float), (len(points), 1))

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh = self.o3d.geometry.TriangleMesh()
        for point, color in zip(points, colors):
            sphere = self.o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=8)
            sphere.translate(point)
            sphere.paint_uniform_color(color)
            mesh += sphere

        mesh.compute_vertex_normals()
        ok = self.o3d.io.write_triangle_mesh(str(path), mesh)
        if not ok:
            return None
        return {
            "path": str(path),
            "markers": int(len(points)),
        }

    def points_to_voxel_grid(self, points, color, colors=None): # NEW: render map cells as real Open3D voxels, not point sprites
        points = np.asarray(points, dtype=float).reshape((-1, 3))
        point_cloud = self.o3d.geometry.PointCloud()
        if not len(points):
            return self.o3d.geometry.VoxelGrid()

        point_cloud.points = self.o3d.utility.Vector3dVector(points)
        colors = np.asarray(colors, dtype=float).reshape((-1, 3)) if colors is not None else None
        if colors is None or len(colors) != len(points):
            colors = np.tile(np.asarray(color, dtype=float).reshape((1, 3)), (len(points), 1))
        point_cloud.colors = self.o3d.utility.Vector3dVector(colors)
        min_bound = np.array([self.x_min, self.y_min, self.z_min], dtype=float)
        max_bound = np.array([self.x_max, self.y_max, self.z_max], dtype=float)
        return self.o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            point_cloud,
            voxel_size=self.resolution,
            min_bound=min_bound,
            max_bound=max_bound,
        )

    def replace_viewer_geometry(self, attr_name, geometry): # NEW: VoxelGrid contents are easiest to refresh by replacing geometry
        old_geometry = getattr(self, attr_name)
        self.viewer.remove_geometry(old_geometry, reset_bounding_box=False)
        setattr(self, attr_name, geometry)
        self.viewer.add_geometry(geometry, reset_bounding_box=False)

    def to_rgb_uint8(self, rgb, depth_shape): # NEW: prepare Isaac RGB frames for Open3D TSDF integration
        if rgb is None:
            height, width = depth_shape
            return np.full((height, width, 3), 160, dtype=np.uint8)
        rgb = np.asarray(rgb)
        if rgb.ndim != 3:
            height, width = depth_shape
            return np.full((height, width, 3), 160, dtype=np.uint8)
        if rgb.shape[-1] == 4:
            rgb = rgb[:, :, :3]
        rgb = rgb.astype(np.uint8, copy=False)
        height = min(rgb.shape[0], depth_shape[0])
        width = min(rgb.shape[1], depth_shape[1])
        return np.ascontiguousarray(rgb[:height, :width, :3])

    def to_depth_float(self, depth, rgb_shape): # NEW: Open3D wants invalid depth as 0, in meters
        depth = np.asarray(depth, dtype=np.float32)
        height = min(depth.shape[0], rgb_shape[0])
        width = min(depth.shape[1], rgb_shape[1])
        depth = np.ascontiguousarray(depth[:height, :width])
        valid = np.isfinite(depth) & (depth >= 0.05) & (depth <= self.tsdf_depth_trunc)
        return np.where(valid, depth, 0.0).astype(np.float32)

    def integrate_tsdf(self, rgb, depth, K, T_camera_world): # NEW: viewer-only surface reconstruction
        if rgb is None or depth is None or K is None or T_camera_world is None:
            return

        rgb = self.to_rgb_uint8(rgb, np.asarray(depth).shape)
        depth = self.to_depth_float(depth, rgb.shape[:2])
        if not np.any(depth > 0.0):
            return

        K = np.asarray(K, dtype=float)
        height, width = depth.shape
        intrinsic = self.o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            float(K[0, 0]),
            float(K[1, 1]),
            float(K[0, 2]),
            float(K[1, 2]),
        )

        T_camera_world = np.asarray(T_camera_world, dtype=float).copy()
        for col in range(3):
            norm = np.linalg.norm(T_camera_world[:3, col])
            if norm > 0.0:
                T_camera_world[:3, col] /= norm
        extrinsic = np.linalg.inv(T_camera_world)

        rgbd = self.o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=self.o3d.geometry.Image(rgb),
            depth=self.o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=self.tsdf_depth_trunc,
            convert_rgb_to_intensity=False,
        )
        self.tsdf_volume.integrate(rgbd, intrinsic, extrinsic)
        self.viewer_tsdf_frames += 1

    def update_tsdf_mesh(self, force=False): # NEW: replace the live mesh with current TSDF surface
        if not force and (self.viewer_tsdf_frames % self.tsdf_mesh_update_interval) != 0:
            return len(self.tsdf_mesh.vertices)

        mesh = self.tsdf_volume.extract_triangle_mesh()
        if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            return 0
        mesh.compute_vertex_normals()
        self.tsdf_mesh.vertices = mesh.vertices
        self.tsdf_mesh.triangles = mesh.triangles
        self.tsdf_mesh.vertex_normals = mesh.vertex_normals
        self.tsdf_mesh.triangle_normals = mesh.triangle_normals
        if mesh.has_vertex_colors():
            self.tsdf_mesh.vertex_colors = mesh.vertex_colors
        self.viewer.update_geometry(self.tsdf_mesh)
        return len(mesh.vertices)

    def update_frontier_blob(self, frontier_points, max_blob_points=1200): # NEW: see-through frontier hull/outline
        if len(frontier_points) < 4:
            self.frontier_blob.points = self.o3d.utility.Vector3dVector(np.empty((0, 3), dtype=float))
            self.frontier_blob.lines = self.o3d.utility.Vector2iVector(np.empty((0, 2), dtype=np.int32))
            self.frontier_blob.colors = self.o3d.utility.Vector3dVector(np.empty((0, 3), dtype=float))
            self.viewer.update_geometry(self.frontier_blob)
            return

        points = frontier_points
        if len(points) > max_blob_points:
            step = max(1, len(points) // max_blob_points)
            points = points[::step]

        point_cloud = self.o3d.geometry.PointCloud()
        point_cloud.points = self.o3d.utility.Vector3dVector(points)
        try:
            alpha = max(2.5 * self.resolution, 0.35)
            mesh = self.o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha)
            line_set = self.o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        except Exception:
            line_set = self.o3d.geometry.LineSet()

        if len(line_set.lines) == 0:
            line_set.points = self.o3d.utility.Vector3dVector(np.empty((0, 3), dtype=float))
            line_set.lines = self.o3d.utility.Vector2iVector(np.empty((0, 2), dtype=np.int32))
            line_set.colors = self.o3d.utility.Vector3dVector(np.empty((0, 3), dtype=float))
        else:
            line_set.paint_uniform_color([1.0, 0.45, 0.0])

        self.frontier_blob.points = line_set.points
        self.frontier_blob.lines = line_set.lines
        self.frontier_blob.colors = line_set.colors
        self.viewer.update_geometry(self.frontier_blob)

    def focus_viewer(self, occupied_points, frontier_points, centroid_points, best_viewpoint): # NEW: start close to the active map instead of Open3D's default far view
        candidates = []
        for points in (frontier_points, occupied_points, centroid_points, best_viewpoint):
            points = np.asarray(points, dtype=float).reshape((-1, 3))
            if len(points):
                candidates.append(points)

        if len(self.tsdf_mesh.vertices):
            bbox = self.tsdf_mesh.get_axis_aligned_bounding_box()
            extent = bbox.get_extent()
            if np.isfinite(extent).all() and np.linalg.norm(extent) > 0.0:
                candidates.append(np.asarray(bbox.get_box_points(), dtype=float))

        if not candidates:
            return

        points = np.vstack(candidates)
        lookat = np.mean(points, axis=0)
        view = self.viewer.get_view_control()
        view.set_lookat(lookat.tolist())
        view.set_front([0.65, -0.65, 0.38])
        view.set_up([0.0, 0.0, 1.0])
        view.set_zoom(0.45)

    def update_viewer(self, show_free=True, max_points=50000, frontier_points=None, frontier_colors=None, centroid_points=None, best_viewpoint=None, best_yaw=None, rgb=None, depth=None, K=None, T_camera_world=None):
        if not hasattr(self, "viewer"):
            return

        self.integrate_tsdf(rgb, depth, K, T_camera_world)
        mesh_vertices = self.update_tsdf_mesh(force=True)

        occupied_indices = np.argwhere(self.grid == self.OCCUPIED)

        if len(occupied_indices) > max_points:
            occupied_indices = occupied_indices[::max(1, len(occupied_indices) // max_points)]

        occupied_points = self.grid_to_map(occupied_indices) if len(occupied_indices) else np.empty((0, 3), dtype=float)
        frontier_points = np.asarray(frontier_points, dtype=float).reshape((-1, 3)) if frontier_points is not None else np.empty((0, 3), dtype=float)
        frontier_colors = np.asarray(frontier_colors, dtype=float).reshape((-1, 3)) if frontier_colors is not None else None
        if frontier_colors is None or len(frontier_colors) != len(frontier_points):
            frontier_colors = np.tile(np.array([[1.0, 0.55, 0.0]], dtype=float), (len(frontier_points), 1))
        centroid_points = np.asarray(centroid_points, dtype=float).reshape((-1, 3)) if centroid_points is not None else np.empty((0, 3), dtype=float)
        best_viewpoint = np.asarray(best_viewpoint, dtype=float).reshape((1, 3)) if best_viewpoint is not None else np.empty((0, 3), dtype=float)
        occupied_points = self.map_to_world(occupied_points)
        frontier_points = self.map_to_world(frontier_points)
        centroid_points = self.map_to_world(centroid_points)
        best_viewpoint = self.map_to_world(best_viewpoint)

        self.occupied_cloud.points = self.o3d.utility.Vector3dVector(occupied_points)
        self.occupied_cloud.colors = self.o3d.utility.Vector3dVector(
            np.tile(np.array([[0.0, 0.2, 1.0]], dtype=float), (len(occupied_points), 1))
        )
        self.viewer.update_geometry(self.occupied_cloud)

        self.frontier_cloud.points = self.o3d.utility.Vector3dVector(frontier_points)
        self.frontier_cloud.colors = self.o3d.utility.Vector3dVector(frontier_colors)
        self.viewer.update_geometry(self.frontier_cloud)
        self.update_frontier_blob(frontier_points)

        self.centroid_cloud.points = self.o3d.utility.Vector3dVector(centroid_points)
        self.centroid_cloud.colors = self.o3d.utility.Vector3dVector(
            np.tile(np.array([[0.0, 0.0, 0.0]], dtype=float), (len(centroid_points), 1))
        )

        self.best_viewpoint_cloud.points = self.o3d.utility.Vector3dVector(best_viewpoint)
        self.best_viewpoint_cloud.colors = self.o3d.utility.Vector3dVector(
            np.tile(np.array([[1.0, 0.0, 1.0]], dtype=float), (len(best_viewpoint), 1))
        )

        if len(best_viewpoint) and best_yaw is not None:
            heading_start = best_viewpoint[0]
            heading_dir_map = np.array([np.cos(best_yaw), np.sin(best_yaw), 0.0], dtype=float)
            heading_dir_world = self.world_to_map_R.T @ heading_dir_map if self.world_to_map_R is not None else heading_dir_map
            heading_end = heading_start + 0.75 * heading_dir_world
            self.best_viewpoint_heading.points = self.o3d.utility.Vector3dVector(np.vstack([heading_start, heading_end]))
            self.best_viewpoint_heading.lines = self.o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
            self.best_viewpoint_heading.colors = self.o3d.utility.Vector3dVector(np.array([[1.0, 0.0, 1.0]], dtype=float))
        else:
            self.best_viewpoint_heading.points = self.o3d.utility.Vector3dVector(np.empty((0, 3), dtype=float))
            self.best_viewpoint_heading.lines = self.o3d.utility.Vector2iVector(np.empty((0, 2), dtype=np.int32))
            self.best_viewpoint_heading.colors = self.o3d.utility.Vector3dVector(np.empty((0, 3), dtype=float))

        self.viewer.update_geometry(self.centroid_cloud)
        self.viewer.update_geometry(self.best_viewpoint_cloud)
        self.viewer.update_geometry(self.best_viewpoint_heading)

        render_option = self.viewer.get_render_option()
        if render_option is not None:
            render_option.point_size = 5.0
            render_option.line_width = 2.0

        self.viewer.poll_events()
        self.viewer.update_renderer()

        if self.viewer_needs_reset and (mesh_vertices + len(occupied_points) + len(frontier_points) + len(centroid_points) + len(best_viewpoint)) > 0:
            self.viewer.reset_view_point(True)
            self.focus_viewer(occupied_points, frontier_points, centroid_points, best_viewpoint)
            self.viewer_needs_reset = False
            self.viewer.poll_events()
            self.viewer.update_renderer()
