from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
import heapq
import os
import shlex
import shutil
import subprocess
import tempfile
import time

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    njit = None


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _frontier_candidate_indices_numba(grid, frontier_flag, search_min, search_max, candidates):
        count = 0
        sx = grid.shape[0]
        sy = grid.shape[1]
        sz = grid.shape[2]
        for ix in range(search_min[0], search_max[0] + 1):
            for iy in range(search_min[1], search_max[1] + 1):
                for iz in range(search_min[2], search_max[2] + 1):
                    if frontier_flag[ix, iy, iz] != 0:
                        continue
                    if grid[ix, iy, iz] != 0:
                        continue

                    has_unknown = False
                    if ix > 0 and grid[ix - 1, iy, iz] == -1:
                        has_unknown = True
                    elif ix + 1 < sx and grid[ix + 1, iy, iz] == -1:
                        has_unknown = True
                    elif iy > 0 and grid[ix, iy - 1, iz] == -1:
                        has_unknown = True
                    elif iy + 1 < sy and grid[ix, iy + 1, iz] == -1:
                        has_unknown = True
                    elif iz > 0 and grid[ix, iy, iz - 1] == -1:
                        has_unknown = True
                    elif iz + 1 < sz and grid[ix, iy, iz + 1] == -1:
                        has_unknown = True

                    if has_unknown:
                        candidates[count, 0] = ix
                        candidates[count, 1] = iy
                        candidates[count, 2] = iz
                        count += 1
        return count


    @njit(cache=True)
    def _count_visible_cells_numba(
        grid,
        pos,
        yaw,
        cluster,
        resolution,
        x_min,
        y_min,
        z_min,
        horizontal_fov,
        vertical_fov,
    ):
        sx = grid.shape[0]
        sy = grid.shape[1]
        sz = grid.shape[2]
        forward_x = np.cos(yaw)
        forward_y = np.sin(yaw)
        half_horizontal_fov = 0.5 * horizontal_fov
        half_vertical_fov = 0.5 * vertical_fov
        visib_num = 0

        for i in range(cluster.shape[0]):
            dx = cluster[i, 0] - pos[0]
            dy = cluster[i, 1] - pos[1]
            dz = cluster[i, 2] - pos[2]
            distance = np.sqrt(dx * dx + dy * dy + dz * dz)
            if distance < 1.0e-6:
                continue

            horizontal_distance = np.sqrt(dx * dx + dy * dy)
            if horizontal_distance < 1.0e-6:
                continue

            horizontal_dot = (dx / horizontal_distance) * forward_x + (dy / horizontal_distance) * forward_y
            if horizontal_dot < -1.0:
                horizontal_dot = -1.0
            elif horizontal_dot > 1.0:
                horizontal_dot = 1.0
            horizontal_angle = np.arccos(horizontal_dot)
            vertical_angle = np.arctan2(dz, horizontal_distance)

            if horizontal_angle > half_horizontal_fov:
                continue
            if abs(vertical_angle) > half_vertical_fov:
                continue

            visible = True
            if distance > resolution:
                steps = int(distance / resolution)
                if steps > 1:
                    for k in range(1, steps):
                        alpha = k / steps
                        qx = pos[0] + alpha * dx
                        qy = pos[1] + alpha * dy
                        qz = pos[2] + alpha * dz
                        ix = int(np.floor((qx - x_min) / resolution))
                        iy = int(np.floor((qy - y_min) / resolution))
                        iz = int(np.floor((qz - z_min) / resolution))

                        if ix < 0 or ix >= sx or iy < 0 or iy >= sy or iz < 0 or iz >= sz:
                            continue

                        state = grid[ix, iy, iz]
                        if state == 1 or state == -1:
                            visible = False
                            break

            if visible:
                visib_num += 1

        return visib_num


    @njit(cache=True)
    def _is_frontier_cell_numba(grid, frontier_flag, ix, iy, iz):
        sx = grid.shape[0]
        sy = grid.shape[1]
        sz = grid.shape[2]
        if ix < 0 or ix >= sx or iy < 0 or iy >= sy or iz < 0 or iz >= sz:
            return False
        if frontier_flag[ix, iy, iz] == 1:
            return False
        if grid[ix, iy, iz] != 0:
            return False
        if ix > 0 and grid[ix - 1, iy, iz] == -1:
            return True
        if ix + 1 < sx and grid[ix + 1, iy, iz] == -1:
            return True
        if iy > 0 and grid[ix, iy - 1, iz] == -1:
            return True
        if iy + 1 < sy and grid[ix, iy + 1, iz] == -1:
            return True
        if iz > 0 and grid[ix, iy, iz - 1] == -1:
            return True
        if iz + 1 < sz and grid[ix, iy, iz + 1] == -1:
            return True
        return False


    @njit(cache=True)
    def _expand_frontiers_numba(
        grid,
        frontier_flag,
        candidates,
        candidate_count,
        queue,
        cluster_indices,
        cluster_offsets,
        cluster_sizes,
        resolution,
        x_min,
        y_min,
        z_min,
        cluster_min,
    ):
        cluster_count = 0
        total_indices = 0

        for c in range(candidate_count):
            start_ix = candidates[c, 0]
            start_iy = candidates[c, 1]
            start_iz = candidates[c, 2]
            if not _is_frontier_cell_numba(grid, frontier_flag, start_ix, start_iy, start_iz):
                continue

            cluster_start = total_indices
            head = 0
            tail = 0
            queue[tail, 0] = start_ix
            queue[tail, 1] = start_iy
            queue[tail, 2] = start_iz
            tail += 1
            frontier_flag[start_ix, start_iy, start_iz] = 1
            cluster_indices[total_indices, 0] = start_ix
            cluster_indices[total_indices, 1] = start_iy
            cluster_indices[total_indices, 2] = start_iz
            total_indices += 1

            while head < tail:
                cur_ix = queue[head, 0]
                cur_iy = queue[head, 1]
                cur_iz = queue[head, 2]
                head += 1

                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        for dz in range(-1, 2):
                            if dx == 0 and dy == 0 and dz == 0:
                                continue
                            nix = cur_ix + dx
                            niy = cur_iy + dy
                            niz = cur_iz + dz

                            if not _is_frontier_cell_numba(grid, frontier_flag, nix, niy, niz):
                                continue

                            pos_z = (niz + 0.5) * resolution + z_min
                            if pos_z < 0.4:
                                continue

                            frontier_flag[nix, niy, niz] = 1
                            queue[tail, 0] = nix
                            queue[tail, 1] = niy
                            queue[tail, 2] = niz
                            tail += 1
                            cluster_indices[total_indices, 0] = nix
                            cluster_indices[total_indices, 1] = niy
                            cluster_indices[total_indices, 2] = niz
                            total_indices += 1

            cluster_size = total_indices - cluster_start
            if cluster_size > cluster_min:
                cluster_offsets[cluster_count] = cluster_start
                cluster_sizes[cluster_count] = cluster_size
                cluster_count += 1

        return cluster_count


def timing_section(timing, name):
    return timing.section(name) if timing is not None else nullcontext()


@dataclass
class Viewpoint:
    pos: np.ndarray
    yaw: float
    visib_num: int


@dataclass
class Frontier:
    indices: np.ndarray
    cells: np.ndarray
    filtered_cells: np.ndarray = None
    average: np.ndarray = None
    id: int = -1
    viewpoints: list = field(default_factory=list)
    box_min: np.ndarray = None
    box_max: np.ndarray = None
    index_min: np.ndarray = None
    index_max: np.ndarray = None
    paths: list = field(default_factory=list)
    costs: list = field(default_factory=list)


# need to create a cost later

class FrontierFinder:
    def __init__(
            self,
            occupancy_map,
            cluster_min=100,
            cluster_size_xy=1.5,
            cluster_size_z=10.0,
            down_sample=2,
            min_candidate_dist=0.75,
            min_candidate_clearance=0.25,
            candidate_dphi=np.deg2rad(30.0), # was 15
            candidate_rmin=.5,
            candidate_rmax=2,
            candidate_rnum=3,
            min_visib_num=10,
            min_view_finish_fraction=0.2,
            inflate=(1.0, 1.0, 0.5),
            horizontal_fov=np.deg2rad(80.0),
            vertical_fov=np.deg2rad(60.0),
            max_vel=1.0,
            max_yaw_rate=np.deg2rad(60.0),
            direction_weight=1.5,
            refined_num=7,
            refined_radius=5.0,
            top_view_num=15,
            max_decay=0.8,
            global_astar_resolution=0.4,
            global_astar_max_time=0.001,
            global_astar_fail_cost=1000.0,
            lkh_command=None,
            tsp_dir=None,
            debug_viewpoint_sampling=False,
    ):
        self.map = occupancy_map

        self.frontier_flag = np.zeros(self.map.shape, dtype=np.uint8)
        frontier_buffer_size = int(np.prod(self.map.shape))
        self.frontier_candidate_buffer = np.empty((frontier_buffer_size, 3), dtype=np.int32)
        self.frontier_queue_buffer = np.empty((frontier_buffer_size, 3), dtype=np.int32)
        self.frontier_cluster_indices_buffer = np.empty((frontier_buffer_size, 3), dtype=np.int32)
        self.frontier_cluster_offsets = np.empty(frontier_buffer_size, dtype=np.int32)
        self.frontier_cluster_sizes = np.empty(frontier_buffer_size, dtype=np.int32)

        self.frontiers = []
        self.dormant_frontiers = []
        self.tmp_frontiers = []
        self.removed_ids = []
        self.first_new_frontier = None
        self.next_frontier = None

        self.cluster_min = int(cluster_min)
        self.cluster_size_xy = float(cluster_size_xy)
        self.cluster_size_z = float(cluster_size_z)
        self.down_sample = int(down_sample)

        self.min_candidate_dist = float(min_candidate_dist)
        self.min_candidate_clearance = float(min_candidate_clearance)
        self.candidate_dphi = float(candidate_dphi)
        self.candidate_rmin = float(candidate_rmin)
        self.candidate_rmax = float(candidate_rmax)
        self.candidate_rnum = int(candidate_rnum)
        self.min_visib_num = int(min_visib_num)
        self.min_view_finish_fraction = float(min_view_finish_fraction)
        self.inflate = np.asarray(inflate, dtype=float)
        self.horizontal_fov = float(horizontal_fov)
        self.vertical_fov = float(vertical_fov)

        self.resolution = self.map.resolution
        self.max_vel = float(max_vel)
        self.max_yaw_rate = float(max_yaw_rate)
        self.direction_weight = float(direction_weight)
        self.refined_num = int(refined_num)
        self.refined_radius = float(refined_radius)
        self.top_view_num = int(top_view_num)
        self.max_decay = float(max_decay)
        self.global_astar_resolution = float(global_astar_resolution)
        self.global_astar_max_time = float(global_astar_max_time)
        self.global_astar_fail_cost = float(global_astar_fail_cost)
        self.lkh_command = lkh_command
        self.tsp_dir = tsp_dir
        self.debug_viewpoint_sampling = bool(debug_viewpoint_sampling)
        self.debug_step = None
    def search_frontiers(self): # NEW: FUEL-style incremental frontier search
        # finds new frontiers. and this is all voxels within the update range that are free and neighbor unknown
        self.tmp_frontiers.clear()

        update_min = np.asarray(self.map.local_bound_min, dtype=int) # defines the box of voxels that changed with latest update
        update_max = np.asarray(self.map.local_bound_max, dtype=int)

        self.removed_ids.clear()
        kept_frontiers = []
        for frontier_id, frontier in enumerate(self.frontiers):
            if self.have_overlap(frontier.index_min, frontier.index_max, update_min, update_max):
                if self.is_frontier_changed(frontier):
                    self.reset_frontier_flag(frontier)
                    self.removed_ids.append(frontier_id)
                    continue
            kept_frontiers.append(frontier)

        self.frontiers = kept_frontiers

        kept_dormant = []
        for frontier in self.dormant_frontiers:
            if self.have_overlap(frontier.index_min, frontier.index_max, update_min, update_max):
                if self.is_frontier_changed(frontier):
                    self.reset_frontier_flag(frontier)
                    continue
            kept_dormant.append(frontier)

        self.dormant_frontiers = kept_dormant

        inflate_voxels = np.ceil(self.inflate / self.map.resolution).astype(int) # used to increase bounding box to update more voxels

        search_min = np.maximum(update_min - inflate_voxels, 0)
        search_max = np.minimum(update_max + inflate_voxels, np.asarray(self.map.shape, dtype=int) - 1)

        if NUMBA_AVAILABLE:
            candidate_count = _frontier_candidate_indices_numba( # number of frontiers in the range
                self.map.grid,
                self.frontier_flag,
                search_min.astype(np.int32),
                search_max.astype(np.int32),
                self.frontier_candidate_buffer,
            )
            cluster_count = _expand_frontiers_numba(
                self.map.grid,
                self.frontier_flag,
                self.frontier_candidate_buffer,
                int(candidate_count),
                self.frontier_queue_buffer,
                self.frontier_cluster_indices_buffer,
                self.frontier_cluster_offsets,
                self.frontier_cluster_sizes,
                float(self.map.resolution),
                float(self.map.x_min),
                float(self.map.y_min),
                float(self.map.z_min),
                int(self.cluster_min),
            )
            for cluster_index in range(cluster_count):
                start = int(self.frontier_cluster_offsets[cluster_index])
                size = int(self.frontier_cluster_sizes[cluster_index])
                indices = self.frontier_cluster_indices_buffer[start:start + size].copy()
                frontier = Frontier(
                    indices=indices,
                    cells=self.map.grid_to_map(indices),
                )
                self.compute_frontier_info(frontier)
                self.tmp_frontiers.append(frontier)
        else:
            for ix in range(search_min[0], search_max[0] + 1):
                for iy in range(search_min[1], search_max[1] + 1):
                    for iz in range(search_min[2], search_max[2] + 1):
                        if self.frontier_flag[ix, iy, iz] != 0:
                            continue

                        index = np.array([ix, iy, iz], dtype=int)

                        if self.known_free(index) and self.is_neighbor_unknown(index):
                            self.expand_frontier(index)

        self.split_large_frontiers(self.tmp_frontiers)

        return self.tmp_frontiers

    def expand_frontier(self, first): # NEW: grow one frontier seed into a connected cluster
        # Creates clusters of the newly found frontiers

        first = np.asarray(first, dtype=int).reshape(3)

        cell_queue = deque()
        expanded_indices = []
        expanded_points = []

        first_point = self.map.grid_to_map(first.reshape(1, 3))[0]
        expanded_indices.append(first.copy())
        expanded_points.append(first_point)
        cell_queue.append(first.copy())
        self.frontier_flag[first[0], first[1], first[2]] = 1

        while cell_queue:
            cur = cell_queue.popleft()

            for nbr in self.all_neighbors(cur):
                nbr = np.asarray(nbr, dtype=int).reshape(3)

                if not self.map.in_bounds_indices(nbr.reshape(1, 3))[0]:
                    continue

                ix, iy, iz = nbr

                if self.frontier_flag[ix, iy, iz] == 1:
                    continue

                if not (self.known_free(nbr) and self.is_neighbor_unknown(nbr)):
                    continue

                pos = self.map.grid_to_map(nbr.reshape(1, 3))[0]

                if pos[2] < 0.4:
                    continue

                expanded_indices.append(nbr.copy())
                expanded_points.append(pos)
                cell_queue.append(nbr.copy())
                self.frontier_flag[ix, iy, iz] = 1

        if len(expanded_points) > self.cluster_min:
            frontier = Frontier(
                indices=np.asarray(expanded_indices, dtype=int),
                cells=np.asarray(expanded_points, dtype=float),
            )
            self.compute_frontier_info(frontier)
            self.tmp_frontiers.append(frontier)

    def split_large_frontiers(self, frontiers): # NEW: split oversized frontier clusters
        output = []

        for frontier in frontiers:
            splits = self.split_horizontally(frontier)

            if splits is not None:
                output.extend(splits)
            else:
                output.append(frontier)

        frontiers[:] = output

    def split_horizontally(self, frontier): # NEW: recursively split a frontier that is too wide in XY
        cells = np.asarray(frontier.cells, dtype=float).reshape((-1, 3))
        indices = np.asarray(frontier.indices, dtype=int).reshape((-1, 3))

        filtered = frontier.filtered_cells # the down sampled original
        if filtered is None or len(filtered) == 0:
            filtered = cells
        filtered = np.asarray(filtered, dtype=float).reshape((-1, 3))

        mean_xy = frontier.average[:2]

        need_split = np.any(np.linalg.norm(filtered[:, :2] - mean_xy, axis=1) > self.cluster_size_xy)
        if not need_split:
            return None

        diffs = filtered[:, :2] - mean_xy
        cov = (diffs.T @ diffs) / max(1, len(filtered))

        values, vectors = np.linalg.eigh(cov)
        first_pc = vectors[:, np.argmax(values)]

        side = (cells[:, :2] - mean_xy) @ first_pc

        mask1 = side >= 0
        mask2 = ~mask1

        if not np.any(mask1) or not np.any(mask2):
            return None

        ftr1 = Frontier(indices=indices[mask1], cells=cells[mask1])
        ftr2 = Frontier(indices=indices[mask2], cells=cells[mask2])

        self.compute_frontier_info(ftr1)
        self.compute_frontier_info(ftr2)

        output = []

        split1 = self.split_horizontally(ftr1)
        if split1 is None:
            output.append(ftr1)
        else:
            output.extend(split1)

        split2 = self.split_horizontally(ftr2)
        if split2 is None:
            output.append(ftr2)
        else:
            output.extend(split2)

        return output

    def is_in_boxes(self, boxes, index): # NEW: check if a voxel index is inside any xyz box
        point = self.map.grid_to_map(np.asarray(index, dtype=int).reshape(1, 3))[0]

        for box_min, box_max in boxes:
            box_min = np.asarray(box_min, dtype=float)
            box_max = np.asarray(box_max, dtype=float)

            if np.all(point > box_min) and np.all(point < box_max):
                return True

        return False

    def merge_frontiers(self, ftr1, ftr2): # NEW: merge ftr2 into ftr1
        ftr1.indices = np.vstack([ftr1.indices, ftr2.indices])
        ftr1.cells = np.vstack([ftr1.cells, ftr2.cells])
        self.compute_frontier_info(ftr1)
        return ftr1

    def can_be_merged(self, ftr1, ftr2): # NEW: check whether merged frontier stays within size limits
        cells = np.vstack([ftr1.cells, ftr2.cells])
        merged_avg = cells.mean(axis=0)

        diffs = cells - merged_avg
        too_wide_xy = np.any(np.linalg.norm(diffs[:, :2], axis=1) > self.cluster_size_xy)
        too_tall_z = np.any(np.abs(diffs[:, 2]) > self.cluster_size_z)

        return not (too_wide_xy or too_tall_z)

    def have_overlap(self, min1, max1, min2, max2): # NEW: check whether two index boxes overlap
        min1 = np.asarray(min1)
        max1 = np.asarray(max1)
        min2 = np.asarray(min2)
        max2 = np.asarray(max2)

        overlap_min = np.maximum(min1, min2)
        overlap_max = np.minimum(max1, max2)

        return np.all(overlap_min <= overlap_max)

    def is_frontier_changed(self, frontier): # NEW: true if any frontier voxel is no longer valid
        for index in frontier.indices:
            if not (self.known_free(index) and self.is_neighbor_unknown(index)):
                return True
        return False

    def reset_frontier_flag(self, frontier): # NEW: clear cluster assignment flags before removing an old frontier
        for index in frontier.indices:
            index = np.asarray(index, dtype=int).reshape(3)
            if self.map.in_bounds_indices(index.reshape(1, 3))[0]:
                self.frontier_flag[tuple(index)] = 0

    def compute_frontier_info(self, frontier): # NEW: compute centroid, boxes, and downsampled cells
        # compute centroid, boxes, and downsampled cells
        frontier.cells = np.asarray(frontier.cells, dtype=float).reshape((-1, 3))
        frontier.indices = np.asarray(frontier.indices, dtype=int).reshape((-1, 3))

        frontier.average = frontier.cells.mean(axis=0)

        frontier.box_min = frontier.cells.min(axis=0)
        frontier.box_max = frontier.cells.max(axis=0)

        frontier.index_min = frontier.indices.min(axis=0)
        frontier.index_max = frontier.indices.max(axis=0)

        frontier.filtered_cells = self.downsample(frontier.cells)

        return frontier

    def compute_frontiers_to_visit(self):
        self.first_new_frontier = len(self.frontiers)

        new_num = 0
        new_dormant_num = 0
        sampling_debug_total = {
            "frontiers_sampled": 0,
            "candidates": 0,
            "passed": 0,
            "outside_map": 0,
            "occupied": 0,
            "near_unknown": 0,
            "degenerate_yaw": 0,
            "not_enough_visible_cells": 0,
            "fov_fail": 0,
            "ray_blocked_occupied": 0,
            "ray_blocked_unknown": 0,
            "ray_out_of_map": 0,
        }

        for tmp_frontier in self.tmp_frontiers: # newly detected frontiers
            self.sample_viewpoints(tmp_frontier)
            sampling_debug = getattr(tmp_frontier, "viewpoint_debug", None)
            if sampling_debug is not None:
                sampling_debug_total["frontiers_sampled"] += 1
                for key in sampling_debug_total:
                    if key == "frontiers_sampled":
                        continue
                    sampling_debug_total[key] += int(sampling_debug.get(key, 0))

            if len(tmp_frontier.viewpoints) > 0:
                new_num += 1
                tmp_frontier.viewpoints.sort(key=lambda viewpoint: viewpoint.visib_num, reverse=True)
                self.frontiers.append(tmp_frontier)
            else:
                new_dormant_num += 1
                self.dormant_frontiers.append(tmp_frontier)

        for idx, frontier in enumerate(self.frontiers):
            frontier.id = idx

        return {
            "new": new_num,
            "new_dormant": new_dormant_num,
            "active": len(self.frontiers),
            "dormant": len(self.dormant_frontiers),
            "viewpoint_sampling": sampling_debug_total,
        }

    def get_top_viewpoints_info(self, cur_pos): # NEW: get best usable viewpoint for each active frontier
        cur_pos = np.asarray(cur_pos, dtype=float).reshape(3)
        points = []
        yaws = []
        averages = []

        for frontier in self.frontiers:
            if not frontier.viewpoints:
                continue

            chosen = None
            for viewpoint in frontier.viewpoints:
                if np.linalg.norm(viewpoint.pos - cur_pos) < self.min_candidate_dist:
                    continue
                chosen = viewpoint
                break

            if chosen is None:
                chosen = frontier.viewpoints[0]

            points.append(chosen.pos)
            yaws.append(chosen.yaw)
            averages.append(frontier.average)

        return points, yaws, averages

    def get_viewpoints_info(self, cur_pos, ids, view_num, max_decay): # NEW: get several top viewpoints for selected frontier IDs
        cur_pos = np.asarray(cur_pos, dtype=float).reshape(3)
        points = []
        yaws = []

        for frontier_id in ids:
            frontier = next((f for f in self.frontiers if f.id == frontier_id), None)
            if frontier is None or not frontier.viewpoints:
                points.append([])
                yaws.append([])
                continue

            pts = []
            ys = []
            visib_thresh = frontier.viewpoints[0].visib_num * max_decay

            for viewpoint in frontier.viewpoints:
                if len(pts) >= view_num or viewpoint.visib_num <= visib_thresh:
                    break
                if np.linalg.norm(viewpoint.pos - cur_pos) < self.min_candidate_dist:
                    continue
                pts.append(viewpoint.pos)
                ys.append(viewpoint.yaw)

            if not pts:
                for viewpoint in frontier.viewpoints:
                    if len(pts) >= view_num or viewpoint.visib_num <= visib_thresh:
                        break
                    pts.append(viewpoint.pos)
                    ys.append(viewpoint.yaw)

            points.append(pts)
            yaws.append(ys)

        return points, yaws

    def get_frontiers(self): # NEW: return active frontier clusters
        return [frontier.cells for frontier in self.frontiers]

    def get_dormant_frontiers(self): # NEW: return dormant frontier clusters
        return [frontier.cells for frontier in self.dormant_frontiers]

    def get_frontier_boxes(self): # NEW: return frontier box centers and sizes
        boxes = []
        for frontier in self.frontiers:
            center = 0.5 * (frontier.box_max + frontier.box_min)
            scale = frontier.box_max - frontier.box_min
            boxes.append((center, scale))
        return boxes

    def needs_frontier_cost_matrix_update(self):
        frontier_num = len(self.frontiers)
        old_num = self.first_new_frontier
        if old_num is None:
            old_num = frontier_num
        old_num = int(np.clip(old_num, 0, frontier_num))

        if self.removed_ids:
            return True
        if old_num < frontier_num:
            return True
        return any(
            len(frontier.costs) != frontier_num or len(frontier.paths) != frontier_num
            for frontier in self.frontiers
        )
    def update_frontier_cost_matrix(self, timing=None): # NEW: incrementally update pairwise costs/paths between active frontiers
        frontier_num = len(self.frontiers)
        old_num = self.first_new_frontier
        if old_num is None:
            old_num = frontier_num
        old_num = int(np.clip(old_num, 0, frontier_num))

        if self.removed_ids:
            for frontier in self.frontiers[:old_num]:
                for removed_id in sorted(self.removed_ids, reverse=True):
                    if 0 <= removed_id < len(frontier.costs):
                        del frontier.costs[removed_id]
                    if 0 <= removed_id < len(frontier.paths):
                        del frontier.paths[removed_id]
            self.removed_ids.clear()

        cache_valid = all(
            len(frontier.costs) == old_num and len(frontier.paths) == old_num
            for frontier in self.frontiers[:old_num]
        )
        reused_pairs = old_num * max(0, old_num - 1) if cache_valid else 0
        if not cache_valid:
            old_num = 0
            for frontier in self.frontiers:
                frontier.costs = []
                frontier.paths = []

        for frontier in self.frontiers[old_num:]:
            frontier.costs = []
            frontier.paths = []

        def compute_pair(i, j):
            frontier_i = self.frontiers[i]
            frontier_j = self.frontiers[j]
            if not frontier_i.viewpoints or not frontier_j.viewpoints:
                cost = float("inf")
                path = []
            else:
                viewpoint_i = frontier_i.viewpoints[0]
                viewpoint_j = frontier_j.viewpoints[0]
                cost, path = self.compute_viewpoint_cost(
                    viewpoint_i.pos,
                    viewpoint_i.yaw,
                    viewpoint_j.pos,
                    viewpoint_j.yaw,
                    cur_vel=np.zeros(3, dtype=float),
                    yaw_rate=0.0,
                    timing=timing,
                    need_path=False,
                    allow_astar=False,
                )
            return cost, path

        computed_pairs = 0

        def append_pair(i, j):
            nonlocal computed_pairs
            cost, path = compute_pair(i, j)
            computed_pairs += 1
            self.frontiers[i].costs.append(cost)
            self.frontiers[i].paths.append(path)
            self.frontiers[j].costs.append(cost)
            self.frontiers[j].paths.append([
                np.asarray(point, dtype=float).reshape(3).copy()
                for point in reversed(path)
            ])

        for i in range(old_num):
            for j in range(old_num, frontier_num):
                append_pair(i, j)

        for i in range(old_num, frontier_num):
            for j in range(i, frontier_num):
                if i == j:
                    self.frontiers[i].costs.append(0.0)
                    self.frontiers[i].paths.append([])
                else:
                    append_pair(i, j)

        for idx, frontier in enumerate(self.frontiers):
            frontier.id = idx

        self.first_new_frontier = frontier_num
        if timing is not None:
            timing.add_detail(f"cache reused_directed {reused_pairs}")
            timing.add_detail(f"computed_pairs {computed_pairs}")

    def get_full_cost_matrix(self, cur_pos, cur_yaw=0.0, cur_vel=None, yaw_rate=0.0, timing=None): # NEW: build FUEL-style ATSP matrix with current pose as node 0
        cur_pos = np.asarray(cur_pos, dtype=float).reshape(3)
        cur_yaw = float(cur_yaw)
        cur_vel = np.zeros(3, dtype=float) if cur_vel is None else np.asarray(cur_vel, dtype=float).reshape(3)
        frontier_num = len(self.frontiers)
        mat = np.full((frontier_num + 1, frontier_num + 1), float("inf"), dtype=float)
        mat[:, 0] = 0.0
        mat[0, 0] = 0.0

        for i, frontier in enumerate(self.frontiers, start=1):
            if len(frontier.costs) != frontier_num:
                continue
            mat[i, 1:] = np.asarray(frontier.costs, dtype=float)

        for j, frontier in enumerate(self.frontiers, start=1):
            if not frontier.viewpoints:
                continue
            viewpoint = frontier.viewpoints[0]
            cost, _ = self.compute_viewpoint_cost(
                cur_pos,
                cur_yaw,
                viewpoint.pos,
                viewpoint.yaw,
                cur_vel=cur_vel,
                yaw_rate=yaw_rate,
                timing=timing,
            )
            mat[0, j] = cost

        return mat

    def find_global_tour(self, cur_pos, cur_yaw=0.0, cur_vel=None, yaw_rate=0.0, timing=None): # NEW: FUEL-style coarse frontier route
        cur_pos = np.asarray(cur_pos, dtype=float).reshape(3)

        with timing_section(timing, "frontier_pair_costs") as pair_timing:
            self.update_frontier_cost_matrix(timing=pair_timing)

        with timing_section(timing, "current_to_frontiers") as current_timing:
            cost_matrix = self.get_full_cost_matrix(
                cur_pos,
                cur_yaw,
                cur_vel=cur_vel,
                yaw_rate=yaw_rate,
                timing=current_timing,
            )

        with timing_section(timing, "atsp_solver"):
            if len(self.frontiers) == 0:
                frontier_ids = []
            elif len(self.frontiers) == 1:
                frontier_ids = [0]
            else:
                frontier_ids = self.solve_atsp_lkh(cost_matrix)

        with timing_section(timing, "route_reconstruct"):
            global_tour = self.get_path_for_tour(cur_pos, frontier_ids)

        return {
            "frontier_ids": frontier_ids,
            "cost_matrix": cost_matrix,
            "global_tour": global_tour,
        }

    def select_refinement_ids(self, cur_pos, frontier_ids, refined_num=None, refined_radius=None): # NEW: choose first local route frontiers to refine
        cur_pos = np.asarray(cur_pos, dtype=float).reshape(3)
        refined_num = self.refined_num if refined_num is None else int(refined_num)
        refined_radius = self.refined_radius if refined_radius is None else float(refined_radius)

        selected = []
        for frontier_id in frontier_ids[:max(0, refined_num)]:
            frontier = self.frontiers[int(frontier_id)]
            if not frontier.viewpoints:
                continue
            candidate = frontier.viewpoints[0].pos
            if np.linalg.norm(candidate - cur_pos) > refined_radius and len(selected) >= 2:
                break
            selected.append(int(frontier_id))

        return selected

    def refine_local_tour(
            self,
            cur_pos,
            cur_yaw=0.0,
            cur_vel=None,
            yaw_rate=0.0,
            frontier_ids=None,
            refined_num=None,
            refined_radius=None,
            top_view_num=None,
            max_decay=None,
            timing=None,
    ): # NEW: FUEL-style layered local viewpoint refinement
        cur_pos = np.asarray(cur_pos, dtype=float).reshape(3)
        cur_yaw = float(cur_yaw)
        cur_vel = np.zeros(3, dtype=float) if cur_vel is None else np.asarray(cur_vel, dtype=float).reshape(3)
        top_view_num = self.top_view_num if top_view_num is None else int(top_view_num)
        max_decay = self.max_decay if max_decay is None else float(max_decay)

        if frontier_ids is None:
            route = self.find_global_tour(cur_pos, cur_yaw, cur_vel=cur_vel, yaw_rate=yaw_rate)
            frontier_ids = route["frontier_ids"]
        else:
            frontier_ids = [int(frontier_id) for frontier_id in frontier_ids]

        with timing_section(timing, "select_frontiers"):
            selected_ids = self.select_refinement_ids(
                cur_pos,
                frontier_ids,
                refined_num=refined_num,
                refined_radius=refined_radius,
            )
        if not selected_ids:
            return {"frontier_ids": [], "points": [], "yaws": [], "refined_tour": [cur_pos]}

        with timing_section(timing, "get_viewpoints"):
            point_groups, yaw_groups = self.get_viewpoints_info(cur_pos, selected_ids, top_view_num, max_decay)
        if not point_groups or any(len(group) == 0 for group in point_groups):
            return {"frontier_ids": selected_ids, "points": [], "yaws": [], "refined_tour": [cur_pos]}

        last_costs = [0.0]
        last_nodes = [(cur_pos, cur_yaw, cur_vel)]
        parent_layers = []

        with timing_section(timing, "dp_edge_costs") as edge_timing:
            for layer_idx, (points, yaws) in enumerate(zip(point_groups, yaw_groups)):
                if layer_idx == len(point_groups) - 1:
                    points = points[:1]
                    yaws = yaws[:1]

                cur_costs = [float("inf")] * len(points)
                cur_parents = [-1] * len(points)

                for j, (point, yaw) in enumerate(zip(points, yaws)):
                    point = np.asarray(point, dtype=float).reshape(3)
                    for i, (prev_point, prev_yaw, prev_vel) in enumerate(last_nodes):
                        edge_cost, _ = self.compute_viewpoint_cost(
                            prev_point,
                            prev_yaw,
                            point,
                            yaw,
                        cur_vel=prev_vel,
                        yaw_rate=0.0,
                        timing=edge_timing,
                    )
                        total_cost = last_costs[i] + edge_cost
                        if total_cost < cur_costs[j]:
                            cur_costs[j] = total_cost
                            cur_parents[j] = i

                parent_layers.append(cur_parents)
                last_costs = cur_costs
                last_nodes = [
                    (
                        np.asarray(point, dtype=float).reshape(3),
                        float(yaw),
                        np.zeros(3, dtype=float),
                    )
                    for point, yaw in zip(points, yaws)
                ]

        best_idx = int(np.argmin(last_costs))
        refined_points = []
        refined_yaws = []
        for layer_idx in range(len(parent_layers) - 1, -1, -1):
            points = point_groups[layer_idx]
            yaws = yaw_groups[layer_idx]
            if layer_idx == len(parent_layers) - 1:
                points = points[:1]
                yaws = yaws[:1]
            refined_points.append(np.asarray(points[best_idx], dtype=float).reshape(3))
            refined_yaws.append(float(yaws[best_idx]))
            best_idx = parent_layers[layer_idx][best_idx]

        refined_points.reverse()
        refined_yaws.reverse()

        with timing_section(timing, "refine_path_build"):
            refined_tour = [cur_pos]
            for point in refined_points:
                self.append_path(refined_tour, self.search_path(refined_tour[-1], point))

        return {
            "frontier_ids": selected_ids,
            "points": refined_points,
            "yaws": refined_yaws,
            "refined_tour": refined_tour,
        }

    def get_path_for_tour(self, cur_pos, frontier_ids): # NEW: reconstruct coarse path from cached A* paths
        cur_pos = np.asarray(cur_pos, dtype=float).reshape(3)
        frontier_ids = [int(frontier_id) for frontier_id in frontier_ids]
        if not frontier_ids:
            return [cur_pos]

        path = []
        first_frontier = self.frontiers[frontier_ids[0]]
        if not first_frontier.viewpoints:
            return [cur_pos]

        self.append_path(path, self.search_path(cur_pos, first_frontier.viewpoints[0].pos))
        for from_id, to_id in zip(frontier_ids[:-1], frontier_ids[1:]):
            segment = self.frontiers[from_id].paths[to_id]
            if not segment:
                from_viewpoint = self.frontiers[from_id].viewpoints[0]
                to_viewpoint = self.frontiers[to_id].viewpoints[0]
                segment = self.search_path(from_viewpoint.pos, to_viewpoint.pos)
            self.append_path(path, segment)

        return path

    def solve_atsp_lkh(self, cost_matrix): # NEW: solve ATSP with an LKH-compatible executable
        try:
            return self.solve_atsp_elkai(cost_matrix)
        except ImportError:
            pass

        lkh_command = self.resolve_lkh_command()
        if lkh_command is None:
            raise RuntimeError(
                "No ATSP solver configured. Install elkai in this Python environment, "
                "pass lkh_command=... to FrontierFinder, or set LKH_COMMAND."
            )

        if self.tsp_dir is None:
            with tempfile.TemporaryDirectory(prefix="fuel_atsp_") as tmp_dir:
                return self.run_lkh_atsp(cost_matrix, tmp_dir, lkh_command)

        os.makedirs(self.tsp_dir, exist_ok=True)
        return self.run_lkh_atsp(cost_matrix, self.tsp_dir, lkh_command)

    def solve_atsp_elkai(self, cost_matrix): # NEW: in-process LKH3-backed Python solver
        import elkai

        int_matrix = self.integer_cost_matrix(cost_matrix)
        route = elkai.DistanceMatrix(int_matrix).solve_tsp()
        return self.solver_route_to_frontier_ids(route, len(int_matrix) - 1)

    @staticmethod
    def integer_cost_matrix(cost_matrix, scale=100): # NEW: convert float costs to LKH-safe integers
        matrix = np.asarray(cost_matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("ATSP cost matrix must be square.")

        finite = matrix[np.isfinite(matrix)]
        large_cost = 1000000 if finite.size == 0 else int(max(1000000, np.max(finite) * scale * 10))
        int_matrix = []
        for row in matrix:
            int_row = []
            for cost in row:
                int_row.append(large_cost if not np.isfinite(cost) else int(cost * scale))
            int_matrix.append(int_row)
        return int_matrix

    @staticmethod
    def solver_route_to_frontier_ids(route, frontier_num): # NEW: convert solver node order to frontier IDs
        route = [int(node) for node in route]
        if 0 in route:
            zero_idx = route.index(0)
            route = route[zero_idx + 1:] + route[:zero_idx]

        frontier_ids = []
        seen = set()
        for node in route:
            if node == 0:
                continue
            frontier_id = node - 1
            if 0 <= frontier_id < frontier_num and frontier_id not in seen:
                frontier_ids.append(frontier_id)
                seen.add(frontier_id)
        return frontier_ids

    def resolve_lkh_command(self): # NEW: locate LKH from explicit config, env, or PATH
        command = self.lkh_command or os.environ.get("LKH_COMMAND")
        if command:
            return command
        return shutil.which("LKH") or shutil.which("lkh") or shutil.which("lkh_tsp")

    def run_lkh_atsp(self, cost_matrix, tsp_dir, lkh_command): # NEW: write files, run solver, parse result
        problem_path = os.path.join(tsp_dir, "single.tsp")
        parameter_path = os.path.join(tsp_dir, "single.par")
        output_path = os.path.join(tsp_dir, "single.txt")

        self.write_atsp_problem(cost_matrix, problem_path)
        with open(parameter_path, "w", encoding="ascii") as par_file:
            par_file.write(f"PROBLEM_FILE = {problem_path}\n")
            par_file.write("RUNS = 1\n")
            par_file.write(f"OUTPUT_TOUR_FILE = {output_path}\n")

        command = shlex.split(lkh_command, posix=(os.name != "nt")) + [parameter_path]
        result = subprocess.run(command, cwd=tsp_dir, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                "LKH solver failed with exit code "
                f"{result.returncode}.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )

        return self.read_lkh_tour(output_path, len(cost_matrix) - 1)

    @staticmethod
    def write_atsp_problem(cost_matrix, problem_path, scale=100): # NEW: TSPLIB FULL_MATRIX writer
        int_matrix = FrontierFinder.integer_cost_matrix(cost_matrix, scale=scale)

        with open(problem_path, "w", encoding="ascii") as prob_file:
            prob_file.write("NAME : single\n")
            prob_file.write("TYPE : ATSP\n")
            prob_file.write(f"DIMENSION : {len(int_matrix)}\n")
            prob_file.write("EDGE_WEIGHT_TYPE : EXPLICIT\n")
            prob_file.write("EDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
            prob_file.write("EDGE_WEIGHT_SECTION\n")
            for row in int_matrix:
                prob_file.write(" ".join(str(cost) for cost in row) + "\n")
            prob_file.write("EOF")

    @staticmethod
    def read_lkh_tour(output_path, frontier_num): # NEW: convert LKH node IDs to frontier IDs
        if not os.path.exists(output_path):
            raise RuntimeError(f"LKH did not write expected tour file: {output_path}")

        in_tour = False
        frontier_ids = []
        with open(output_path, "r", encoding="ascii", errors="ignore") as tour_file:
            for raw_line in tour_file:
                line = raw_line.strip()
                if line == "TOUR_SECTION":
                    in_tour = True
                    continue
                if not in_tour:
                    continue
                if line == "-1" or line == "EOF":
                    break
                try:
                    solver_id = int(line)
                except ValueError:
                    continue
                if solver_id == 1:
                    continue
                frontier_id = solver_id - 2
                if 0 <= frontier_id < frontier_num:
                    frontier_ids.append(frontier_id)

        return frontier_ids

    def compute_viewpoint_cost(
            self,
            p1,
            yaw1,
            p2,
            yaw2,
            cur_vel=None,
            yaw_rate=0.0,
            timing=None,
            need_path=True,
            allow_astar=True,
    ): # NEW: FUEL-style viewpoint transition cost
        t_path = time.perf_counter()
        if need_path:
            path, path_distance = self.search_path_cost(p1, p2)
        else:
            p1_arr = np.asarray(p1, dtype=float).reshape(3)
            p2_arr = np.asarray(p2, dtype=float).reshape(3)
            euclidean_distance = float(np.linalg.norm(p2_arr - p1_arr))
            if allow_astar:
                _, path_distance = self.search_path_cost(p1_arr, p2_arr)
            else:
                # Global frontier ordering only needs a scalar edge weight. Avoid A* and
                # path storage here; the chosen tour/reference is validated later.
                path_distance = euclidean_distance if self.is_segment_clear(p1_arr, p2_arr) else 2.0 * euclidean_distance
            path = []
        if timing is not None:
            timing.add_elapsed("path_search", time.perf_counter() - t_path)

        t_score = time.perf_counter()
        pos_cost = path_distance / max(self.max_vel, 1e-6)

        cur_vel = np.zeros(3, dtype=float) if cur_vel is None else np.asarray(cur_vel, dtype=float).reshape(3)
        if np.linalg.norm(cur_vel) > 1e-3:
            direction = np.asarray(p2, dtype=float).reshape(3) - np.asarray(p1, dtype=float).reshape(3)
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 1e-6:
                direction = direction / direction_norm
                vel_dir = cur_vel / np.linalg.norm(cur_vel)
                heading_diff = np.arccos(float(np.clip(np.dot(vel_dir, direction), -1.0, 1.0)))
                pos_cost += self.direction_weight * heading_diff

        yaw_cost = abs(self.wrap_yaw(float(yaw2) - float(yaw1))) / max(self.max_yaw_rate, 1e-6)
        cost = max(pos_cost, yaw_cost)
        if timing is not None:
            timing.add_elapsed("cost_scoring", time.perf_counter() - t_score)
        return cost, path

    def search_path(self, p1, p2): # NEW: straight-line connection first, grid A* fallback
        path, _ = self.search_path_cost(p1, p2)
        return path

    def search_path_cost(self, p1, p2):
        p1 = np.asarray(p1, dtype=float).reshape(3)
        p2 = np.asarray(p2, dtype=float).reshape(3)

        if self.is_segment_clear(p1, p2):
            return [p1, p2], float(np.linalg.norm(p2 - p1))

        astar_path = self.astar_path(p1, p2)
        if astar_path is not None:
            return astar_path, self.path_length(astar_path)

        return [p1, p2], self.global_astar_fail_cost

    @staticmethod
    def path_length(path): # NEW: length of a polyline path
        if path is None or len(path) < 2:
            return 0.0
        pts = np.asarray(path, dtype=float).reshape((-1, 3))
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())

    def is_segment_clear(self, p1, p2): # NEW: true if segment stays in known free space
        if not self.map.is_in_map(np.vstack([p1, p2])).all():
            return False

        for point in (p1, p2):
            idx = self.map.map_to_grid(point.reshape(1, 3))[0]
            if not self.is_traversable_index(idx):
                return False

        for idx in self.map.raycast_indices(p1, p2):
            if not self.is_traversable_index(idx):
                return False

        return True

    def astar_path(self, p1, p2, max_expansions=20000): # NEW: coarse FUEL-style A* through known free space
        p1 = np.asarray(p1, dtype=float).reshape(3)
        p2 = np.asarray(p2, dtype=float).reshape(3)
        resolution = max(self.global_astar_resolution, self.map.resolution)
        origin = np.array([self.map.x_min, self.map.y_min, self.map.z_min], dtype=float)
        t_start = time.perf_counter()

        def point_to_key(point):
            return tuple(np.floor((np.asarray(point, dtype=float).reshape(3) - origin) / resolution).astype(int))

        def key_to_point(key):
            return origin + (np.asarray(key, dtype=float) + 0.5) * resolution

        start = point_to_key(p1)
        goal = point_to_key(p2)

        if not self.map.is_in_map(np.vstack([p1, p2])).all():
            return None
        if not self.is_traversable_index(self.map.map_to_grid(p1.reshape(1, 3))[0]):
            return None
        if not self.is_traversable_index(self.map.map_to_grid(p2.reshape(1, 3))[0]):
            return None

        def heuristic(key):
            return float(np.linalg.norm((np.asarray(key, dtype=float) - np.asarray(goal, dtype=float)) * resolution))

        neighbor_offsets = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]

        open_heap = [(heuristic(start), 0.0, start)]
        parents = {start: None}
        g_score = {start: 0.0}
        closed = set()

        expansions = 0
        while open_heap and expansions < max_expansions:
            if time.perf_counter() - t_start > self.global_astar_max_time:
                return None
            _, current_g, current_key = heapq.heappop(open_heap)
            if current_key in closed:
                continue
            if current_key == goal:
                return self.reconstruct_coarse_path(parents, current_key, key_to_point, p1, p2)

            closed.add(current_key)
            expansions += 1
            current = np.asarray(current_key, dtype=int)
            current_pos = p1 if current_key == start else key_to_point(current_key)

            for offset in neighbor_offsets:
                neighbor = current + np.asarray(offset, dtype=int)
                neighbor_key = tuple(neighbor)
                if neighbor_key in closed:
                    continue
                neighbor_pos = p2 if neighbor_key == goal else key_to_point(neighbor_key)
                if not self.map.is_in_map(neighbor_pos.reshape(1, 3))[0]:
                    continue
                neighbor_grid = self.map.map_to_grid(neighbor_pos.reshape(1, 3))[0]
                if not self.is_traversable_index(neighbor_grid):
                    continue
                if not self.is_segment_clear(current_pos, neighbor_pos):
                    continue

                step_cost = float(np.linalg.norm(np.asarray(offset, dtype=float) * resolution))

                tentative_g = current_g + step_cost
                if tentative_g >= g_score.get(neighbor_key, float("inf")):
                    continue

                parents[neighbor_key] = current_key
                g_score[neighbor_key] = tentative_g
                heapq.heappush(open_heap, (tentative_g + heuristic(neighbor), tentative_g, neighbor_key))

        return None

    def reconstruct_coarse_path(self, parents, goal_key, key_to_point, p1, p2):
        keys = []
        key = goal_key
        while key is not None:
            keys.append(key)
            key = parents[key]
        keys.reverse()

        if len(keys) <= 2:
            return [p1, p2]

        midpoints = [key_to_point(key) for key in keys[1:-1]]
        return [p1, *midpoints, p2]

    def is_traversable_index(self, index): # NEW: known free voxel check for paths
        index = np.asarray(index, dtype=int).reshape(3)
        if not self.map.in_bounds_indices(index.reshape(1, 3))[0]:
            return False
        return self.map.grid[tuple(index)] == self.map.FREE

    def reconstruct_grid_path(self, parents, goal_key, p1, p2): # NEW: convert searched voxel chain to map-frame points
        keys = []
        key = goal_key
        while key is not None:
            keys.append(key)
            key = parents[key]
        keys.reverse()

        if len(keys) <= 2:
            return [p1, p2]

        midpoints = self.map.grid_to_map(np.asarray(keys[1:-1], dtype=int))
        return [p1, *midpoints, p2]

    @staticmethod
    def append_path(path, segment): # NEW: append segment without duplicating the joint point
        for point in segment:
            point = np.asarray(point, dtype=float).reshape(3)
            if path and np.linalg.norm(path[-1] - point) < 1e-6:
                continue
            path.append(point)

    def choose_next_viewpoint(self, drone_pos, distance_weight=1.0, visibility_weight=0.05): # NEW: choose one target viewpoint across all active frontiers
        drone_pos = np.asarray(drone_pos, dtype=float).reshape(3)
        best_choice = None

        for frontier in self.frontiers:
            if not frontier.viewpoints:
                continue

            viewpoint = frontier.viewpoints[0]
            travel_distance = float(np.linalg.norm(viewpoint.pos - drone_pos))
            visibility_score = float(viewpoint.visib_num)
            total_cost = distance_weight * travel_distance - visibility_weight * visibility_score

            choice = {
                "frontier": frontier,
                "viewpoint": viewpoint,
                "target_pos": viewpoint.pos,
                "target_yaw": viewpoint.yaw,
                "visibility": visibility_score,
                "travel_distance": travel_distance,
                "total_cost": total_cost,
            }

            if best_choice is None or choice["total_cost"] < best_choice["total_cost"]:
                best_choice = choice

        return best_choice

    def sample_viewpoints(self, frontier): # NEW: sample candidate viewpoints around a frontier cluster
        frontier.viewpoints = []
        frontier.viewpoint_debug = {
            "candidates": 0,
            "passed": 0,
            "outside_map": 0,
            "occupied": 0,
            "near_unknown": 0,
            "degenerate_yaw": 0,
            "not_enough_visible_cells": 0,
            "fov_fail": 0,
            "ray_blocked_occupied": 0,
            "ray_blocked_unknown": 0,
            "ray_out_of_map": 0,
        }
        if frontier.filtered_cells is None or len(frontier.filtered_cells) == 0:
            return

        if self.candidate_rnum <= 0:
            radii = [self.candidate_rmin]
        else:
            radii = np.linspace(self.candidate_rmin, self.candidate_rmax, self.candidate_rnum + 1)

        for radius in radii:
            phi = -np.pi
            while phi < np.pi:
                sample_pos = frontier.average + radius * np.array([np.cos(phi), np.sin(phi), 0.0]) # sweep about centroid
                frontier.viewpoint_debug["candidates"] += 1

                def count_fail(reason, extra=None):
                    frontier.viewpoint_debug[reason] = frontier.viewpoint_debug.get(reason, 0) + 1
                    if extra:
                        for key, value in extra.items():
                            frontier.viewpoint_debug[key] = frontier.viewpoint_debug.get(key, 0) + int(value)

                if not self.map.is_in_map(sample_pos.reshape(1, 3))[0]:
                    count_fail("outside_map")
                    phi += self.candidate_dphi
                    continue

                sample_idx = self.map.map_to_grid(sample_pos.reshape(1, 3))[0]
                if not self.map.in_bounds_indices(sample_idx.reshape(1, 3))[0]:
                    count_fail("outside_map")
                    phi += self.candidate_dphi
                    continue
                if self.map.grid[tuple(sample_idx)] == self.map.OCCUPIED:
                    count_fail("occupied")
                    phi += self.candidate_dphi
                    continue

                # Temporarily allow samples near unknown space while debugging mapper/frontier behavior.
                # if self.is_near_unknown(sample_pos):
                #     count_fail("near_unknown")
                #     phi += self.candidate_dphi
                #     continue

                cells = frontier.filtered_cells
                ref_dir = cells[0] - sample_pos
                ref_yaw = np.arctan2(ref_dir[1], ref_dir[0])

                yaw_offsets = []
                ref_norm = np.linalg.norm(ref_dir[:2])
                if ref_norm < 1e-6:
                    count_fail("degenerate_yaw")
                    phi += self.candidate_dphi
                    continue

                ref_unit = ref_dir[:2] / ref_norm
                for cell in cells[1:]:
                    direction = cell - sample_pos
                    direction_norm = np.linalg.norm(direction[:2])
                    if direction_norm < 1e-6:
                        continue
                    direction_unit = direction[:2] / direction_norm
                    dot = float(np.clip(np.dot(direction_unit, ref_unit), -1.0, 1.0))
                    yaw = np.arccos(dot)
                    cross_z = ref_unit[0] * direction_unit[1] - ref_unit[1] * direction_unit[0]
                    if cross_z < 0:
                        yaw = -yaw
                    yaw_offsets.append(yaw)

                avg_yaw = ref_yaw if not yaw_offsets else ref_yaw + float(np.mean(yaw_offsets))
                avg_yaw = self.wrap_yaw(avg_yaw)

                if self.debug_viewpoint_sampling:
                    visib_num, visib_stats = self.count_visible_cells_debug(sample_pos, avg_yaw, cells)
                else:
                    visib_num = self.count_visible_cells(sample_pos, avg_yaw, cells)
                    visib_stats = {}
                if visib_num > self.min_visib_num:
                    frontier.viewpoints.append(Viewpoint(sample_pos, avg_yaw, visib_num))
                    frontier.viewpoint_debug["passed"] += 1
                else:
                    count_fail(
                        "not_enough_visible_cells",
                        {
                            **visib_stats,
                        },
                    )

                phi += self.candidate_dphi

        if self.debug_viewpoint_sampling:
            passed = int(frontier.viewpoint_debug["passed"])
            candidates = int(frontier.viewpoint_debug["candidates"])
            fail = candidates - passed
            fail_parts = [
                f"fov={int(frontier.viewpoint_debug['not_enough_visible_cells'])}",
                f"out={int(frontier.viewpoint_debug['outside_map'])}",
                f"occ={int(frontier.viewpoint_debug['occupied'])}",
                f"near_unknown={int(frontier.viewpoint_debug['near_unknown'])}",
                f"deg_yaw={int(frontier.viewpoint_debug['degenerate_yaw'])}",
            ]
            ray_parts = [
                f"fov_cells={int(frontier.viewpoint_debug['fov_fail'])}",
                f"occ_ray={int(frontier.viewpoint_debug['ray_blocked_occupied'])}",
                f"unknown_ray={int(frontier.viewpoint_debug['ray_blocked_unknown'])}",
                f"out_ray={int(frontier.viewpoint_debug['ray_out_of_map'])}",
            ]
            print(
                "[vp] "
                f"step={self.debug_step} "
                f"avg={np.round(frontier.average, 2).tolist()} "
                f"cells={int(len(frontier.cells))}/{int(len(frontier.filtered_cells))} "
                f"pass={passed}/{candidates} fail={fail} "
                f"fail_reasons=({', '.join(fail_parts)}) "
                f"visibility=({', '.join(ray_parts)})",
                flush=True,
            )

    def is_frontier_covered(self): # NEW: check whether enough active frontier cells changed
        update_min = np.asarray(self.map.local_bound_min, dtype=int)
        update_max = np.asarray(self.map.local_bound_max, dtype=int)

        for frontier_group in (self.frontiers, self.dormant_frontiers):
            for frontier in frontier_group:
                if not self.have_overlap(frontier.index_min, frontier.index_max, update_min, update_max):
                    continue

                change_thresh = int(self.min_view_finish_fraction * len(frontier.indices))
                change_thresh = max(1, change_thresh)
                change_num = 0

                for index in frontier.indices:
                    if not (self.known_free(index) and self.is_neighbor_unknown(index)):
                        change_num += 1
                        if change_num >= change_thresh:
                            return True

        return False

    def is_near_unknown(self, pos): # NEW: check if a candidate viewpoint is too close to unknown space
        pos = np.asarray(pos, dtype=float).reshape(3)
        center = self.map.map_to_grid(pos.reshape(1, 3))[0]
        vox_num = int(np.floor(self.min_candidate_clearance / self.resolution))

        for dx in range(-vox_num, vox_num + 1):
            for dy in range(-vox_num, vox_num + 1):
                for dz in range(-1, 2):
                    idx = center + np.array([dx, dy, dz], dtype=int)
                    if not self.map.in_bounds_indices(idx.reshape(1, 3))[0]:
                        continue
                    if self.map.grid[tuple(idx)] == self.map.UNKNOWN:
                        return True

        return False

    def count_visible_cells(self, pos, yaw, cluster): # NEW: count frontier cells visible from a candidate viewpoint
        pos = np.asarray(pos, dtype=float).reshape(3)
        cluster = np.asarray(cluster, dtype=float).reshape((-1, 3))
        if NUMBA_AVAILABLE:
            return int(_count_visible_cells_numba(
                self.map.grid,
                pos,
                float(yaw),
                cluster,
                float(self.map.resolution),
                float(self.map.x_min),
                float(self.map.y_min),
                float(self.map.z_min),
                float(self.horizontal_fov),
                float(self.vertical_fov),
            ))

        visib_num = 0

        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=float)

        for cell in cluster:
            direction = cell - pos
            distance = np.linalg.norm(direction)
            if distance < 1e-6:
                continue

            horizontal_distance = np.linalg.norm(direction[:2])
            if horizontal_distance < 1e-6:
                continue

            horizontal_dir = np.array([direction[0], direction[1], 0.0], dtype=float) / horizontal_distance
            horizontal_angle = np.arccos(float(np.clip(np.dot(horizontal_dir, forward), -1.0, 1.0)))
            vertical_angle = np.arctan2(direction[2], horizontal_distance)

            if horizontal_angle > 0.5 * self.horizontal_fov:
                continue
            if abs(vertical_angle) > 0.5 * self.vertical_fov:
                continue

            visible = True
            ray_indices = self.map.raycast_indices(pos, cell)
            for idx in ray_indices:
                if not self.map.in_bounds_indices(idx.reshape(1, 3))[0]:
                    visible = False
                    break

                state = self.map.grid[tuple(idx)]
                if state == self.map.OCCUPIED or state == self.map.UNKNOWN:
                    visible = False
                    break

            if visible:
                visib_num += 1

        return visib_num

    def count_visible_cells_debug(self, pos, yaw, cluster): # NEW: count visible cells and explain rejected cells
        pos = np.asarray(pos, dtype=float).reshape(3)
        cluster = np.asarray(cluster, dtype=float).reshape((-1, 3))
        stats = {
            "fov_fail": 0,
            "ray_blocked_occupied": 0,
            "ray_blocked_unknown": 0,
            "ray_out_of_map": 0,
        }
        visib_num = 0

        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=float)

        for cell in cluster:
            direction = cell - pos
            distance = np.linalg.norm(direction)
            if distance < 1e-6:
                stats["fov_fail"] += 1
                continue

            horizontal_distance = np.linalg.norm(direction[:2])
            if horizontal_distance < 1e-6:
                stats["fov_fail"] += 1
                continue

            horizontal_dir = np.array([direction[0], direction[1], 0.0], dtype=float) / horizontal_distance
            horizontal_angle = np.arccos(float(np.clip(np.dot(horizontal_dir, forward), -1.0, 1.0)))
            vertical_angle = np.arctan2(direction[2], horizontal_distance)

            if horizontal_angle > 0.5 * self.horizontal_fov or abs(vertical_angle) > 0.5 * self.vertical_fov:
                stats["fov_fail"] += 1
                continue

            visible = True
            ray_indices = self.map.raycast_indices(pos, cell)
            for idx in ray_indices:
                if not self.map.in_bounds_indices(idx.reshape(1, 3))[0]:
                    stats["ray_out_of_map"] += 1
                    visible = False
                    break

                state = self.map.grid[tuple(idx)]
                if state == self.map.OCCUPIED:
                    stats["ray_blocked_occupied"] += 1
                    visible = False
                    break
                if state == self.map.UNKNOWN:
                    stats["ray_blocked_unknown"] += 1
                    visible = False
                    break

            if visible:
                visib_num += 1

        return visib_num, stats

    def downsample(self, cells): # NEW: voxel-grid downsample frontier cell positions
        cells = np.asarray(cells, dtype=float).reshape((-1, 3))

        if len(cells) == 0:
            return cells

        leaf_size = self.resolution * self.down_sample
        keys = np.floor(cells / leaf_size).astype(int)

        _, unique_indices = np.unique(keys, axis=0, return_index=True)

        return cells[np.sort(unique_indices)]

    def wrap_yaw(self, yaw): # NEW: wrap yaw to [-pi, pi]
        while yaw < -np.pi:
            yaw += 2.0 * np.pi
        while yaw > np.pi:
            yaw -= 2.0 * np.pi
        return yaw

    def search_clear_voxel(self, index): # NEW: find nearest known-free voxel from a seed
        index = np.asarray(index, dtype=int).reshape(3)
        queue = deque([index])
        visited = set()

        while queue:
            cur = queue.popleft()
            key = tuple(cur)
            if key in visited:
                continue
            visited.add(key)

            if not self.map.in_bounds_indices(cur.reshape(1, 3))[0]:
                continue

            if self.known_free(cur):
                return cur

            for nbr in self.six_neighbors(cur):
                queue.append(nbr)

        return index

    def six_neighbors(self, index): # NEW: return 6 axis-aligned neighbors
        index = np.asarray(index, dtype=int).reshape(3)
        offsets = np.array([
            [-1, 0, 0],
            [1, 0, 0],
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, -1],
            [0, 0, 1],
        ], dtype=int)
        return [index + offset for offset in offsets]

    def ten_neighbors(self, index): # NEW: return FUEL-style 8 horizontal plus vertical neighbors
        index = np.asarray(index, dtype=int).reshape(3)
        neighbors = []

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                neighbors.append(index + np.array([dx, dy, 0], dtype=int))

        neighbors.append(index + np.array([0, 0, -1], dtype=int))
        neighbors.append(index + np.array([0, 0, 1], dtype=int))
        return neighbors

    def all_neighbors(self, index): # NEW: return all 26 neighbors
        index = np.asarray(index, dtype=int).reshape(3)
        neighbors = []

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbors.append(index + np.array([dx, dy, dz], dtype=int))

        return neighbors

    def is_neighbor_unknown(self, index): # NEW: true if any 6-neighbor is unknown
        for nbr in self.six_neighbors(index):
            if not self.map.in_bounds_indices(nbr.reshape(1, 3))[0]:
                continue
            if self.map.grid[tuple(nbr)] == self.map.UNKNOWN:
                return True
        return False

    def known_free(self, index): # NEW: true if voxel is currently known free
        index = np.asarray(index, dtype=int).reshape(3)
        if not self.map.in_bounds_indices(index.reshape(1, 3))[0]:
            return False
        return self.map.grid[tuple(index)] == self.map.FREE

    def in_map(self, index): # NEW: true if index is inside occupancy map bounds
        index = np.asarray(index, dtype=int).reshape(3)
        return self.map.in_bounds_indices(index.reshape(1, 3))[0]
