from enum import Enum, auto
import time


class ExplorationState(Enum):
    INIT = auto()
    WARMUP = auto()
    EXPLORE_PLAN = auto()
    EXPLORE_EXEC = auto()
    TARGET_INSPECT = auto()
    FINISH = auto()


class ExplorationFSM:
    def __init__(
        self,
        *,
        mapper,
        frontier_finder,
        kino_planner,
        dt,
        map_update_interval,
        plan_update_interval,
        viewer_update_interval,
        debug_log,
    ):
        self.mapper = mapper
        self.frontier_finder = frontier_finder
        self.kino_planner = kino_planner
        self.dt = float(dt)

        self.map_update_interval = int(map_update_interval)
        self.plan_update_interval = int(plan_update_interval)
        self.viewer_update_interval = int(viewer_update_interval)

        self.debug_log = debug_log
        self.state = ExplorationState.INIT
        self.step_count = 0

        self.active_refs = []
        self.ref_index = 0
        self.best_viewpoint = None
        self.best_yaw = None
        self.active_kino_path_world = []
        self.current_target_world = None

        self.exploration_started = False
        self.finished = False
        self.last_map_update_step = None
        self.last_plan_step = None
        self.last_frontier_stats = None
        self.last_map_stats = None
        self.last_viewpoint_sampling = None

    def transition(self, next_state, reason=""):
        if next_state == self.state:
            return
        previous = self.state
        self.state = next_state
        if self.debug_log is not None:
            self.debug_log(
                "fsm_transition",
                {
                    "step": self.step_count,
                    "from": previous.name,
                    "to": next_state.name,
                    "reason": reason,
                },
                verbose=True,
            )

    def should_update_mapping(self, step_count):
        self.step_count = int(step_count)
        return self.map_update_interval > 0 and self.step_count % self.map_update_interval == 0

    def record_plan(self, step_count):
        self.last_plan_step = int(step_count)

    def update_mapping_if_due(self, *, step_count, rgb, depth, camera_K, T_camera_world):
        self.step_count = int(step_count)
        if not self.should_update_mapping(self.step_count):
            return None
        return self.update_mapping(
            step_count=self.step_count,
            rgb=rgb,
            depth=depth,
            camera_K=camera_K,
            T_camera_world=T_camera_world,
        )

    def update_mapping(self, *, step_count, rgb, depth, camera_K, T_camera_world):
        self.step_count = int(step_count)
        mapper = self.mapper
        frontier_finder = self.frontier_finder

        map_t0 = time.perf_counter()

        section_t0 = time.perf_counter()
        points_world, endpoint_hits, camera_pos_world = mapper.depth_to_world_points(
            depth=depth,
            K=camera_K,
            T_camera_world=T_camera_world,
        )
        depth_points_time = time.perf_counter() - section_t0

        section_t0 = time.perf_counter()
        mapper.input_point_cloud_world(points_world, camera_pos_world, endpoint_hits=endpoint_hits) #occupied, free, unknown
        raycast_free_time = time.perf_counter() - section_t0

        mapper.integrate_tsdf(rgb, depth, camera_K, T_camera_world)

        section_t0 = time.perf_counter()
        frontier_finder.search_frontiers()
        frontier_search_time = time.perf_counter() - section_t0

        frontier_finder.debug_step = self.step_count

        section_t0 = time.perf_counter()
        frontier_stats = frontier_finder.compute_frontiers_to_visit()
        frontier_visit_time = time.perf_counter() - section_t0

        section_t0 = time.perf_counter()
        if frontier_finder.needs_frontier_cost_matrix_update():
            frontier_finder.update_frontier_cost_matrix()
        frontier_cost_time = time.perf_counter() - section_t0

        map_total_time = time.perf_counter() - map_t0
        viewpoint_sampling = frontier_stats.pop("viewpoint_sampling", None)
        map_stats = mapper.map_stats()

        timing = {
            "step": self.step_count,
            "total": map_total_time,
            "depth_points": depth_points_time,
            "raycast_free": raycast_free_time,
            "frontier_search": frontier_search_time,
            "frontier_visit": frontier_visit_time,
            "frontier_costs": frontier_cost_time,
        }
        update = {
            "step": self.step_count,
            "known_pct": map_stats["known_pct"],
            "frontiers": frontier_stats,
        }

        if self.debug_log is not None:
            self.debug_log("map_timing", timing)
            self.debug_log("map_update", update)
            self._log_dormant_reason(frontier_stats, viewpoint_sampling)
            if viewpoint_sampling is not None and viewpoint_sampling["frontiers_sampled"] > 0:
                self.debug_log(
                    "vp_sampling",
                    (
                        f"step={self.step_count} "
                        f"frontiers={viewpoint_sampling['frontiers_sampled']} "
                        f"pass={viewpoint_sampling['passed']}/{viewpoint_sampling['candidates']} "
                        f"fail_fov={viewpoint_sampling['not_enough_visible_cells']} "
                        f"fail_out={viewpoint_sampling['outside_map']} "
                        f"fail_occ={viewpoint_sampling['occupied']} "
                        f"fail_near_unknown={viewpoint_sampling['near_unknown']} "
                        f"fail_deg_yaw={viewpoint_sampling['degenerate_yaw']} "
                        f"fov_cells={viewpoint_sampling['fov_fail']} "
                        f"occ_ray={viewpoint_sampling['ray_blocked_occupied']} "
                        f"unknown_ray={viewpoint_sampling['ray_blocked_unknown']} "
                        f"out_ray={viewpoint_sampling['ray_out_of_map']}"
                    ),
                )

        self.last_map_update_step = self.step_count
        self.last_frontier_stats = frontier_stats
        self.last_map_stats = map_stats
        self.last_viewpoint_sampling = viewpoint_sampling
        self.update_finish_state(frontier_stats)
        return {
            "timing": timing,
            "map_stats": map_stats,
            "frontier_stats": frontier_stats,
            "viewpoint_sampling": viewpoint_sampling,
        }


    def _log_dormant_reason(self, frontier_stats, viewpoint_sampling):
        if viewpoint_sampling is None:
            return
        if int(frontier_stats.get("active", 0)) != 0 or int(frontier_stats.get("dormant", 0)) == 0:
            return

        reason_keys = [
            "not_enough_visible_cells",
            "outside_map",
            "occupied",
            "near_unknown",
            "degenerate_yaw",
            "fov_fail",
            "ray_blocked_occupied",
            "ray_blocked_unknown",
            "ray_out_of_map",
        ]
        top_reason = max(reason_keys, key=lambda key: int(viewpoint_sampling.get(key, 0)))
        top_count = int(viewpoint_sampling.get(top_reason, 0))
        self.debug_log(
            "dormant_frontiers",
            {
                "step": self.step_count,
                "dormant": int(frontier_stats.get("dormant", 0)),
                "sampled": int(viewpoint_sampling.get("frontiers_sampled", 0)),
                "passed": int(viewpoint_sampling.get("passed", 0)),
                "candidates": int(viewpoint_sampling.get("candidates", 0)),
                "top_reason": top_reason,
                "top_count": top_count,
            },
        )
    def update_finish_state(self, frontier_stats=None):
        if frontier_stats is None:
            active = len(self.frontier_finder.frontiers)
            dormant = len(self.frontier_finder.dormant_frontiers)
        else:
            active = int(frontier_stats.get("active", 0))
            dormant = int(frontier_stats.get("dormant", 0))
        tmp = len(self.frontier_finder.tmp_frontiers)

        if active == 0 and dormant == 0 and tmp == 0:
            self.finished = True
            self.transition(ExplorationState.FINISH, "no_frontiers_remaining")
            return True
        return False
    def should_request_plan(
        self,
        *,
        step_count,
        active_refs,
        ref_index,
        exploration_started=True,
        target_detected=False,
        collision_risk=False,
    ):
        self.step_count = int(step_count)
        self.active_refs = [] if active_refs is None else active_refs
        self.ref_index = int(ref_index)

        if self.finished:
            return False, ""

        if self.last_plan_step == self.step_count:
            return False, ""

        if not self.active_refs:
            return True, "no_active_refs"

        if self.ref_index >= len(self.active_refs):
            return True, "refs_exhausted"

        return self.should_replan(
            step_count=self.step_count,
            active_refs=self.active_refs,
            ref_index=self.ref_index,
            exploration_started=exploration_started,
            target_detected=target_detected,
            collision_risk=collision_risk,
        )
    def should_replan(
        self,
        *,
        step_count,
        active_refs=None,
        ref_index=None,
        exploration_started=True,
        target_detected=False,
        collision_risk=False,
        near_end_refs=20, # mplay with
        frontier_check_delay_steps=30,
    ):
        self.step_count = int(step_count)
        if self.finished or not exploration_started:
            return False, ""

        if target_detected:
            return True, "target_detected"

        if collision_risk:
            return True, "collision_risk"

        refs = self.active_refs if active_refs is None else active_refs
        index = self.ref_index if ref_index is None else int(ref_index)

        if not refs:
            return True, "no_active_refs"

        remaining_refs = len(refs) - index
        if remaining_refs <= int(near_end_refs):
            return True, "trajectory_near_end"

        if (
            self.plan_update_interval > 0
            and self.last_plan_step is not None
            and self.step_count - self.last_plan_step >= self.plan_update_interval
        ):
            return True, "periodic_replan"

        if self.step_count >= int(frontier_check_delay_steps):
            if self.frontier_finder.is_frontier_covered():
                return True, "frontier_covered"

        return False, ""






