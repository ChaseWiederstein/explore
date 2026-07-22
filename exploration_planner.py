import traceback
import time

import numpy as np

from timing_tree import TimingNode
from trajectory_manager import plan_reference


def map_points_to_world(mapper, points_map):
    points_map = np.asarray(points_map, dtype=float).reshape((-1, 3))
    return mapper.map_origin_world + (mapper.world_to_map_R.T @ points_map.T).T


def map_vector_to_world(mapper, vector_map):
    vector_map = np.asarray(vector_map, dtype=float).reshape(3)
    return mapper.world_to_map_R.T @ vector_map


def world_yaw_to_map(mapper, yaw_world):
    forward_world = np.array([np.cos(yaw_world), np.sin(yaw_world), 0.0], dtype=float)
    forward_map = mapper.world_to_map_R @ forward_world
    return float(np.arctan2(forward_map[1], forward_map[0]))


def map_yaw_to_world(mapper, yaw_map):
    forward_map = np.array([np.cos(yaw_map), np.sin(yaw_map), 0.0], dtype=float)
    forward_world = map_vector_to_world(mapper, forward_map)
    return float(np.arctan2(forward_world[1], forward_world[0]))


def refs_map_to_world(mapper, refs_map):
    refs_world = []
    for sigma, sigmad, sigmadd, sigmaddd in refs_map:
        sigma_world = np.array(sigma, dtype=float)
        sigmad_world = np.array(sigmad, dtype=float)
        sigmadd_world = np.array(sigmadd, dtype=float)
        sigmaddd_world = np.array(sigmaddd, dtype=float)

        sigma_world[:3] = map_points_to_world(mapper, sigma[:3])[0]
        sigma_world[3] = map_yaw_to_world(mapper, sigma[3])
        sigmad_world[:3] = map_vector_to_world(mapper, sigmad[:3])
        sigmadd_world[:3] = map_vector_to_world(mapper, sigmadd[:3])
        sigmaddd_world[:3] = map_vector_to_world(mapper, sigmaddd[:3])
        refs_world.append((sigma_world, sigmad_world, sigmadd_world, sigmaddd_world))

    return refs_world


def plan_exploration_refs(
    *,
    drone,
    mapper,
    frontier_finder,
    kino_planner,
    stage,
    dt,
    step_count,
    debug_log,
):
    timing = TimingNode("TOTAL")
    plan_start_time = time.perf_counter()

    with timing.section("state_read_transform"):
        drone_state = drone.get_state()
        drone_pos_map = mapper.world_to_map(np.asarray(drone_state["pos"], dtype=float).reshape(1, 3))[0]
        drone_vel_map = mapper.world_to_map_R @ np.asarray(drone_state["vel"], dtype=float).reshape(3)
        drone_yaw_map = world_yaw_to_map(mapper, float(drone_state["yaw"]))

    debug_log(
        "plan_start",
        {
            "step": step_count,
            "pos_map": np.round(drone_pos_map, 3).tolist(),
            "vel_map": np.round(drone_vel_map, 3).tolist(),
            "yaw_map_deg": round(float(np.rad2deg(drone_yaw_map)), 1),
            "frontiers": len(frontier_finder.frontiers),
        },
    )

    try:
        with timing.section("global_route") as global_timing:
            coarse = frontier_finder.find_global_tour(
                cur_pos=drone_pos_map,
                cur_yaw=drone_yaw_map,
                cur_vel=drone_vel_map,
                timing=global_timing,
            )
        debug_log(
            "coarse_route",
            {
                "ids": coarse["frontier_ids"],
                "matrix_shape": list(coarse["cost_matrix"].shape),
                "global_tour_pts": len(coarse["global_tour"]),
            },
        )
        with timing.section("local_refine") as refine_timing:
            refined = frontier_finder.refine_local_tour(
                cur_pos=drone_pos_map,
                cur_yaw=drone_yaw_map,
                cur_vel=drone_vel_map,
                frontier_ids=coarse["frontier_ids"],
                timing=refine_timing,
            )
    except RuntimeError as exc:
        timing.elapsed = time.perf_counter() - plan_start_time
        debug_log("exploration_plan_failed", str(exc))
        debug_log("plan_timing_tree", {"step": step_count, "timing": timing})
        return [], None, None, []
    except Exception:
        timing.elapsed = time.perf_counter() - plan_start_time
        debug_log("exploration_plan_exception")
        traceback.print_exc()
        debug_log("plan_timing_tree", {"step": step_count, "timing": timing})
        return [], None, None, []

    if not refined["points"]:
        debug_log(
            "exploration_plan",
            {"choice": None, "reason": "no_refined_viewpoint", "frontier_ids": refined["frontier_ids"]},
        )
        timing.elapsed = time.perf_counter() - plan_start_time
        debug_log("plan_timing_tree", {"step": step_count, "timing": timing})
        return [], None, None, []

    target_pos_map = refined["points"][0]
    target_yaw_map = refined["yaws"][0]
    with timing.section("guide_path"):
        guide_path = frontier_finder.search_path(drone_pos_map, target_pos_map)

    with timing.section("reference") as reference_timing:
        refs_map, plan_msg, kino_path_map = plan_reference(
            planner=kino_planner,
            mapper=mapper,
            stage=stage,
            start_pos=drone_pos_map,
            start_vel=drone_vel_map,
            start_yaw=target_yaw_map,
            goal_pos=target_pos_map,
            dt=dt,
            reference_check_spacing=mapper.resolution,
            guide_path=guide_path,
            timing=reference_timing,
        )
    debug_log(
        "exploration_plan:",
        {
            "step": step_count,
            "pos": np.round(drone_pos_map, 3).tolist(),
            "target": np.round(target_pos_map, 3).tolist(),
            "yaw_deg": round(float(np.rad2deg(target_yaw_map)), 1),
            "frontiers": refined["frontier_ids"],
            "refs": len(refs_map),
            "plan": plan_msg,
        },
    )
    timing.elapsed = time.perf_counter() - plan_start_time
    debug_log("plan_timing_tree", {"step": step_count, "timing": timing})
    if not refs_map:
        kino_path_world = map_points_to_world(
            mapper,
            np.asarray([pos for pos, _ in kino_path_map], dtype=float),
        ) if kino_path_map else []
        return [], target_pos_map, target_yaw_map, kino_path_world

    refs_world = refs_map_to_world(mapper, refs_map)
    kino_path_world = map_points_to_world(
        mapper,
        np.asarray([pos for pos, _ in kino_path_map], dtype=float),
    ) if kino_path_map else []
    debug_log(
        "plan_refs_world",
        {"count": len(refs_world), "first": np.round(refs_world[0][0], 3).tolist() if refs_world else None},
    )
    return refs_world, target_pos_map, target_yaw_map, kino_path_world


def refresh_exploration_refs(
    *,
    drone,
    mapper,
    frontier_finder,
    kino_planner,
    stage,
    dt,
    step_count,
    debug_log,
):
    refs_world, best_viewpoint, best_yaw, kino_path_world = plan_exploration_refs(
        drone=drone,
        mapper=mapper,
        frontier_finder=frontier_finder,
        kino_planner=kino_planner,
        stage=stage,
        dt=dt,
        step_count=step_count,
        debug_log=debug_log,
    )

    target_world = None
    if best_viewpoint is not None:
        target_world = mapper.map_to_world(best_viewpoint)[0]

    return refs_world, best_viewpoint, best_yaw, kino_path_world, target_world
