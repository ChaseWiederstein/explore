from pathlib import Path
import sys
import time
import traceback

import numpy as np
from isaacsim import SimulationApp
import run_logging
from mapper import OccupancyMap
from frontier import FrontierFinder
from FSM import ExplorationFSM
from isaac_utilities import IsaacTrajectoryDebugDraw, update_trajectory_draw
from kino_astar import KinodynamicAStar
from exploration_planner import refresh_exploration_refs
from run_logging import debug_log


ROOT = Path(__file__).resolve().parent
UAM_ROOT = Path(r"C:/robotics/UAM")
sys.path.insert(0, str(UAM_ROOT))

SCENE_PATH = "C:/robotics/explore/river.usd"
DRONE_BODY_PATH = "/World/iris/body"
RGB_CAMERA_PATH = "/World/iris/body/Gemini2/Orbbec_Gemini2/camera_rgb/camera_rgb/Stream_rgb"
DEPTH_CAMERA_PATH = "/World/iris/body/Gemini2/Orbbec_Gemini2/camera_ir_left/camera_left/Stream_depth"

PHYSICS_DT = 1/60
CAMERA_RESOLUTION = (100, 100)
VERBOSE_DEBUG = False
DRAW_TRAJECTORY = True
DRAW_TRAJECTORY_INTERVAL = 5
DRAW_HISTORY_MAX_POINTS = 3000
DRAW_TARGET_MAX_POINTS = 200
LIVE_MAP_VIEWER = True
TSDF_SNAPSHOT_INTERVAL = 100
TSDF_SNAPSHOT_MAX_STEP = 2000
run_logging.VERBOSE_DEBUG = VERBOSE_DEBUG


def wrap_angle(angle):
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def mapper_volume_corners_world(mapper):
    corners_map = np.array([
        [mapper.x_min, mapper.y_min, mapper.z_min],
        [mapper.x_max, mapper.y_min, mapper.z_min],
        [mapper.x_min, mapper.y_max, mapper.z_min],
        [mapper.x_max, mapper.y_max, mapper.z_min],
        [mapper.x_min, mapper.y_min, mapper.z_max],
        [mapper.x_max, mapper.y_min, mapper.z_max],
        [mapper.x_min, mapper.y_max, mapper.z_max],
        [mapper.x_max, mapper.y_max, mapper.z_max],
    ], dtype=float)
    return mapper.map_to_world(corners_map)

def print_ref_swap_debug(step_count, drone_state, old_ref, new_refs):
    if not new_refs:
        return
    new_sigma, new_sigmad, _, _ = new_refs[0]
    drone_pos = np.asarray(drone_state["pos"], dtype=float).reshape(3)
    drone_vel = np.asarray(drone_state["vel"], dtype=float).reshape(3)
    drone_yaw = float(drone_state["yaw"])
    delta_pos = np.asarray(new_sigma[:3], dtype=float) - drone_pos
    delta_vel = np.asarray(new_sigmad[:3], dtype=float) - drone_vel
    delta_yaw = wrap_angle(float(new_sigma[3]) - drone_yaw)
    old_text = "none"
    if old_ref is not None:
        old_sigma, old_sigmad, _, _ = old_ref
        old_text = (
            f"old_pos {np.round(old_sigma[:3], 3).tolist()} "
            f"old_vel {np.round(old_sigmad[:3], 3).tolist()} "
            f"old_yaw {np.rad2deg(float(old_sigma[3])):.1f}"
        )
    print(
        f"REF_SWAP {step_count} | {old_text} | "
        f"drone_pos {np.round(drone_pos, 3).tolist()} "
        f"drone_vel {np.round(drone_vel, 3).tolist()} "
        f"drone_yaw {np.rad2deg(drone_yaw):.1f} | "
        f"new_pos {np.round(new_sigma[:3], 3).tolist()} "
        f"new_vel {np.round(new_sigmad[:3], 3).tolist()} "
        f"new_yaw {np.rad2deg(float(new_sigma[3])):.1f} | "
        f"dpos {np.linalg.norm(delta_pos):.3f} "
        f"dvel {np.linalg.norm(delta_vel):.3f} "
        f"dyaw {np.rad2deg(delta_yaw):.1f}",
        flush=True,
    )


def main():
    simulation_app = SimulationApp({"headless": False})

    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim
    from isaacsim.core.utils.stage import get_current_stage, open_stage
    from isaacsim.sensors.camera import Camera
    from low_level_controller import Drone, LowLevelController

    open_stage(SCENE_PATH)
    world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT)

    drone_body = RigidPrim(prim_paths_expr=DRONE_BODY_PATH, name="drone_body")
    drone = Drone(drone_body, None)
    controller = LowLevelController(drone)
    mapper = OccupancyMap(resolution=0.25, volume_size=(10, 10.0, 3.0), z_min=-.25)
    kino_planner = KinodynamicAStar(mapper, max_acc=.5, max_vel=1.0, tau=0.8, goal_tol=2*mapper.resolution, planning_horizon=7.0, lambda_heu=10.0)
    frontier_finder = FrontierFinder(mapper, max_vel=kino_planner.max_vel, max_yaw_rate=np.deg2rad(60.0), lkh_command=None, debug_viewpoint_sampling=False)
    fsm = ExplorationFSM(
        mapper=mapper,
        frontier_finder=frontier_finder,
        kino_planner=kino_planner,
        dt=PHYSICS_DT,
        map_update_interval=20,
        plan_update_interval=120,
        viewer_update_interval=600,
        debug_log=debug_log,
    )

    mapper.start_viewer()
    mapper.start_tsdf_recorder()

    rgb_camera = Camera(prim_path=RGB_CAMERA_PATH, name="rgb_camera", resolution=CAMERA_RESOLUTION)
    depth_camera = Camera(prim_path=DEPTH_CAMERA_PATH, name="depth_camera", resolution=CAMERA_RESOLUTION)
    trajectory_draw = IsaacTrajectoryDebugDraw() if DRAW_TRAJECTORY else None

    world.reset()
    drone_body.initialize()
    rgb_camera.initialize()
    depth_camera.initialize()
    depth_camera.add_distance_to_image_plane_to_frame()
    debug_log("world_initialized")
    ZERO_VEL = np.zeros((1, 3), dtype=np.float32)
    start_state = drone.get_state()
    mapper.set_map_frame(start_state["pos"], start_state["yaw"]) # NEW: anchor local map at drone start pose
    if trajectory_draw is not None:
        trajectory_draw.set_volume_box(mapper_volume_corners_world(mapper), color=(1.0, 0.0, 0.0, 1.0), size=8.0)
        trajectory_draw.update([], [], [], [])
    camera_K = depth_camera.get_intrinsics_matrix() # NEW: cache camera intrinsics for depth projection

    drone_body.set_linear_velocities(ZERO_VEL)
    drone_body.set_angular_velocities(ZERO_VEL)

    for _ in range(30):
        world.step(render=True)
    debug_log("warmup_done")

    start_state = drone.get_state()
    x0, y0, z0 = float(start_state["pos"][0]), float(start_state["pos"][1]), float(start_state["pos"][2])
    yaw0 = float(start_state["yaw"])
    scan_targets = [
        ("hold", (x0, y0, z0 + 1.0), yaw0, 5.0), # NEW: hold target type
    ]
    active_refs = []
    for target in scan_targets:
        target_type = target[0]
        if target_type == "hold":
            _, pos, yaw, duration_s = target
            start_pos = None
            end_pos = np.asarray(pos, dtype=float)
        else:
            _, start_pos, end_pos, yaw, duration_s = target
            start_pos = np.asarray(start_pos, dtype=float)
            end_pos = np.asarray(end_pos, dtype=float)

        steps = int(round(duration_s / PHYSICS_DT))
        for step_i in range(steps):
            if target_type == "linear":
                alpha = step_i / max(steps - 1, 1)
                pos_cmd = (1.0 - alpha) * start_pos + alpha * end_pos
            else:
                pos_cmd = end_pos
            sigma = np.array([pos_cmd[0], pos_cmd[1], pos_cmd[2], yaw], dtype=float)
            active_refs.append((sigma, np.zeros(4, dtype=float), np.zeros(4, dtype=float), np.zeros(4, dtype=float)))

    best_viewpoint = None
    best_yaw = None
    ref_index = 0
    exploration_started = False
    StepCount = 0
    flown_path_world = []
    target_points_world = []
    active_kino_path_world = []
    tsdf_snapshot_saved = set()

    def make_hold_refs(duration_s=1.0):
        state = drone.get_state()
        sigma = np.array([state["pos"][0], state["pos"][1], state["pos"][2], state["yaw"]], dtype=float)
        refs = [
            (sigma, np.zeros(4, dtype=float), np.zeros(4, dtype=float), np.zeros(4, dtype=float))
            for _ in range(int(round(float(duration_s) / PHYSICS_DT)))
        ]
        return refs, state

    def request_plan(reason):
        nonlocal active_refs, best_viewpoint, best_yaw, ref_index, active_kino_path_world, exploration_started
        exploration_started = True

        if fsm.finished or fsm.update_finish_state():
            active_refs, state = make_hold_refs()
            ref_index = 0
            active_kino_path_world = []
            fsm.active_refs = active_refs
            fsm.ref_index = ref_index
            debug_log("hold_fallback", {"step": StepCount, "refs": len(active_refs), "pos": np.round(state["pos"], 3).tolist()})
            return

        fsm.record_plan(StepCount)
        debug_log("refs_exhausted" if reason == "refs_exhausted" else "replan_request", {"step": StepCount, "reason": reason, "planning": True})
        old_ref = active_refs[min(ref_index - 1, len(active_refs) - 1)] if active_refs and ref_index > 0 else None
        drone_state_before_plan = drone.get_state()
        active_refs, best_viewpoint, best_yaw, active_kino_path_world, target_world = refresh_exploration_refs(
            drone=drone,
            mapper=mapper,
            frontier_finder=frontier_finder,
            kino_planner=kino_planner,
            stage=get_current_stage(),
            dt=PHYSICS_DT,
            step_count=StepCount,
            debug_log=debug_log,
        )
        print(f"PLAN_LEN {StepCount} | refs {len(active_refs)} | seconds {len(active_refs) * PHYSICS_DT:.2f}", flush=True)
        print_ref_swap_debug(StepCount, drone_state_before_plan, old_ref, active_refs)
        if target_world is not None:
            target_points_world.append(target_world)
            if len(target_points_world) > DRAW_TARGET_MAX_POINTS:
                del target_points_world[:len(target_points_world) - DRAW_TARGET_MAX_POINTS]
        ref_index = 0
        fsm.active_refs = active_refs
        fsm.ref_index = ref_index
        update_trajectory_draw(trajectory_draw, StepCount, DRAW_TRAJECTORY_INTERVAL, flown_path_world, active_refs, ref_index, target_points_world, active_kino_path_world, force=True)

        if not active_refs:
            active_refs, state = make_hold_refs()
            ref_index = 0
            fsm.active_refs = active_refs
            fsm.ref_index = ref_index
            debug_log("hold_fallback", {"step": StepCount, "refs": len(active_refs), "pos": np.round(state["pos"], 3).tolist()})

    try:
        debug_log("loop_start")
        while simulation_app.is_running():
            should_plan, plan_reason = fsm.should_request_plan(
                step_count=StepCount,
                active_refs=active_refs,
                ref_index=ref_index,
                exploration_started=exploration_started,
            )
            if should_plan:
                request_plan(plan_reason)

            if not active_refs or ref_index >= len(active_refs):
                active_refs, state = make_hold_refs()
                ref_index = 0
                fsm.active_refs = active_refs
                fsm.ref_index = ref_index
                debug_log("hold_fallback", {"step": StepCount, "refs": len(active_refs), "pos": np.round(state["pos"], 3).tolist()})
            sigma, sigmad, sigmadd, sigmaddd = active_refs[ref_index]
            ref_index += 1
            u, _ = controller.PDControlLaw(sigma, sigmad, sigmadd, sigmaddd)
            drone.apply_control(u)
            world.step(render=True)
            current_state = drone.get_state()
            flown_path_world.append(np.asarray(current_state["pos"], dtype=float).reshape(3).copy())
            if len(flown_path_world) > DRAW_HISTORY_MAX_POINTS:
                del flown_path_world[:len(flown_path_world) - DRAW_HISTORY_MAX_POINTS]
            update_trajectory_draw(trajectory_draw, StepCount, DRAW_TRAJECTORY_INTERVAL, flown_path_world, active_refs, ref_index, target_points_world, active_kino_path_world)

            rgb = rgb_camera.get_rgb()
            depth = depth_camera.get_depth()
            T_camera_world = np.linalg.inv(depth_camera.get_view_matrix_ros())
            mapping_update = fsm.update_mapping_if_due(
                step_count=StepCount,
                rgb=rgb,
                depth=depth,
                camera_K=camera_K,
                T_camera_world=T_camera_world,
            )
            if mapping_update is not None:
                if LIVE_MAP_VIEWER and StepCount % fsm.viewer_update_interval == 0:
                    frontier_clusters = frontier_finder.frontiers + frontier_finder.dormant_frontiers

                    if frontier_clusters:
                        frontier_points = np.vstack([f.cells for f in frontier_clusters])
                        centroid_points = np.vstack([f.average for f in frontier_clusters])
                        palette = np.array([
                            [1.0, 0.2, 0.0],
                            [1.0, 0.75, 0.0],
                            [0.0, 0.9, 1.0],
                            [0.9, 0.0, 1.0],
                            [0.2, 1.0, 0.2],
                            [1.0, 0.45, 0.7],
                            [0.4, 0.4, 1.0],
                            [1.0, 1.0, 1.0],
                        ], dtype=float)
                        frontier_colors = np.vstack([
                            np.tile(palette[i % len(palette)], (len(frontier.cells), 1))
                            for i, frontier in enumerate(frontier_clusters)
                        ])
                    else:
                        frontier_points = None
                        frontier_colors = None
                        centroid_points = None

                    mapper.update_viewer(
                        show_free=True,
                        frontier_points=frontier_points,
                        frontier_colors=frontier_colors,
                        centroid_points=centroid_points,
                        best_viewpoint=best_viewpoint,
                        best_yaw=best_yaw,
                    )

                should_plan, plan_reason = fsm.should_request_plan(
                    step_count=StepCount,
                    active_refs=active_refs,
                    ref_index=ref_index,
                    exploration_started=exploration_started,
                )
                if should_plan:
                    request_plan(plan_reason)

            if StepCount > 0 and StepCount <= TSDF_SNAPSHOT_MAX_STEP and StepCount % TSDF_SNAPSHOT_INTERVAL == 0 and StepCount not in tsdf_snapshot_saved:
                mesh_path = ROOT / "exports" / f"tsdf_step_{StepCount}.ply"
                frontier_path = ROOT / "exports" / f"frontiers_step_{StepCount}.ply"
                centroid_path = ROOT / "exports" / f"centroids_step_{StepCount}.ply"
                viewpoint_path = ROOT / "exports" / f"viewpoints_step_{StepCount}.ply"
                heading_path = ROOT / "exports" / f"viewpoint_headings_step_{StepCount}.ply"
                reference_path = ROOT / "exports" / f"reference_step_{StepCount}.ply"
                kino_path = ROOT / "exports" / f"kino_path_step_{StepCount}.ply"
                mesh_info = mapper.save_tsdf_mesh(mesh_path)
                if mesh_info is not None:
                    frontier_clusters = frontier_finder.frontiers + frontier_finder.dormant_frontiers
                    frontier_info = None
                    centroid_info = None
                    viewpoint_info = None
                    heading_info = None
                    reference_info = None
                    kino_info = None


                    if frontier_clusters:
                        palette = np.array([
                            [1.0, 0.2, 0.0],
                            [1.0, 0.75, 0.0],
                            [0.0, 0.9, 1.0],
                            [0.9, 0.0, 1.0],
                            [0.2, 1.0, 0.2],
                            [1.0, 0.45, 0.7],
                            [0.4, 0.4, 1.0],
                            [1.0, 1.0, 1.0],
                        ], dtype=float)
                        frontier_points_map = np.vstack([f.cells for f in frontier_clusters])
                        frontier_points_world = mapper.map_to_world(frontier_points_map)
                        frontier_colors = np.vstack([
                            np.tile(palette[i % len(palette)], (len(frontier.cells), 1))
                            for i, frontier in enumerate(frontier_clusters)
                        ])
                        frontier_info = mapper.save_point_cloud(frontier_path, frontier_points_world, colors=frontier_colors)

                        centroid_points_map = np.vstack([frontier.average for frontier in frontier_clusters])
                        centroid_points_world = mapper.map_to_world(centroid_points_map)
                        centroid_colors = np.vstack([
                            palette[i % len(palette)]
                            for i, _ in enumerate(frontier_clusters)
                        ])
                        centroid_info = mapper.save_sphere_markers(centroid_path, centroid_points_world, colors=centroid_colors)

                        viewpoint_points_map = []
                        viewpoint_colors = []
                        heading_points_map = []
                        heading_lines = []
                        heading_colors = []
                        for frontier_index, frontier in enumerate(frontier_clusters):
                            color = palette[frontier_index % len(palette)]
                            for viewpoint in frontier.viewpoints:
                                pos = np.asarray(viewpoint.pos, dtype=float).reshape(3)
                                yaw = float(viewpoint.yaw)
                                heading_start = pos
                                heading_end = pos + 0.45 * np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=float)
                                line_start_index = len(heading_points_map)
                                viewpoint_points_map.append(pos)
                                viewpoint_colors.append(color)
                                heading_points_map.append(heading_start)
                                heading_points_map.append(heading_end)
                                heading_lines.append([line_start_index, line_start_index + 1])
                                heading_colors.append(color)

                        if viewpoint_points_map:
                            viewpoint_points_world = mapper.map_to_world(np.asarray(viewpoint_points_map, dtype=float))
                            viewpoint_info = mapper.save_point_cloud(
                                viewpoint_path,
                                viewpoint_points_world,
                                colors=np.asarray(viewpoint_colors, dtype=float),
                            )
                        if heading_lines:
                            heading_points_world = mapper.map_to_world(np.asarray(heading_points_map, dtype=float))
                            heading_info = mapper.save_line_set(
                                heading_path,
                                heading_points_world,
                                np.asarray(heading_lines, dtype=np.int32),
                                colors=np.asarray(heading_colors, dtype=float),
                            )
                    reference_points = np.asarray(
                        [ref[0][:3] for ref in active_refs[ref_index:]],
                        dtype=float,
                    ).reshape((-1, 3))
                    if len(reference_points) >= 2:
                        reference_lines = np.asarray(
                            [[i, i + 1] for i in range(len(reference_points) - 1)],
                            dtype=np.int32,
                        )
                        reference_info = mapper.save_line_set(
                            reference_path,
                            reference_points,
                            reference_lines,
                            colors=np.tile(np.array([[1.0, 0.05, 0.0]], dtype=float), (len(reference_lines), 1)),
                        )

                    kino_points = np.asarray(active_kino_path_world, dtype=float).reshape((-1, 3))
                    if len(kino_points) >= 2:
                        kino_lines = np.asarray(
                            [[i, i + 1] for i in range(len(kino_points) - 1)],
                            dtype=np.int32,
                        )
                        kino_info = mapper.save_line_set(
                            kino_path,
                            kino_points,
                            kino_lines,
                            colors=np.tile(np.array([[1.0, 0.55, 0.0]], dtype=float), (len(kino_lines), 1)),
                        )
                    debug_log(
                        "tsdf_snapshot",
                        {
                            "step": StepCount,
                            "saved": True,
                            "mesh_path": mesh_info["path"],
                            "vertices": mesh_info["vertices"],
                            "triangles": mesh_info["triangles"],
                            "frontier_points": 0 if frontier_info is None else frontier_info["points"],
                            "centroids": 0 if centroid_info is None else centroid_info["markers"],
                            "viewpoints": 0 if viewpoint_info is None else viewpoint_info["points"],
                            "headings": 0 if heading_info is None else heading_info["lines"],
                            "reference_lines": 0 if reference_info is None else reference_info["lines"],
                            "kino_lines": 0 if kino_info is None else kino_info["lines"],
                        },
                    )
                    tsdf_snapshot_saved.add(StepCount)
                else:
                    debug_log("tsdf_snapshot", {"step": StepCount, "saved": False})

            StepCount += 1
    except KeyboardInterrupt:
        debug_log("keyboard_interrupt")
    except Exception:
        debug_log("main_exception")
        traceback.print_exc()
    finally:
        debug_log("shutdown", {"step": StepCount, "app_running": simulation_app.is_running()})
        simulation_app.close()


if __name__ == "__main__":
    main()







