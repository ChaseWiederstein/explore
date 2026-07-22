from contextlib import nullcontext

import numpy as np

from bspline import bspline_refs_from_kino_path


def timing_section(timing, name):
    return timing.section(name) if timing is not None else nullcontext()


def validate_path(points, mapper, spacing):
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return False, None

    last_checked = points[0]
    if mapper.is_occupied(last_checked):
        return False, last_checked

    for point in points[1:]:
        if np.linalg.norm(point - last_checked) < spacing:
            continue
        if mapper.is_occupied(point):
            return False, point
        last_checked = point

    if mapper.is_occupied(points[-1]):
        return False, points[-1]

    return True, None


def path_length(path):
    if path is None or len(path) < 2:
        return 0.0

    points = np.asarray(path, dtype=float).reshape((-1, 3))
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def truncate_path(path, max_length):
    points = np.asarray(path, dtype=float).reshape((-1, 3))
    if len(points) < 2:
        return points

    max_length = float(max_length)
    kept = [points[0]]
    traveled = 0.0

    for point in points[1:]:
        last = kept[-1]
        segment = point - last
        segment_length = float(np.linalg.norm(segment))
        if segment_length <= 1e-9:
            continue

        if traveled + segment_length >= max_length:
            ratio = max(0.0, min(1.0, (max_length - traveled) / segment_length))
            kept.append(last + ratio * segment)
            break

        kept.append(point)
        traveled += segment_length

    return np.asarray(kept, dtype=float)


def geometric_kino_path(path, start_vel=None):
    points = np.asarray(path, dtype=float).reshape((-1, 3))
    velocities = [np.zeros(3, dtype=float) for _ in points]
    if start_vel is not None and len(velocities) > 0:
        velocities[0] = np.asarray(start_vel, dtype=float).reshape(3)
    return list(zip(points, velocities))


def trajectory_collision_risk(refs, start_index, mapper, lookahead_distance):
    if not refs or start_index >= len(refs):
        return False, None, 0.0

    current_pos = np.asarray(refs[start_index][0][:3], dtype=float)

    for idx in range(start_index, len(refs)):
        pos = np.asarray(refs[idx][0][:3], dtype=float)
        distance = float(np.linalg.norm(pos - current_pos))
        if distance > lookahead_distance:
            return False, None, distance
        if mapper.is_occupied(pos):
            return True, pos, distance

    return False, None, float(np.linalg.norm(np.asarray(refs[-1][0][:3], dtype=float) - current_pos))


def takeoff_refs(start_pos, start_vel, target_z, dt, duration, yaw=0.0):
    start_pos = np.asarray(start_pos, dtype=float)
    start_vel = np.asarray(start_vel, dtype=float)
    end_pos = np.array([start_pos[0], start_pos[1], float(target_z)], dtype=float)
    end_vel = np.zeros(3, dtype=float)

    steps = max(1, int(round(duration / dt)))
    duration = float(duration)

    refs = []
    for k in range(steps):
        s = min(max(k * float(dt) / max(duration, 1e-6), 0.0), 1.0)
        h = 3.0 * s * s - 2.0 * s * s * s
        hd = (6.0 * s - 6.0 * s * s) / max(duration, 1e-6)
        hdd = (6.0 - 12.0 * s) / max(duration * duration, 1e-6)
        hddd = -12.0 / max(duration ** 3, 1e-6)
        delta = end_pos - start_pos
        pos = start_pos + h * delta
        vel = hd * delta
        acc = hdd * delta
        jerk = hddd * delta

        sigma = np.array([pos[0], pos[1], pos[2], float(yaw)], dtype=float)
        sigmad = np.array([vel[0], vel[1], vel[2], 0.0], dtype=float)
        sigmadd = np.array([acc[0], acc[1], acc[2], 0.0], dtype=float)
        sigmaddd = np.array([jerk[0], jerk[1], jerk[2], 0.0], dtype=float)
        refs.append((sigma, sigmad, sigmadd, sigmaddd))

    return refs


def refs_to_points(refs):
    return np.asarray([ref[0][:3] for ref in refs], dtype=float).reshape((-1, 3))


def plan_reference(
    *,
    planner,
    mapper,
    stage,
    start_pos,
    start_vel,
    start_yaw,
    goal_pos,
    dt,
    reference_check_spacing,
    draw_reference_path=None,
    draw_waypoint_markers=None,
    guide_path=None,
    close_distance=0,
    far_distance=100000,
    timing=None,
):
    with timing_section(timing, "clear_old_draw"):
        stage.RemovePrim("/World/planned_waypoints")
        stage.RemovePrim("/World/planned_reference_path")

    start_pos = np.asarray(start_pos, dtype=float).reshape(3)
    goal_pos = np.asarray(goal_pos, dtype=float).reshape(3)
    active_goal = goal_pos
    guide_len = None
    plan_mode = "kino"
    kino_path = None

    with timing_section(timing, "mode_select"):
        if guide_path is not None and len(guide_path) >= 2:
            guide_path = np.asarray(guide_path, dtype=float).reshape((-1, 3))
            guide_len = path_length(guide_path)
            if guide_len < float(close_distance):
                plan_mode = "geometric_close"
                kino_path = geometric_kino_path(guide_path, start_vel=start_vel)
            elif guide_len > float(far_distance):
                plan_mode = "geometric_truncated"
                guide_path = truncate_path(guide_path, far_distance)
                active_goal = guide_path[-1]
                kino_path = geometric_kino_path(guide_path, start_vel=start_vel)

    if kino_path is None:
        with timing_section(timing, "kino_astar") as kino_timing:
            kino_path = planner.search_waypoints(
                waypoints=[start_pos, active_goal],
                start_vel=start_vel,
                timing=kino_timing,
            )

    if kino_path is None:
        return [], f"plan_failed mode={plan_mode}", []

    final_goal_error = float(np.linalg.norm(np.asarray(kino_path[-1][0], dtype=float) - active_goal))
    if not planner.reached_goal(kino_path[-1][0], active_goal):
        return [], (
            f"plan_incomplete mode={plan_mode} final_error={final_goal_error:.2f}"
        ), kino_path

    with timing_section(timing, "bspline_parameterize") as bspline_timing:
        refs, control_points, bspline_debug = bspline_refs_from_kino_path(
            kino_path,
            controller_dt=dt,
            astar_tau=planner.tau,
            max_vel=planner.max_vel,
            yaw=start_yaw,
            ctrl_point_distance=0.35,
            degree=3,
            requested_control_points=None,
        )
        bspline_timing.add_detail(f"astar_samples {bspline_debug.get('samples_from_astar', 0)}")
        bspline_timing.add_detail(f"control_points {bspline_debug.get('control_points', 0)}")
        bspline_timing.add_detail(f"knot_span {bspline_debug.get('knot_span', 0.0):.3f}")

    if len(refs) == 0:
        return [], f"bspline_empty mode={plan_mode}", kino_path

    reference_path = refs_to_points(refs)

    with timing_section(timing, "bspline_collision"):
        bspline_ok, blocked_point = validate_path(
            reference_path,
            mapper,
            spacing=reference_check_spacing,
        )

    if draw_waypoint_markers is not None:
        waypoint_points = np.asarray([pos for pos, _ in kino_path], dtype=float)
        draw_waypoint_markers(
            stage,
            waypoint_points,
            root_path="/World/planned_waypoints",
            color=(0.0, 0.2, 1.0),
            radius=0.03,
        )

    if draw_reference_path is not None:
        draw_reference_path(
            stage,
            reference_path,
            curve_path="/World/planned_reference_path",
            color=(1.0, 0.2, 0.0) if bspline_ok else (1.0, 0.0, 0.0),
            width=0.03,
        )

    if not bspline_ok:
        if blocked_point is None:
            return [], f"bspline_blocked mode={plan_mode}", kino_path
        return [], (
            f"bspline_blocked at "
            f"({blocked_point[0]:.2f}, {blocked_point[1]:.2f}, {blocked_point[2]:.2f}) mode={plan_mode}"
        ), kino_path

    with timing_section(timing, "refs_generate"):
        refs = list(refs)
    guide_msg = "" if guide_len is None else f" guide={guide_len:.2f}"
    return (
        refs,
        (
            f"mode={plan_mode}{guide_msg} kino={len(kino_path)} "
            f"astar_samples={bspline_debug.get('samples_from_astar', 0)} "
            f"control_points={bspline_debug.get('control_points', 0)} "
            f"refs={len(refs)}"
        ),
        kino_path,
    )
