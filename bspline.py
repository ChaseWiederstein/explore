import numpy as np


def _kino_arrays(kino_path):
    positions = []
    velocities = []
    for pos, vel in kino_path:
        positions.append(np.asarray(pos, dtype=float).reshape(3))
        velocities.append(np.asarray(vel, dtype=float).reshape(3))
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    return positions, velocities


def _segment_coefficients(p0, v0, p1, v1, duration):
    duration = max(float(duration), 1e-6)
    a0 = np.asarray(p0, dtype=float)
    a1 = np.asarray(v0, dtype=float)
    delta = np.asarray(p1, dtype=float) - a0
    a2 = (3.0 * delta / (duration * duration)) - ((2.0 * a1 + np.asarray(v1, dtype=float)) / duration)
    a3 = (-2.0 * delta / (duration ** 3)) + ((a1 + np.asarray(v1, dtype=float)) / (duration * duration))
    return a0, a1, a2, a3


def _segment_state(p0, v0, p1, v1, duration, t):
    a0, a1, a2, a3 = _segment_coefficients(p0, v0, p1, v1, duration)
    t = max(0.0, min(float(t), float(duration)))
    pos = a0 + a1 * t + a2 * t * t + a3 * t * t * t
    vel = a1 + 2.0 * a2 * t + 3.0 * a3 * t * t
    acc = 2.0 * a2 + 6.0 * a3 * t
    return pos, vel, acc


def _state_at_time(positions, velocities, tau, t):
    if len(positions) == 1:
        return positions[0].copy(), velocities[0].copy()

    total_time = tau * (len(positions) - 1)
    t = max(0.0, min(float(t), total_time))
    if t >= total_time:
        return positions[-1].copy(), velocities[-1].copy()

    seg = min(int(t // tau), len(positions) - 2)
    local_t = t - seg * tau
    pos, vel, _ = _segment_state(
        positions[seg],
        velocities[seg],
        positions[seg + 1],
        velocities[seg + 1],
        tau,
        local_t,
    )
    return pos, vel


def sample_kino_path_for_bspline(
    kino_path,
    *,
    astar_tau,
    max_vel,
    ctrl_point_distance=0.35,
    degree=3,
    requested_control_points=None,
    min_segments=8,
):
    positions, velocities = _kino_arrays(kino_path)
    if len(positions) < 2:
        return None

    total_time = float(astar_tau) * (len(positions) - 1)
    if total_time <= 1e-9:
        return None

    if requested_control_points is not None:
        control_points_requested = max(int(requested_control_points), degree + 2)
        sample_count = control_points_requested - degree + 1
        segment_count = max(sample_count - 1, 1)
    else:
        sample_dt_guess = float(ctrl_point_distance) / max(float(max_vel), 1e-6)
        segment_count = max(int(np.floor(total_time / sample_dt_guess)), int(min_segments))

    sample_dt = total_time / float(segment_count)
    times = np.linspace(0.0, total_time, segment_count + 1)
    point_set = np.asarray([_state_at_time(positions, velocities, astar_tau, t)[0] for t in times], dtype=float)

    _, _, start_acc = _segment_state(positions[0], velocities[0], positions[1], velocities[1], astar_tau, 0.0)
    _, _, end_acc = _segment_state(positions[-2], velocities[-2], positions[-1], velocities[-1], astar_tau, astar_tau)
    derivatives = [
        velocities[0].copy(),
        velocities[-1].copy(),
        start_acc,
        end_acc,
    ]

    return {
        "point_set": point_set,
        "derivatives": derivatives,
        "sample_dt": sample_dt,
        "duration": total_time,
        "degree": int(degree),
    }


def parameterize_to_bspline(sample_dt, point_set, start_end_derivatives, degree=3):
    point_set = np.asarray(point_set, dtype=float).reshape((-1, 3))
    if degree != 3:
        raise ValueError("Only cubic B-spline parameterization is wired right now.")
    if len(point_set) < 2:
        raise ValueError("Need at least two sampled kino positions.")
    if len(start_end_derivatives) != 4:
        raise ValueError("Need start_vel, end_vel, start_acc, end_acc.")

    sample_dt = float(sample_dt)
    sample_count = len(point_set)
    control_count = sample_count + degree - 1
    a = np.zeros((sample_count + 4, control_count), dtype=float)
    b = np.zeros((sample_count + 4, 3), dtype=float)

    pt_to_pos = np.array([1.0, 4.0, 1.0], dtype=float) / 6.0
    pt_to_vel = np.array([-1.0, 0.0, 1.0], dtype=float) / (2.0 * sample_dt)
    pt_to_acc = np.array([1.0, -2.0, 1.0], dtype=float) / (sample_dt * sample_dt)

    for i in range(sample_count):
        a[i, i:i + 3] = pt_to_pos
        b[i] = point_set[i]

    a[sample_count, 0:3] = pt_to_vel
    a[sample_count + 1, sample_count - 1:sample_count + 2] = pt_to_vel
    a[sample_count + 2, 0:3] = pt_to_acc
    a[sample_count + 3, sample_count - 1:sample_count + 2] = pt_to_acc

    for i, derivative in enumerate(start_end_derivatives):
        b[sample_count + i] = np.asarray(derivative, dtype=float).reshape(3)

    control_points = np.linalg.lstsq(a, b, rcond=None)[0]
    return control_points


def uniform_knots(control_count, degree, knot_span):
    n = int(control_count) - 1
    m = n + int(degree) + 1
    knots = np.zeros(m + 1, dtype=float)
    for i in range(m + 1):
        if i <= degree:
            knots[i] = float(-degree + i) * float(knot_span)
        else:
            knots[i] = knots[i - 1] + float(knot_span)
    return knots


def evaluate_de_boor(control_points, degree, knots, t):
    control_points = np.asarray(control_points, dtype=float)
    degree = int(degree)
    knots = np.asarray(knots, dtype=float)
    u = float(t) + knots[degree]
    u = min(max(knots[degree], u), knots[len(knots) - degree - 1])

    k = degree
    while k + 1 < len(knots) and knots[k + 1] < u:
        k += 1

    d = [control_points[k - degree + i].copy() for i in range(degree + 1)]
    for r in range(1, degree + 1):
        for i in range(degree, r - 1, -1):
            denom = knots[i + 1 + k - r] - knots[i + k - degree]
            alpha = 0.0 if abs(denom) <= 1e-12 else (u - knots[i + k - degree]) / denom
            d[i] = (1.0 - alpha) * d[i - 1] + alpha * d[i]
    return d[degree]


def derivative_curve(control_points, degree, knots):
    control_points = np.asarray(control_points, dtype=float)
    degree = int(degree)
    knots = np.asarray(knots, dtype=float)
    if degree <= 0 or len(control_points) < 2:
        return np.zeros((0, control_points.shape[1]), dtype=float), max(degree - 1, 0), knots[1:-1]

    derivative_points = []
    for i in range(len(control_points) - 1):
        denom = knots[i + degree + 1] - knots[i + 1]
        scale = 0.0 if abs(denom) <= 1e-12 else degree / denom
        derivative_points.append(scale * (control_points[i + 1] - control_points[i]))
    return np.asarray(derivative_points, dtype=float), degree - 1, knots[1:-1]


def evaluate_bspline_with_derivatives(control_points, degree, knot_span, t):
    knots = uniform_knots(len(control_points), degree, knot_span)
    pos = evaluate_de_boor(control_points, degree, knots, t)

    vel_points, vel_degree, vel_knots = derivative_curve(control_points, degree, knots)
    vel = evaluate_de_boor(vel_points, vel_degree, vel_knots, t) if len(vel_points) else np.zeros(3, dtype=float)

    acc_points, acc_degree, acc_knots = derivative_curve(vel_points, vel_degree, vel_knots)
    acc = evaluate_de_boor(acc_points, acc_degree, acc_knots, t) if len(acc_points) else np.zeros(3, dtype=float)

    jerk_points, jerk_degree, jerk_knots = derivative_curve(acc_points, acc_degree, acc_knots)
    jerk = evaluate_de_boor(jerk_points, jerk_degree, jerk_knots, t) if len(jerk_points) else np.zeros(3, dtype=float)

    return pos, vel, acc, jerk


def bspline_refs_from_kino_path(
    kino_path,
    *,
    controller_dt,
    astar_tau,
    max_vel,
    yaw,
    ctrl_point_distance=0.35,
    degree=3,
    requested_control_points=None,
):
    sampled = sample_kino_path_for_bspline(
        kino_path,
        astar_tau=astar_tau,
        max_vel=max_vel,
        ctrl_point_distance=ctrl_point_distance,
        degree=degree,
        requested_control_points=requested_control_points,
    )
    if sampled is None:
        return [], np.empty((0, 3), dtype=float), {}

    control_points = parameterize_to_bspline(
        sampled["sample_dt"],
        sampled["point_set"],
        sampled["derivatives"],
        degree=degree,
    )

    duration = sampled["duration"]
    steps = max(1, int(np.ceil(duration / float(controller_dt))) + 1)
    refs = []
    yaw = float(yaw)
    for k in range(steps):
        t = min(k * float(controller_dt), duration)
        pos, vel, acc, jerk = evaluate_bspline_with_derivatives(control_points, degree, sampled["sample_dt"], t)
        sigma = np.array([pos[0], pos[1], pos[2], yaw], dtype=float)
        sigmad = np.array([vel[0], vel[1], vel[2], 0.0], dtype=float)
        sigmadd = np.array([acc[0], acc[1], acc[2], 0.0], dtype=float)
        sigmaddd = np.array([jerk[0], jerk[1], jerk[2], 0.0], dtype=float)
        refs.append((sigma, sigmad, sigmadd, sigmaddd))

    debug = {
        "degree": degree,
        "samples_from_astar": len(sampled["point_set"]),
        "control_points": len(control_points),
        "requested_control_points": requested_control_points,
        "knot_span": sampled["sample_dt"],
        "duration": duration,
    }
    return refs, control_points, debug
