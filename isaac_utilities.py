import numpy as np


def update_trajectory_draw(
    trajectory_draw,
    step_count,
    draw_interval,
    flown_path_world,
    active_refs,
    ref_index,
    target_points_world,
    active_kino_path_world,
    force=False,
):
    if trajectory_draw is None:
        return
    if not force and step_count % draw_interval != 0:
        return
    trajectory_draw.update(
        flown_path_world,
        active_refs[ref_index:],
        target_points_world,
        active_kino_path_world,
    )


class IsaacTrajectoryDebugDraw:
    def __init__(self):
        self.draw = None
        self.available = False
        self.static_lines = []
        try:
            from omni.isaac.debug_draw import _debug_draw
            self.draw = _debug_draw.acquire_debug_draw_interface()
            self.available = True
        except Exception:
            try:
                from isaacsim.util.debug_draw import _debug_draw
                self.draw = _debug_draw.acquire_debug_draw_interface()
                self.available = True
            except Exception as exc:
                print(f"DRAW | disabled | {exc}", flush=True)

    def set_volume_box(self, corners_world, color=(1.0, 0.0, 0.0, 1.0), size=6.0):
        corners = np.asarray(corners_world, dtype=float).reshape((8, 3))
        edges = (
            (0, 1), (1, 3), (3, 2), (2, 0),
            (4, 5), (5, 7), (7, 6), (6, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        )
        self.static_lines = [
            (corners[start].copy(), corners[end].copy(), tuple(color), float(size))
            for start, end in edges
        ]


    def clear(self):
        if not self.available:
            return
        try:
            self.draw.clear_lines()
        except Exception:
            pass
        try:
            self.draw.clear_points()
        except Exception:
            pass

    def update(self, history_world, refs_world, targets_world, kino_path_world=None):
        if not self.available:
            return

        self.clear()

        starts = []
        ends = []
        colors = []
        sizes = []

        for start, end, color, size in self.static_lines:
            starts.append(tuple(start.tolist()))
            ends.append(tuple(end.tolist()))
            colors.append(color)
            sizes.append(size)

        history = np.asarray(history_world, dtype=float).reshape((-1, 3))
        if len(history) >= 2:
            for start, end in zip(history[:-1], history[1:]):
                starts.append(tuple(start.tolist()))
                ends.append(tuple(end.tolist()))
                colors.append((0.0, 1.0, 0.1, 1.0))
                sizes.append(3.0)

        plan_points = [
            np.asarray(ref[0][:3], dtype=float)
            for ref in refs_world
        ]
        if len(plan_points) >= 2:
            for start, end in zip(plan_points[:-1], plan_points[1:]):
                starts.append(tuple(start.tolist()))
                ends.append(tuple(end.tolist()))
                colors.append((1.0, 0.05, 0.0, 1.0))
                sizes.append(5.0)

        kino_points = (
            np.asarray(kino_path_world, dtype=float).reshape((-1, 3))
            if kino_path_world is not None and len(kino_path_world) > 0
            else np.empty((0, 3))
        )
        if len(kino_points) >= 2:
            for start, end in zip(kino_points[:-1], kino_points[1:]):
                starts.append(tuple(start.tolist()))
                ends.append(tuple(end.tolist()))
                colors.append((1.0, 0.55, 0.0, 1.0))
                sizes.append(7.0)

        if starts:
            self.draw.draw_lines(starts, ends, colors, sizes)

        targets = (
            np.asarray(targets_world, dtype=float).reshape((-1, 3))
            if targets_world
            else np.empty((0, 3))
        )
        if len(targets) > 0:
            if hasattr(self.draw, "draw_points"):
                self.draw.draw_points(
                    [tuple(target.tolist()) for target in targets],
                    [(1.0, 0.9, 0.0, 1.0) for _ in targets],
                    [18.0 for _ in targets],
                )
            else:
                marker_radius = 0.12
                marker_starts = []
                marker_ends = []
                marker_colors = []
                marker_sizes = []
                for target in targets:
                    for offset in (
                        np.array([marker_radius, 0.0, 0.0]),
                        np.array([0.0, marker_radius, 0.0]),
                        np.array([0.0, 0.0, marker_radius]),
                    ):
                        marker_starts.append(tuple((target - offset).tolist()))
                        marker_ends.append(tuple((target + offset).tolist()))
                        marker_colors.append((1.0, 0.9, 0.0, 1.0))
                        marker_sizes.append(6.0)
                self.draw.draw_lines(marker_starts, marker_ends, marker_colors, marker_sizes)
