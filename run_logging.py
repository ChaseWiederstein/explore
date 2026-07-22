import re


VERBOSE_DEBUG = False


def debug_log(label, data=None, verbose=None):
    clean_label = str(label).rstrip(":")

    if verbose is None:
        verbose = VERBOSE_DEBUG

    if verbose:
        if data is None:
            print(f"[debug] {clean_label}", flush=True)
        else:
            print(f"[debug] {clean_label}: {data}", flush=True)
        return

    def vec_text(values):
        if values is None:
            return "n/a"
        return "(" + ", ".join(f"{float(value):.2f}" for value in values[:3]) + ")"

    def plan_mode(plan_msg):
        match = re.search(r"\bmode=([A-Za-z0-9_]+)", str(plan_msg))
        return match.group(1) if match else "unknown"

    def plan_failure(plan_msg):
        plan_msg = str(plan_msg)
        if plan_msg.startswith("bspline_blocked"):
            match = re.search(r"at \(([^)]+)\)", plan_msg)
            return f"blocked at ({match.group(1)})" if match else "blocked"
        if plan_msg.startswith("plan_failed"):
            return "planner failed"
        if plan_msg.startswith("plan_incomplete"):
            return "incomplete"
        if plan_msg.startswith("bspline_empty"):
            return "empty path"
        return plan_msg

    if clean_label == "map_update" and isinstance(data, dict):
        frontiers = data.get("frontiers", {})
        print(
            f"STEP {data.get('step')} | known {float(data.get('known_pct', 0.0)):.1f}% "
            f"| frontiers {frontiers.get('active', 0)} active, {frontiers.get('dormant', 0)} dormant",
            flush=True,
        )
        return

    if clean_label == "map_timing" and isinstance(data, dict):
        print(
            f"MAPTIME {data.get('step')} | total {float(data.get('total', 0.0)):.3f}s "
            f"| depth_points {float(data.get('depth_points', 0.0)):.3f}s "
            f"| raycast_free {float(data.get('raycast_free', 0.0)):.3f}s "
            f"| frontier_search {float(data.get('frontier_search', 0.0)):.3f}s "
            f"| frontier_visit {float(data.get('frontier_visit', 0.0)):.3f}s "
            f"| frontier_costs {float(data.get('frontier_costs', 0.0)):.3f}s",
            flush=True,
        )
        return

    if clean_label == "dormant_frontiers" and isinstance(data, dict):
        print(
            f"DORMANT {data.get('step')} | {data.get('dormant', 0)} dormant "
            f"| viewpoints {data.get('passed', 0)}/{data.get('candidates', 0)} "
            f"| top_fail {data.get('top_reason')}={data.get('top_count', 0)}",
            flush=True,
        )
        return
    if clean_label == "tsdf_snapshot" and isinstance(data, dict):
        if data.get("saved"):
            print(
                f"TSDF {data.get('step')} | saved | mesh {data.get('mesh_path')} "
                f"| vertices {data.get('vertices', 0)} | triangles {data.get('triangles', 0)} "
                f"| frontiers {data.get('frontier_points', 0)} "
                f"| centroids {data.get('centroids', 0)} "
                f"| viewpoints {data.get('viewpoints', 0)} "
                f"| headings {data.get('headings', 0)}",
                flush=True,
            )
        else:
            print(f"TSDF {data.get('step')} | save skipped | no mesh yet", flush=True)
        return

    if clean_label == "plan_start" and isinstance(data, dict):
        print(
            f"PLAN {data.get('step')} | start | frontiers {data.get('frontiers', 0)} "
            f"| pos {vec_text(data.get('pos_map'))} | yaw {float(data.get('yaw_map_deg', 0.0)):.1f} deg",
            flush=True,
        )
        return

    if clean_label == "exploration_plan" and isinstance(data, dict):
        step = data.get("step")
        prefix = f"PLAN {step}" if step is not None else "PLAN"
        if data.get("refs", 0) > 0:
            print(
                f"{prefix} | OK | pos {vec_text(data.get('pos'))} | target {vec_text(data.get('target'))} "
                f"| mode {plan_mode(data.get('plan'))} | refs {data.get('refs')}",
                flush=True,
            )
        else:
            print(
                f"{prefix} | FAIL | pos {vec_text(data.get('pos'))} | target {vec_text(data.get('target'))} "
                f"| mode {plan_mode(data.get('plan'))} | {plan_failure(data.get('plan'))}",
                flush=True,
            )
        return

    if clean_label == "hold_fallback" and isinstance(data, dict):
        step = data.get("step")
        prefix = f"HOLD {step}" if step is not None else "HOLD"
        print(f"{prefix} | {data.get('refs', 0)} refs | pos {vec_text(data.get('pos'))}", flush=True)
        return

    if clean_label == "plan_timing_tree" and isinstance(data, dict):
        timing = data.get("timing")
        if timing is not None:
            for line in timing.format_lines(step=data.get("step")):
                print(line, flush=True)
        return

    if clean_label == "exploration_plan_failed":
        print(f"PLAN | FAIL | {data}", flush=True)
        return

    if clean_label == "exploration_plan_exception":
        print("PLAN | EXCEPTION", flush=True)
        return

    if clean_label == "world_initialized":
        print("WORLD | initialized", flush=True)
        return

    if clean_label == "warmup_done":
        print("WORLD | warmup done", flush=True)
        return

    if clean_label == "loop_start":
        print("LOOP | start", flush=True)
        return

    if clean_label == "keyboard_interrupt":
        print("STOP | keyboard interrupt", flush=True)
        return

    if clean_label == "main_exception":
        print("STOP | exception", flush=True)
        return

    if clean_label == "shutdown":
        step = data.get("step") if isinstance(data, dict) else None
        print(f"STOP | shutdown at step {step}", flush=True)
        return

    if clean_label in {"refs_exhausted", "coarse_route", "plan_refs_world", "vp_sampling"}:
        return

    if data is None:
        print(clean_label, flush=True)
    else:
        print(f"{clean_label} | {data}", flush=True)

