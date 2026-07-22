import heapq
import time
from dataclasses import dataclass
import numpy as np


@dataclass
class Node:
    pos: np.ndarray
    vel: np.ndarray

    g: float
    f: float

    parent: object = None
    u: np.ndarray = None

    def __lt__(self, other):
        return self.f < other.f
class KinodynamicAStar:

    def __init__(
        self,
        occupancy_map,
        max_acc=4.0,
        max_vel=4.0,
        tau=0.5,
        rho=.05,
        w_time=10.0,
        goal_tol=0.1,
        replan_length = 1.5,
        collision_risk_tol = .1, # could change need to look at voxel inflation
        planning_horizon=7.0,
        lambda_heu=.5,

    ):
        self.map = occupancy_map

        self.max_acc = float(max_acc)
        self.max_vel = float(max_vel)

        self.tau = float(tau)
        self.rho = float(rho)
        self.w_time = float(w_time)

        self.goal_tol = float(goal_tol)
        self.replan_length = float(replan_length)
        self.collision_risk_tol = float(collision_risk_tol)
        self.planning_horizon = float(planning_horizon)
        self.lambda_heu = float(lambda_heu)

        self.max_iterations = 50000
        self.controls = self.generate_controls()
        self.profile = None

    def _reset_profile(self):
        self.profile = {
            "total_time": 0.0,
            "pop_time": 0.0,
            "propagate_time": 0.0,
            "velocity_prune_time": 0.0,
            "closed_check_time": 0.0,
            "collision_time": 0.0,
            "is_occupied_time": 0.0,
            "score_time": 0.0,
            "push_time": 0.0,
            "iterations": 0,
            "children_attempted": 0,
            "velocity_pruned": 0,
            "closed_pruned": 0,
            "collision_pruned": 0,
            "collision_samples": 0,
            "nodes_pushed": 0,
        }

    def _attach_profile_timing(self, timing, status):
        if timing is None or self.profile is None:
            return

        self.profile["status"] = status
        timing.add_detail(f"status {status}")
        timing.add_detail(f"iters {self.profile['iterations']}")
        timing.add_detail(f"children {self.profile['children_attempted']}")

        timing.add_child("pop", self.profile["pop_time"])
        timing.add_child("propagate", self.profile["propagate_time"])
        timing.add_child("velocity_prune", self.profile["velocity_prune_time"], f"pruned {self.profile['velocity_pruned']}")
        timing.add_child("closed_check", self.profile["closed_check_time"], f"pruned {self.profile['closed_pruned']}")
        collision = timing.add_child(
            "collision_check",
            self.profile["collision_time"],
            [
                f"pruned {self.profile['collision_pruned']}",
                f"samples {self.profile['collision_samples']}",
            ],
        )
        collision.add_child("is_occupied", self.profile["is_occupied_time"])
        timing.add_child("score", self.profile["score_time"])
        timing.add_child("push", self.profile["push_time"], f"nodes {self.profile['nodes_pushed']}")

    def generate_controls(self):
        vals = np.linspace(-self.max_acc, self.max_acc, 5)

        controls = []

        for ax in vals:
            for ay in vals:
                for az in vals:
                    controls.append(
                        np.array([ax, ay, az], dtype=float)
                    )

        return controls

    def propagate(self, pos, vel, u):
        pos = np.asarray(pos, dtype=float)
        vel = np.asarray(vel, dtype=float)
        u = np.asarray(u, dtype=float)

        new_pos = (
                pos
                + vel * self.tau
                + 0.5 * u * self.tau ** 2
        )

        new_vel = (
                vel
                + u * self.tau
        )

        return new_pos, new_vel

    def heuristic(self, pos, goal):
        return np.linalg.norm(pos - goal)

    def kinodynamic_heuristic_with_time(self, pos, vel, goal_pos, goal_vel=None):
        pos = np.asarray(pos, dtype=float)
        vel = np.asarray(vel, dtype=float)
        goal_pos = np.asarray(goal_pos, dtype=float)
        if goal_vel is None:
            goal_vel = np.zeros(3, dtype=float)
        else:
            goal_vel = np.asarray(goal_vel, dtype=float)

        dp = goal_pos - pos
        v0 = vel
        v1 = goal_vel

        c1 = -36.0 * float(np.dot(dp, dp))
        c2 = 24.0 * float(np.dot(v0 + v1, dp))
        c3 = -4.0 * float(np.dot(v0, v0) + np.dot(v0, v1) + np.dot(v1, v1))
        c4 = 0.0
        c5 = self.w_time

        roots = np.roots([c5, c4, c3, c2, c1])
        candidates = [
            float(root.real)
            for root in roots
            if abs(root.imag) < 1e-6 and root.real > 1e-6
        ]

        v_max = max(self.max_vel * 0.5, 1e-6)
        t_bar = float(np.linalg.norm(dp, ord=np.inf) / v_max)
        candidates.append(max(t_bar, 1e-6))

        best_cost = float("inf")
        for t in candidates:
            if t < t_bar:
                continue
            cost = -c1 / (3.0 * t ** 3) - c2 / (2.0 * t ** 2) - c3 / t + self.w_time * t
            if cost < best_cost:
                best_cost = float(cost)
                best_time = float(t)

        if not np.isfinite(best_cost):
            return self.heuristic(pos, goal_pos), max(t_bar, 1e-6)

        tie_breaker = 1.0 + 1.0 / 10000.0
        return tie_breaker * best_cost, best_time

    def kinodynamic_heuristic(self, pos, vel, goal_pos, goal_vel=None):
        cost, _ = self.kinodynamic_heuristic_with_time(pos, vel, goal_pos, goal_vel)
        return cost

    def edge_cost(self, u):
        return (
                np.dot(u, u)*.1
                + self.rho
        ) * self.tau

    def reached_goal(self, pos, goal):
        voxel_size = float(self.map.voxel_size)
        tol_voxels = max(1, int(np.ceil(self.goal_tol / voxel_size)))

        pos_idx = self.state_key(pos)
        goal_idx = self.state_key(goal)

        return all(
            abs(pos_idx[i] - goal_idx[i]) <= tol_voxels
            for i in range(3)
        )

    def state_key(self, pos):
        voxel_size = self.map.voxel_size

        idx = np.floor(
            np.asarray(pos) / voxel_size
        ).astype(int)

        return tuple(idx)

    def reconstruct_path(self, node):

        path = []

        while node is not None:
            path.append(
                (
                    node.pos.copy(),
                    node.vel.copy(),
                )
            )

            node = node.parent

        path.reverse()

        return path

    def primitive_collision_free(
            self,
            pos,
            vel,
            u,
            num_checks=5,
    ):

        for k in range(1, num_checks + 1):

            dt = self.tau * k / num_checks

            check_pos = (
                    pos
                    + vel * dt
                    + 0.5 * u * dt ** 2
            )

            if self.profile is not None:
                self.profile["collision_samples"] += 1
                t_occ = time.perf_counter()
                occupied = self.map.is_occupied(check_pos)
                self.profile["is_occupied_time"] += time.perf_counter() - t_occ
            else:
                occupied = self.map.is_occupied(check_pos)

            if occupied:
                return False

        return True


    def search(self, start_pos, start_vel, goal_pos, goal_vel=None, timing=None):
        self._reset_profile()
        search_t0 = time.perf_counter()

        start_pos = np.asarray(start_pos, dtype=float)
        start_vel = np.asarray(start_vel, dtype=float)

        goal_pos = np.asarray(goal_pos, dtype=float)
        if goal_vel is None:
            goal_vel = np.zeros(3, dtype=float)
        else:
            goal_vel = np.asarray(goal_vel, dtype=float)
        goal_distance = float(np.linalg.norm(goal_pos - start_pos))

        start_node = Node(
            pos=start_pos,
            vel=start_vel,
            g=0.0,
            f=self.lambda_heu * self.kinodynamic_heuristic(start_pos, start_vel, goal_pos, goal_vel),
        )

        open_set = []
        heapq.heappush(open_set, start_node)
        closed_set = set()
        iterations = 0

        while open_set:
            iterations += 1
            self.profile["iterations"] = iterations
            if iterations >= self.max_iterations:
                self.profile["total_time"] = time.perf_counter() - search_t0
                self._attach_profile_timing(timing, "max_iterations")
                return None

            t_pop = time.perf_counter()
            current = heapq.heappop(open_set)
            self.profile["pop_time"] += time.perf_counter() - t_pop
            current_key = self.state_key(current.pos)

            if self.reached_goal(
                    current.pos,
                    goal_pos
            ):
                self.profile["total_time"] = time.perf_counter() - search_t0
                self._attach_profile_timing(timing, "success")
                return self.reconstruct_path(current)

            # Fast-Planner-style receding horizon: return a valid local segment
            # if the real goal is farther away than the planning horizon.
            if goal_distance > self.planning_horizon and np.linalg.norm(current.pos - start_pos) >= self.planning_horizon:
                self.profile["total_time"] = time.perf_counter() - search_t0
                self._attach_profile_timing(timing, "horizon")
                return self.reconstruct_path(current)

            closed_set.add(current_key)

            for u in self.controls: # propagate best state forward
                self.profile["children_attempted"] += 1
                t_prop = time.perf_counter()
                new_pos, new_vel = self.propagate(
                    current.pos,
                    current.vel,
                    u,
                )
                self.profile["propagate_time"] += time.perf_counter() - t_prop

                t_vel = time.perf_counter()
                if np.linalg.norm(new_vel) > self.max_vel:
                    self.profile["velocity_pruned"] += 1
                    self.profile["velocity_prune_time"] += time.perf_counter() - t_vel
                    continue
                self.profile["velocity_prune_time"] += time.perf_counter() - t_vel

                t_closed = time.perf_counter()
                key = self.state_key(new_pos)
                if key in closed_set:
                    self.profile["closed_pruned"] += 1
                    self.profile["closed_check_time"] += time.perf_counter() - t_closed
                    continue
                self.profile["closed_check_time"] += time.perf_counter() - t_closed

                t_score = time.perf_counter()
                g = current.g + self.edge_cost(u)
                h = self.kinodynamic_heuristic(new_pos, new_vel, goal_pos, goal_vel)
                f = g + self.lambda_heu * h
                self.profile["score_time"] += time.perf_counter() - t_score

                t_collision = time.perf_counter()
                if not self.primitive_collision_free(
                        current.pos,
                        current.vel,
                        u,
                ):
                    self.profile["collision_pruned"] += 1
                    self.profile["collision_time"] += time.perf_counter() - t_collision
                    continue
                self.profile["collision_time"] += time.perf_counter() - t_collision

                child = Node(
                    pos=new_pos,
                    vel=new_vel,
                    g=g,
                    f=f,
                    parent=current,
                    u=u,
                )

                t_push = time.perf_counter()
                heapq.heappush(open_set, child)
                self.profile["nodes_pushed"] += 1
                self.profile["push_time"] += time.perf_counter() - t_push

        self.profile["total_time"] = time.perf_counter() - search_t0
        self._attach_profile_timing(timing, "no_path")
        return None

    def search_waypoints(
            self,
            waypoints,
            start_vel=None,
            timing=None,
    ):
        waypoints = [
            np.asarray(p, dtype=float)
            for p in waypoints
        ]

        if start_vel is None:
            current_vel = np.zeros(3)
        else:
            current_vel = np.asarray(start_vel, dtype=float)

        full_path = []

        for i in range(len(waypoints) - 1):

            segment = self.search(
                start_pos=waypoints[i],
                start_vel=current_vel,
                goal_pos=waypoints[i + 1],
                timing=timing,
            )

            if segment is None:
                return None

            if i > 0:
                segment = segment[1:]

            full_path.extend(segment)

            # use final velocity of this segment as next start velocity
            current_vel = full_path[-1][1]

        return full_path

class BoxObstacleMap:
    def __init__(self):
        self.voxel_size = 0.2

    def is_occupied(self, pos):
        x, y, z = pos

        return (
            0.8 <= x <= 1.2 and
            -0.4 <= y <= 0.4 and
            -0.4 <= z <= 0.4
        )


import matplotlib.pyplot as plt
import numpy as np


def plot_path(path, start, goal, obstacle):

    path = np.asarray(path)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        path[:, 0],
        path[:, 1],
        path[:, 2],
        linewidth=2,
        label="trajectory"
    )

    if start is not None:
        start = np.asarray(start)
        ax.scatter(
            start[0],
            start[1],
            start[2],
            s=100,
            marker="o",
            label="start"
        )

    if goal is not None:
        goal = np.asarray(goal)
        ax.scatter(
            goal[0],
            goal[1],
            goal[2],
            s=100,
            marker="x",
            label="goal"
        )

    if obstacle is not None:
        xmin, xmax, ymin, ymax, zmin, zmax = obstacle

        xs = [xmin, xmax]
        ys = [ymin, ymax]
        zs = [zmin, zmax]

        for x in xs:
            for y in ys:
                ax.plot([x, x], [y, y], [zmin, zmax])

        for x in xs:
            for z in zs:
                ax.plot([x, x], [ymin, ymax], [z, z])

        for y in ys:
            for z in zs:
                ax.plot([xmin, xmax], [y, y], [z, z])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()

    plt.show()

if __name__ == "__main__":
    planner = KinodynamicAStar(
        occupancy_map=BoxObstacleMap(),
        goal_tol=0.1
    )

    path = planner.search(
        start_pos=np.array([0., 0., 0.]),
        start_vel=np.array([0., 0., 0.]),
        goal_pos=np.array([2., 0., 0.]),
    )
    print(path)
    plot_path(
        path,
        start=np.array([0., 0., 0.]),
        goal=np.array([2., 0., 0.]),
        obstacle=(0.8, 1.2, -0.4, 0.4, -0.4, 0.4)
    )
