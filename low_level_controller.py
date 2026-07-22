import numpy as np


IRIS_MASS = 1.5
IRIS_MAXS_ACC = 17.78 #estimated
IRIS_INERTIA = np.diag([0.0291249994, 0.0291249994, 0.0552249998])
IRIS_KF = 8.54858e-6
IRIS_KM = 1e-6
IRIS_MIN_ROTOR_SPEED = 0.0
IRIS_MAX_ROTOR_SPEED = 1100.0
IRIS_ROTOR_DIRECTIONS = np.array([-1.0, -1.0, 1.0, 1.0], dtype=float)
IRIS_ROTOR_POSITIONS = np.array(
    [
        [0.1379854530, -0.2067163885, 0.0230000000],
        [-0.1251116842, 0.2187541425, 0.0230000000],
        [0.1379999965, 0.2025757730, 0.0230000000],
        [-0.1241535991, -0.2223457992, 0.0230000000],
    ],
    dtype=float,
)


def normalize(v, eps=1e-8):
    norm = np.linalg.norm(v)
    if norm < eps:
        return np.zeros_like(v)
    return v / norm


def vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def quat_wxyz_to_matrix(quat):
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def yaw_from_R(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _iris_wrench_limits_from_rotors(
    rotor_positions=IRIS_ROTOR_POSITIONS,
    kf=IRIS_KF,
    km=IRIS_KM,
    rotor_directions=IRIS_ROTOR_DIRECTIONS,
    omega_max=IRIS_MAX_ROTOR_SPEED,
):
    rotor_positions = np.asarray(rotor_positions, dtype=float)
    x = rotor_positions[:, 0]
    y = rotor_positions[:, 1]
    rotor_directions = np.asarray(rotor_directions, dtype=float)

    omega2 = float(omega_max) ** 2

    f_max = 4.0 * float(kf) * omega2

    # Axis-wise symmetric maxima implied by independent rotor saturation bounds.
    # These are conservative "component" bounds; simultaneous (F, tau) feasibility
    # is a smaller polytope.
    tx_pos = float(np.sum(y[y > 0.0])) * float(kf) * omega2
    tx_neg = float(np.sum(-y[y < 0.0])) * float(kf) * omega2
    tx_max = min(tx_pos, tx_neg)

    ty_pos = float(np.sum(-x[x < 0.0])) * float(kf) * omega2  # (-x) positive when x < 0
    ty_neg = float(np.sum(x[x > 0.0])) * float(kf) * omega2   # (-x) negative when x > 0
    ty_max = min(ty_pos, ty_neg)

    tz_pos = float(np.sum(rotor_directions[rotor_directions > 0.0])) * float(km) * omega2
    tz_neg = float(np.sum(-rotor_directions[rotor_directions < 0.0])) * float(km) * omega2
    tz_max = min(tz_pos, tz_neg)

    return float(f_max), np.array([tx_max, ty_max, tz_max], dtype=float)


IRIS_MAX_THRUST, IRIS_MAX_TORQUE = _iris_wrench_limits_from_rotors()


class Drone:
    def __init__(
        self,
        body,
        articulation=None,
        m=IRIS_MASS,
        J=IRIS_INERTIA,
        g=9.81,
        rotor_joint_names=("joint0", "joint1", "joint2", "joint3"),
    ):
        self.body = body
        self.articulation = articulation
        self.m = float(m)
        self.J = np.asarray(J, dtype=float)
        self.J_inv = np.linalg.inv(self.J)
        self.g = float(g)
        self.rotor_joint_names = tuple(rotor_joint_names)
        self.rotor_joint_indices = self._resolve_rotor_joint_indices()
        self.last_applied_wrench = np.zeros(4, dtype=float)

    def get_state(self):
        positions, orientations = self.body.get_world_poses()
        linear_velocities = self.body.get_linear_velocities()
        angular_velocities = self.body.get_angular_velocities()

        pos = np.asarray(positions[0], dtype=float)
        vel = np.asarray(linear_velocities[0], dtype=float)
        quat = np.asarray(orientations[0], dtype=float)
        R = quat_wxyz_to_matrix(quat)

        omega_world = np.asarray(angular_velocities[0], dtype=float)
        omega_body = R.T @ omega_world

        return {
            "pos": pos,
            "vel": vel,
            "R": R,
            "omega": omega_body,
            "yaw": yaw_from_R(R),
        }

    def apply_control(self, u):
        actual_thrust = float(u[0])
        actual_torque_body = np.asarray(u[1:4], dtype=float)

        state = self.get_state()
        R = state["R"]
        zB_world = R[:, 2]

        force_world = actual_thrust * zB_world
        torque_world = R @ actual_torque_body

        forces = force_world.reshape(1, 3).astype(np.float32)
        torques = torque_world.reshape(1, 3).astype(np.float32)
        self.body.apply_forces_and_torques_at_pos(forces=forces, torques=torques, is_global=True)

        self.last_applied_wrench = np.hstack([actual_thrust, actual_torque_body])

    def _resolve_rotor_joint_indices(self):
        if self.articulation is None:
            return None

        indices = []
        for name in self.rotor_joint_names:
            try:
                indices.append(self.articulation.get_dof_index(name))
            except Exception:
                return None
        return np.asarray(indices, dtype=np.int32)


class LowLevelController:
    def __init__(
        self,
        drone,
        omegap=2.0,
        omegaR=8.0,
        max_thrust=None,
        max_torque=None,
    ):
        self.drone = drone

        self.Kp = drone.m * omegap**2 * np.eye(3)
        self.Kv = 2.0 * drone.m * omegap * np.eye(3)
        self.Kr = drone.J * omegaR**2
        self.Kw = 2.0 * drone.J * omegaR

        self.max_thrust = float(max_thrust) if max_thrust is not None else float(IRIS_MAX_THRUST)
        if max_torque is None:
            self.max_torque = np.asarray(IRIS_MAX_TORQUE, dtype=float)
        else:
            self.max_torque = np.asarray(max_torque, dtype=float)

    def PDControlLaw(self, sigma, sigmad, sigmadd, sigmaddd):
        state = self.drone.get_state()
        pos = state["pos"]
        vel = state["vel"]
        R = state["R"]
        omega = state["omega"]

        ep = pos - sigma[0:3]
        ev = vel - sigmad[0:3]

        e3 = np.array([0.0, 0.0, 1.0])
        Fdes = -self.Kp @ ep - self.Kv @ ev + self.drone.m * self.drone.g * e3 + self.drone.m * sigmadd[0:3]

        zB = R[:, 2]
        u1 = np.dot(Fdes, zB)
        u1 = np.clip(u1, 0.0, self.max_thrust)

        zBdes = normalize(Fdes)
        if np.linalg.norm(zBdes) < 1e-8:
            zBdes = e3

        xCdes = np.array([np.cos(sigma[3]), np.sin(sigma[3]), 0.0])
        yBdes = normalize(np.cross(zBdes, xCdes))
        if np.linalg.norm(yBdes) < 1e-8:
            yBdes = np.array([0.0, 1.0, 0.0])
        xBdes = np.cross(yBdes, zBdes)

        Rdes = np.column_stack((xBdes, yBdes, zBdes))
        E = 0.5 * (Rdes.T @ R - R.T @ Rdes)
        eR = vee(E)

        a_dot = sigmaddd[0:3]
        thrust_for_reference = max(abs(u1), 1e-6)
        h = (self.drone.m / thrust_for_reference) * (a_dot - (zBdes @ a_dot) * zBdes)

        p_des = -h @ yBdes
        q_des = h @ xBdes
        r_des = sigmad[3] * (e3 @ zBdes)
        omega_des = np.array([p_des, q_des, r_des])

        ew = omega - omega_des
        torque = -self.Kr @ eR - self.Kw @ ew + np.cross(omega, self.drone.J @ omega)
        torque = np.clip(torque, -self.max_torque, self.max_torque)

        u = np.hstack([u1, torque])
        debug = {
            "ep": ep,
            "ev": ev,
            "eR": eR,
            "ew": ew,
            "Fdes": Fdes,
            "Rdes": Rdes,
        }

        return u, debug
