import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import numpy as np
import matplotlib.pyplot as plt

class BackboardMonteCarlo:
    def __init__(
        self,
        shooter_pos=(0.0, 4.57, 2.1),   # (x, y, z) default single shooter position
        court_bounds=((-4.5, 4.5), (1.8, 7.2)), # (x_range, y_range) in meters for court-wide shooting
        hoop_pos=(0.0, 0.38, 3.048),    # Hoop center (0.38m in front of board, 10ft high)
        board_bounds=((-0.91, 0.91), (2.90, 3.96)), # NBA board x and z bounds (meters)
        restitution=0.78,               # Effective speed retention ratio ||v_out|| / ||v_in||
        g=9.81
    ):
        self.p0 = np.array(shooter_pos, dtype=np.float64)
        self.court_x, self.court_y = court_bounds
        self.ph = np.array(hoop_pos, dtype=np.float64)
        self.x_bounds, self.z_bounds = board_bounds
        self.e = restitution
        self.g = g

    def estimate_v0_for_distance(self, target_z=3.4, theta_deg=50.0):
        """
        Estimates the nominal initial speed (v0) required to hit the target height on the backboard
        given the shooter's y-distance and release height.
        """
        y_dist = float(self.p0[1])
        z0 = float(self.p0[2])
        delta_z = target_z - z0
        theta = np.radians(theta_deg)
        denom = 2.0 * (y_dist * np.tan(theta) - delta_z)
        if denom <= 0:
            return 8.0
        return float((y_dist / np.cos(theta)) * np.sqrt(self.g / denom))

    def run_monte_carlo(
        self,
        n_shots=500_000,
        random_court=False,
        court_x=None,
        court_y=None,
        v0_range=None,
        theta_deg=(42, 58),
        phi_deg=(-12, 12),
        pos_jitter=0.05,
        seed=None,
    ):
        """
        1. Simulates N parabolic trajectories toward the backboard (fixed shooter or random court positions).
        2. Filters shots that hit within the physical backboard boundaries.
        3. Solves the inverse parabolic kinematics & required surface normals.
        """
        rng = np.random.default_rng(seed)

        if random_court:
            cx = court_x if court_x is not None else self.court_x
            cy = court_y if court_y is not None else self.court_y

            x0 = rng.uniform(*cx, size=n_shots)
            y0 = rng.uniform(*cy, size=n_shots)
            z0 = rng.normal(2.10, pos_jitter, size=n_shots)

            xt = rng.uniform(self.x_bounds[0] * 0.75, self.x_bounds[1] * 0.75, size=n_shots)
            zt = rng.uniform(self.z_bounds[0] + 0.15, self.z_bounds[1] - 0.15, size=n_shots)

            dx = xt - x0
            dy = -y0
            dz = zt - z0
            dist_h = np.sqrt(dx**2 + dy**2)

            theta = np.radians(rng.uniform(45, 55, size=n_shots))
            phi = np.arctan2(dx, -dy) + np.radians(rng.normal(0, 1.8, size=n_shots))

            denom = 2.0 * (dist_h * np.tan(theta) - dz)
            valid_d = denom > 0
            if not np.any(valid_d):
                return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

            x0, y0, z0 = x0[valid_d], y0[valid_d], z0[valid_d]
            dist_h, theta, phi, denom = dist_h[valid_d], theta[valid_d], phi[valid_d], denom[valid_d]

            v0 = (dist_h / np.cos(theta)) * np.sqrt(self.g / denom) * rng.uniform(0.97, 1.03, size=len(denom))
            vx0 = v0 * np.cos(theta) * np.sin(phi)
            vy0 = -v0 * np.cos(theta) * np.cos(phi)
            vz0 = v0 * np.sin(theta)
        else:
            if v0_range is None:
                if abs(self.p0[1] - 4.57) < 0.2:
                    v0_range = (7.2, 8.8)
                else:
                    v_nom = self.estimate_v0_for_distance()
                    v0_range = (v_nom * 0.92, v_nom * 1.10)

            v0 = rng.uniform(*v0_range, size=n_shots)
            theta = np.radians(rng.uniform(*theta_deg, size=n_shots))
            phi = np.radians(rng.uniform(*phi_deg, size=n_shots))

            x0 = self.p0[0] + rng.normal(0, pos_jitter, size=n_shots)
            y0 = self.p0[1] + rng.normal(0, pos_jitter, size=n_shots)
            z0 = self.p0[2] + rng.normal(0, pos_jitter, size=n_shots)

            vx0 = v0 * np.cos(theta) * np.sin(phi)
            vy0 = -v0 * np.cos(theta) * np.cos(phi)
            vz0 = v0 * np.sin(theta)

        # Time of flight to backboard plane y = 0
        t_hit = -y0 / vy0
        xb = x0 + vx0 * t_hit
        zb = z0 + vz0 * t_hit - 0.5 * self.g * (t_hit ** 2)

        # Filter impacts that land inside the backboard rectangle
        valid = (
            (t_hit > 0) &
            (xb >= self.x_bounds[0]) & (xb <= self.x_bounds[1]) &
            (zb >= self.z_bounds[0]) & (zb <= self.z_bounds[1])
        )

        xb, zb, t_hit = xb[valid], zb[valid], t_hit[valid]
        v_in = np.column_stack([
            vx0[valid],
            vy0[valid],
            vz0[valid] - self.g * t_hit
        ])

        # Solve inverse kinematics from (xb, 0, zb) -> hoop (0, yh, zh)
        dx = self.ph[0] - xb
        dy = np.full_like(xb, self.ph[1])
        dz = self.ph[2] - zb

        v_in_mag = np.linalg.norm(v_in, axis=1)
        v_out_mag = self.e * v_in_mag

        # Quadratic in u = t_bounce^2: 0.25*g^2*u^2 + (g*dz - v_out^2)*u + (dx^2 + dy^2 + dz^2) = 0
        A_q = 0.25 * (self.g ** 2)
        B_q = self.g * dz - (v_out_mag ** 2)
        C_q = dx**2 + dy**2 + dz**2

        discriminant = B_q**2 - 4 * A_q * C_q
        reachable = discriminant >= 0

        # Filter to physically reachable bounces
        xb, zb = xb[reachable], zb[reachable]
        dx, dy, dz = dx[reachable], dy[reachable], dz[reachable]
        v_in = v_in[reachable]
        B_q, discriminant = B_q[reachable], discriminant[reachable]

        # Take the shorter bounce time root u_minus (direct bank) or u_plus (high arc)
        u_root = (-B_q - np.sqrt(discriminant)) / (2 * A_q)
        pos_time = u_root > 1e-4

        xb, zb = xb[pos_time], zb[pos_time]
        dx, dy, dz = dx[pos_time], dy[pos_time], dz[pos_time]
        v_in = v_in[pos_time]
        tb = np.sqrt(u_root[pos_time])

        # Required outgoing velocity vector to hit dead-center of the hoop
        v_out = np.column_stack([
            dx / tb,
            dy / tb,
            (dz + 0.5 * self.g * (tb ** 2)) / tb
        ])

        # Required unit normal n = (v_out - v_in) / ||v_out - v_in||
        delta_v = v_out - v_in
        normals = delta_v / np.linalg.norm(delta_v, axis=1, keepdims=True)

        return xb, zb, v_in, v_out, normals

    def generate_trajectories(
        self,
        n_shots=35,
        random_court=False,
        court_x=None,
        court_y=None,
        v0_range=None,
        theta_deg=(42, 58),
        phi_deg=(-12, 12),
        pos_jitter=0.04,
        pts_per_arc=60,
        only_hits=True,
        seed=None,
    ):
        """
        Simulates individual shot trajectory paths (from a fixed shooter or random court positions).
        If `only_hits=True` (default), collects and returns only shots that successfully hit
        within the backboard boundaries.
        Returns a list of dicts with full 3D trajectory data:
          - incoming arc: (x, y, z) coordinates from release to impact
          - bounce arc: (x, y, z) coordinates from impact to hoop (if reachable)
          - hit_board: whether impact was within backboard bounds
          - banked: whether bounce successfully reaches hoop
          - impact_point: (xb, 0, zb)
          - tb: bounce flight time
          - v_in: incoming velocity vector
          - v_out: outgoing velocity vector
          - normal: surface normal at impact
        """
        if not random_court and v0_range is None:
            if abs(self.p0[1] - 4.57) < 0.2:
                v0_range = (7.2, 8.8)
            else:
                v_nom = self.estimate_v0_for_distance()
                v0_range = (v_nom * 0.92, v_nom * 1.10)

        rng = np.random.default_rng(seed)
        trajectories = []
        max_attempts = n_shots * 150
        attempt = 0

        cx = court_x if court_x is not None else self.court_x
        cy = court_y if court_y is not None else self.court_y

        while len(trajectories) < n_shots and attempt < max_attempts:
            attempt += 1

            if random_court:
                # Random shooter position on the court
                x0 = float(rng.uniform(*cx))
                y0 = float(rng.uniform(*cy))
                z0 = float(rng.normal(2.10, pos_jitter))

                # Aim at backboard target zone
                xt = float(rng.uniform(self.x_bounds[0] * 0.75, self.x_bounds[1] * 0.75))
                zt = float(rng.uniform(self.z_bounds[0] + 0.15, self.z_bounds[1] - 0.15))

                dx = xt - x0
                dy = -y0
                dz = zt - z0
                dist_h = np.sqrt(dx**2 + dy**2)

                theta = float(np.radians(rng.uniform(45, 55)))
                phi = float(np.arctan2(dx, -dy) + np.radians(rng.normal(0, 1.8)))

                denom = 2.0 * (dist_h * np.tan(theta) - dz)
                if denom <= 0:
                    continue
                v0 = float((dist_h / np.cos(theta)) * np.sqrt(self.g / denom) * rng.uniform(0.97, 1.03))
            else:
                v0 = float(rng.uniform(*v0_range))
                theta = float(np.radians(rng.uniform(*theta_deg)))
                phi = float(np.radians(rng.uniform(*phi_deg)))

                x0 = float(self.p0[0] + rng.normal(0, pos_jitter))
                y0 = float(self.p0[1] + rng.normal(0, pos_jitter))
                z0 = float(self.p0[2] + rng.normal(0, pos_jitter))

            vx0 = v0 * np.cos(theta) * np.sin(phi)
            vy0 = -v0 * np.cos(theta) * np.cos(phi)
            vz0 = v0 * np.sin(theta)

            if abs(vy0) < 1e-5:
                continue

            t_hit = -y0 / vy0
            if t_hit <= 0:
                continue

            xb = x0 + vx0 * t_hit
            zb = z0 + vz0 * t_hit - 0.5 * self.g * (t_hit ** 2)

            hit_board = (
                (xb >= self.x_bounds[0]) and (xb <= self.x_bounds[1]) and
                (zb >= self.z_bounds[0]) and (zb <= self.z_bounds[1])
            )

            if only_hits and not hit_board:
                continue

            traj_dict = {
                "shot_id": len(trajectories),
                "p0": np.array([x0, y0, z0]),
                "v0_vec": np.array([vx0, vy0, vz0]),
                "v0_mag": v0,
                "theta_deg": float(np.degrees(theta)),
                "phi_deg": float(np.degrees(phi)),
                "t_hit": float(t_hit),
                "hit_board": bool(hit_board),
                "banked": False,
                "impact_point": np.array([xb, 0.0, zb]),
                "t_bounce": None,
                "v_in": None,
                "v_out": None,
                "normal": None,
                "x_in": None,
                "y_in": None,
                "z_in": None,
                "x_out": None,
                "y_out": None,
                "z_out": None,
            }

            if hit_board:
                t_in = np.linspace(0, t_hit, pts_per_arc)
                x_in = x0 + vx0 * t_in
                y_in = y0 + vy0 * t_in
                z_in = z0 + vz0 * t_in - 0.5 * self.g * (t_in ** 2)
                traj_dict["x_in"] = x_in
                traj_dict["y_in"] = y_in
                traj_dict["z_in"] = z_in

                v_in = np.array([vx0, vy0, vz0 - self.g * t_hit])
                traj_dict["v_in"] = v_in

                dx = self.ph[0] - xb
                dy = self.ph[1]
                dz = self.ph[2] - zb
                v_in_mag = np.linalg.norm(v_in)
                v_out_mag = self.e * v_in_mag

                A_q = 0.25 * (self.g ** 2)
                B_q = self.g * dz - (v_out_mag ** 2)
                C_q = dx**2 + dy**2 + dz**2
                disc = B_q**2 - 4 * A_q * C_q

                if disc >= 0:
                    u_root = (-B_q - np.sqrt(disc)) / (2 * A_q)
                    if u_root > 1e-4:
                        tb = float(np.sqrt(u_root))
                        v_out = np.array([
                            dx / tb,
                            dy / tb,
                            (dz + 0.5 * self.g * (tb ** 2)) / tb
                        ])
                        delta_v = v_out - v_in
                        normal = delta_v / np.linalg.norm(delta_v)

                        t_out = np.linspace(0, tb, pts_per_arc)
                        x_out = xb + v_out[0] * t_out
                        y_out = 0.0 + v_out[1] * t_out
                        z_out = zb + v_out[2] * t_out - 0.5 * self.g * (t_out ** 2)

                        traj_dict["banked"] = True
                        traj_dict["t_bounce"] = tb
                        traj_dict["v_out"] = v_out
                        traj_dict["normal"] = normal
                        traj_dict["x_out"] = x_out
                        traj_dict["y_out"] = y_out
                        traj_dict["z_out"] = z_out
            else:
                # Shot missed the backboard bounds - continue trajectory slightly past y=0
                t_max = min(t_hit * 1.25, t_hit + 0.25)
                t_in = np.linspace(0, t_max, pts_per_arc)
                x_in = x0 + vx0 * t_in
                y_in = y0 + vy0 * t_in
                z_in = z0 + vz0 * t_in - 0.5 * self.g * (t_in ** 2)
                mask = z_in >= 0
                if np.any(mask):
                    idx = np.where(mask)[0]
                    last = idx[-1] + 1
                    traj_dict["x_in"] = x_in[:last]
                    traj_dict["y_in"] = y_in[:last]
                    traj_dict["z_in"] = z_in[:last]
                else:
                    traj_dict["x_in"] = x_in
                    traj_dict["y_in"] = y_in
                    traj_dict["z_in"] = z_in

            trajectories.append(traj_dict)

        return trajectories

    @staticmethod
    def fit_polynomial_surface(xb, zb, normals, degree=4, z_ref=3.048):
        """
        Builds the design matrix A and solves A * c = b for the 3D surface y = f(x, z - z_ref)
        whose partial derivatives match (-nx/ny, -nz/ny) in the least-squares sense.
        """
        # Target partial derivatives: df/dx = -nx/ny, df/dz = -nz/ny
        df_dx_target = -normals[:, 0] / normals[:, 1]
        df_dz_target = -normals[:, 2] / normals[:, 1]

        # Shift z relative to hoop height for numerical conditioning
        zs = zb - z_ref
        xs = xb

        # Enumerate polynomial basis pairs (i, j) for x^i * z^j excluding constant (0,0)
        exponents = [
            (i, j)
            for total_deg in range(1, degree + 1)
            for i in range(total_deg + 1)
            for j in [total_deg - i]
        ]

        n_pts = len(xs)
        n_terms = len(exponents)
        A_x = np.zeros((n_pts, n_terms))
        A_z = np.zeros((n_pts, n_terms))

        for col, (i, j) in enumerate(exponents):
            if i > 0:
                A_x[:, col] = i * (xs ** (i - 1)) * (zs ** j)
            if j > 0:
                A_z[:, col] = j * (xs ** i) * (zs ** (j - 1))

        # Stack [A_x; A_z] * c = [df_dx; df_dz]
        A = np.vstack([A_x, A_z])
        b = np.concatenate([df_dx_target, df_dz_target])

        # Solve normal equations via SVD/QR least squares
        coeffs, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return coeffs, exponents

    @staticmethod
    def evaluate_surface(X, Z, coeffs, exponents, z_ref=3.048):
        """Evaluates surface depth Y = f(X, Z) and its unit normals on a grid."""
        Zs = Z - z_ref
        Y = np.zeros_like(X, dtype=np.float64)
        dY_dx = np.zeros_like(X, dtype=np.float64)
        dY_dz = np.zeros_like(X, dtype=np.float64)

        for c, (i, j) in zip(coeffs, exponents):
            Y += c * (X ** i) * (Zs ** j)
            if i > 0:
                dY_dx += c * i * (X ** (i - 1)) * (Zs ** j)
            if j > 0:
                dY_dz += c * j * (X ** i) * (Zs ** (j - 1))

        # Normal vector [-df/dx, 1, -df/dz] normalized
        norm = np.sqrt(dY_dx**2 + 1.0 + dY_dz**2)
        nx, ny, nz = -dY_dx / norm, 1.0 / norm, -dY_dz / norm
        return Y, (nx, ny, nz)


if __name__ == "__main__":
    sim = BackboardMonteCarlo()
    xb, zb, v_in, v_out, normals = sim.run_monte_carlo(n_shots=500_000)
    print(f"Valid backboard impacts: {len(xb):,}")

    coeffs, exponents = sim.fit_polynomial_surface(xb, zb, normals, degree=4)
    for c, (i, j) in zip(coeffs, exponents):
        print(f"c_(x^{i} z^{j}) = {c:+.5f}")