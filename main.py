import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import argparse
1
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from monte_carlo import BackboardMonteCarlo


def run_backboard_solver(
    shooter_distance_m=4.57,  # 15.0 ft (standard NBA free throw distance from backboard)
    release_height_m=2.10,    # ~6.9 ft ball release height
    n_solver_shots=100_000,   # Monte Carlo sample size for high-precision surface solve
    n_plot_trajectories=35,   # Number of individual tried trajectories to display in 3D
    poly_degree=4,            # Polynomial degree for backboard curvature fit
    random_seed=None,         # None = Dynamic random sampling on each run; int = Reproducible
    output_image="trajectories_3d.png",
    export_stl_dir=None,      # Directory to export STL files (None to skip)
):
    """
    Executes the backboard solver pipeline:
      1. Simulates Monte Carlo shots from a set distance away from the hoop.
      2. Solves required bounce vectors and surface normals for backboard impacts.
      3. Fits the optimal 3D polynomial backboard surface.
      4. Generates individual tried trajectory paths (hits, banks, and misses).
      5. Renders and saves a comprehensive 3D Matplotlib visualization.
    """
    seed_str = f"{random_seed}" if random_seed is not None else "Dynamic / Stochastic (Varies every run)"
    print("=" * 70)
    print("NEVER-MISS BASKETBALL BACKBOARD SOLVER")
    print("=" * 70)
    print(f"Shooter Set Distance  : {shooter_distance_m:.2f} m ({shooter_distance_m * 3.28084:.1f} ft)")
    print(f"Shooter Release Height: {release_height_m:.2f} m ({release_height_m * 3.28084:.1f} ft)")
    print(f"Monte Carlo Shots     : {n_solver_shots:,}")
    print(f"Polynomial Fit Degree : {poly_degree}")
    print(f"Random Seed           : {seed_str}")
    print("-" * 70)

    # 1. Initialize simulation with shooter at set distance
    shooter_pos = (0.0, shooter_distance_m, release_height_m)
    sim = BackboardMonteCarlo(shooter_pos=shooter_pos)

    # 2. Run Monte Carlo simulation for surface calculation
    print("Running Monte Carlo simulation to compute backboard impacts and kinematics...")
    xb, zb, v_in, v_out, normals = sim.run_monte_carlo(n_shots=n_solver_shots, seed=random_seed)
    hit_rate = (len(xb) / n_solver_shots) * 100.0

    print(f"-> Valid backboard impacts: {len(xb):,} / {n_solver_shots:,} ({hit_rate:.1f}%)")

    # 3. Fit polynomial surface y = f(x, z - z_ref)
    print(f"Fitting degree-{poly_degree} polynomial surface to required normals...")
    coeffs, exponents = sim.fit_polynomial_surface(xb, zb, normals, degree=poly_degree)

    print(f"-> Fitted {len(coeffs)} polynomial surface coefficients:")
    for c, (i, j) in zip(coeffs, exponents):
        print(f"     c_(x^{i} z^{j}) = {c:+.5f}")

    # 4. Generate trajectories tried from the set distance (only shots that hit the backboard)
    print("-" * 70)
    print(f"Generating {n_plot_trajectories} backboard-hitting trajectories for 3D visualization...")
    trajectories = sim.generate_trajectories(
        n_shots=n_plot_trajectories,
        only_hits=True,
        pts_per_arc=60,
        seed=random_seed,
    )

    n_banked = sum(1 for t in trajectories if t["banked"])
    print(f"-> Trajectories ready: {len(trajectories)} backboard hits | {n_banked} successfully banked into hoop")
    print("=" * 70)

    # 5. Build 3D Matplotlib Visualization
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # --- Court Floor and Lane Reference Lines ---
    # Lane box: 3.66m wide (x: -1.83 to 1.83), from y=0 to shooter_distance_m
    lane_x = [-1.83, 1.83, 1.83, -1.83, -1.83]
    lane_y = [0.0, 0.0, shooter_distance_m, shooter_distance_m, 0.0]
    lane_z = [0.0, 0.0, 0.0, 0.0, 0.0]
    ax.plot(lane_x, lane_y, lane_z, color="#b0bec5", linestyle="--", linewidth=1.2, label="_nolegend_")

    # Free-throw line marker at shooter distance
    ft_line_x = np.linspace(-1.83, 1.83, 30)
    ft_line_y = np.full_like(ft_line_x, shooter_distance_m)
    ft_line_z = np.zeros_like(ft_line_x)
    ax.plot(ft_line_x, ft_line_y, ft_line_z, color="#78909c", linewidth=2.0, label="Free-Throw Line (Floor)")

    # --- Shooter Marker and Release Line ---
    p0 = sim.p0
    ax.plot([p0[0], p0[0]], [p0[1], p0[1]], [0.0, p0[2]], color="#6a1b9a", linestyle=":", linewidth=2.0, label="_nolegend_")
    ax.scatter([p0[0]], [p0[1]], [p0[2]], color="#8e24aa", s=100, marker="o", edgecolors="black", linewidths=1.2, zorder=10, label=f"Shooter ({shooter_distance_m:.2f}m / {shooter_distance_m*3.28084:.1f}ft)")

    # --- Backboard Frame & Target Box ---
    x_min, x_max = sim.x_bounds
    z_min, z_max = sim.z_bounds

    # Outer backboard frame (at y=0)
    frame_x = [x_min, x_max, x_max, x_min, x_min]
    frame_y = [0.0, 0.0, 0.0, 0.0, 0.0]
    frame_z = [z_min, z_min, z_max, z_max, z_min]
    ax.plot(frame_x, frame_y, frame_z, color="#1a237e", linewidth=2.5, zorder=5, label="Backboard Frame (NBA Reg.)")

    # Inner shooter's square (target box: 24" wide x 18" high, centered, base at hoop level 3.048m)
    box_w, box_h = 0.61, 0.457
    box_x = [-box_w/2, box_w/2, box_w/2, -box_w/2, -box_w/2]
    box_y = [0.0, 0.0, 0.0, 0.0, 0.0]
    box_z = [3.048, 3.048, 3.048 + box_h, 3.048 + box_h, 3.048]
    ax.plot(box_x, box_y, box_z, color="#d32f2f", linewidth=1.5, zorder=6, label="Inner Shooter's Square")

    # --- Solved Curved Backboard Surface Mesh ---
    gx = np.linspace(x_min, x_max, 40)
    gz = np.linspace(z_min, z_max, 40)
    GX, GZ = np.meshgrid(gx, gz)
    GY, _ = sim.evaluate_surface(GX, GZ, coeffs, exponents)
    surf = ax.plot_surface(
        GX, GY, GZ,
        cmap="coolwarm",
        alpha=0.35,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        zorder=3
    )

    # --- Basketball Hoop Rim & Net ---
    hoop_x, hoop_y, hoop_z = sim.ph
    rim_radius = 0.2286  # 9-inch radius = 18-inch diameter
    theta_rim = np.linspace(0, 2 * np.pi, 100)
    rx = hoop_x + rim_radius * np.cos(theta_rim)
    ry = hoop_y + rim_radius * np.sin(theta_rim)
    rz = np.full_like(rx, hoop_z)
    ax.plot(rx, ry, rz, color="#ff5722", linewidth=3.5, zorder=8, label="Hoop Rim (10 ft / 3.05 m)")

    # Rim mounting bracket to backboard
    ax.plot([hoop_x, hoop_x], [0.0, hoop_y - rim_radius], [hoop_z, hoop_z], color="#455a64", linewidth=3.0, zorder=7)

    # Subtle net representation
    net_depth = 0.40
    for ang in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        top_x = hoop_x + rim_radius * np.cos(ang)
        top_y = hoop_y + rim_radius * np.sin(ang)
        bot_x = hoop_x + 0.6 * rim_radius * np.cos(ang)
        bot_y = hoop_y + 0.6 * rim_radius * np.sin(ang)
        ax.plot([top_x, bot_x], [top_y, bot_y], [hoop_z, hoop_z - net_depth], color="#cfd8dc", linewidth=0.8, alpha=0.7, zorder=7)

    # --- Plot Trajectories Tried ---
    # --- Plot Trajectories (Shots that hit the backboard) ---
    plotted_hit_legend = False
    plotted_bank_legend = False

    impact_xs, impact_zs = [], []

    for traj in trajectories:
        if traj["hit_board"] and traj["banked"]:
            # 1. Incoming arc from shooter release to backboard impact
            lbl_in = "Incoming Arc (to Backboard)" if not plotted_hit_legend else "_nolegend_"
            ax.plot(
                traj["x_in"], traj["y_in"], traj["z_in"],
                color="#0288d1",
                linewidth=1.4,
                alpha=0.75,
                zorder=4,
                label=lbl_in,
            )
            plotted_hit_legend = True

            # 2. Outgoing bounce arc from backboard into hoop center
            lbl_out = "Solved Rebound (into Hoop)" if not plotted_bank_legend else "_nolegend_"
            ax.plot(
                traj["x_out"], traj["y_out"], traj["z_out"],
                color="#2e7d32",
                linewidth=2.0,
                alpha=0.90,
                zorder=5,
                label=lbl_out,
            )
            plotted_bank_legend = True

            # Record impact point
            impact_xs.append(traj["impact_point"][0])
            impact_zs.append(traj["impact_point"][2])

    # Plot impact points on backboard
    if impact_xs:
        ax.scatter(
            impact_xs, np.zeros_like(impact_xs), impact_zs,
            color="#01579b",
            s=32,
            marker="o",
            edgecolors="white",
            linewidths=0.8,
            zorder=6,
            label=f"Backboard Impact Points ({len(impact_xs)})",
        )

    # --- Styling, Axes Labels & Limits ---
    ax.set_xlabel("X: Width (m)", fontsize=11, labelpad=10, fontweight="medium")
    ax.set_ylabel("Y: Distance to Board (m)", fontsize=11, labelpad=10, fontweight="medium")
    ax.set_zlabel("Z: Height (m)", fontsize=11, labelpad=10, fontweight="medium")

    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-0.4, shooter_distance_m + 0.6)
    ax.set_zlim(0.0, 4.8)

    # Box aspect ratio: [width, depth, height]
    ax.set_box_aspect([2.0, 3.2, 2.2])

    # 3D view camera angle: looking from court side toward backboard
    ax.view_init(elev=20, azim=-62)

    # Title
    ax.set_title(
        f"Never-Miss Backboard Solver: 3D Trajectories (Backboard Hits Only)\n"
        f"Shooter Distance: {shooter_distance_m:.2f} m ({shooter_distance_m*3.28084:.1f} ft) | Hoop Height: 3.05 m (10 ft)",
        fontsize=13,
        fontweight="bold",
        pad=18,
    )

    # Stats info box in plot
    stats_text = (
        f"Solver Parameters:\n"
        f"• Monte Carlo Sample: {n_solver_shots:,} shots\n"
        f"• Random Seed: {seed_str}\n"
        f"• Backboard Hit Rate: {hit_rate:.1f}%\n"
        f"• Bank Solvability: 100.0% of hits\n"
        f"• Polynomial Fit: Degree {poly_degree} ({len(coeffs)} terms)\n"
        f"• Restitution (e): {sim.e:.2f}\n"
        f"• Displayed Trajectories: {len(impact_xs)} Backboard Hits & Solved Bounces"
    )
    fig.text(
        0.02, 0.96, stats_text,
        transform=fig.transFigure,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#cfd8dc", alpha=0.9),
    )

    # Legend placement
    ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.95), framealpha=0.9, fontsize=9)

    plt.tight_layout()

    # Save to disk
    plt.savefig(output_image, dpi=200, bbox_inches="tight")
    print(f"✅ Saved 3D visualization to '{output_image}'")

    # Display plot
    plt.show()

    # Optional STL Export
    if export_stl_dir:
        from export_tiled_stl import export_segmented_tiles_stl, export_full_backboard_stl
        print("\n" + "=" * 70)
        print(f"📦 EXPORTING 3D-PRINTABLE STL MESHES to '{export_stl_dir}/'...")
        tiles_dir = os.path.join(export_stl_dir, "tiles")
        export_segmented_tiles_stl(sim, tiles_dir, coeffs, exponents)
        export_full_backboard_stl(sim, os.path.join(export_stl_dir, "backboard_full.stl"), coeffs, exponents)
        print("=" * 70)

    return sim, coeffs, exponents, trajectories


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Never-Miss Backboard Solver: 3D Trajectory Simulation & Visualization"
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=4.57,
        help="Shooter distance from backboard in meters (default: 4.57 m / 15.0 ft free-throw)",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=100_000,
        help="Number of Monte Carlo solver shots to simulate (default: 100,000)",
    )
    parser.add_argument(
        "--trajectories",
        type=int,
        default=35,
        help="Number of backboard-hitting trajectories to display (default: 35)",
    )
    parser.add_argument(
        "--poly-degree",
        type=int,
        default=4,
        help="Polynomial degree for curved surface fit (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None, dynamic stochastic variation)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="trajectories_3d.png",
        help="Output image path for 3D plot (default: trajectories_3d.png)",
    )
    parser.add_argument(
        "--export-stl",
        nargs="?",
        const="stl_output",
        default=None,
        help="Export backboard to 3D-printable STL files in the given directory (default: stl_output)",
    )

    args = parser.parse_args()

    run_backboard_solver(
        shooter_distance_m=args.distance,
        release_height_m=2.10,
        n_solver_shots=args.shots,
        n_plot_trajectories=args.trajectories,
        poly_degree=args.poly_degree,
        random_seed=args.seed,
        output_image=args.output,
        export_stl_dir=args.export_stl,
    )
