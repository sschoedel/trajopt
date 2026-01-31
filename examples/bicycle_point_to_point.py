"""
Example: Point-to-point trajectory optimization for bicycle model.

Demonstrates using the trajectory optimizer to find optimal controls for
driving from point A to point B while minimizing control effort.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try Pinocchio-based model first, fall back to hand-written model
try:
    from models.bicycle_model_pinocchio import BicycleModelPinocchio as BicycleModel
    print("Using Pinocchio-based bicycle model")
    USE_PINOCCHIO = True
except ImportError as e:
    print(f"Pinocchio not available ({e}), using hand-written model")
    from models.bicycle_model import BicycleModel
    USE_PINOCCHIO = False

from trajopt_solver import TrajectoryOptimizer
from visualize_mujoco import visualize_trajectory


def plot_trajectory(times, states, controls, model):
    """
    Visualize the optimized trajectory.

    Args:
        times: Time array
        states: State trajectory (4 x N+1)
        controls: Control trajectory (2 x N)
        model: BicycleModel instance
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: X-Y trajectory
    ax = axes[0, 0]
    ax.plot(states[0, :], states[1, :], 'b-', linewidth=2, label='Trajectory')
    ax.plot(states[0, 0], states[1, 0], 'go', markersize=10, label='Start')
    ax.plot(states[0, -1], states[1, -1], 'ro', markersize=10, label='Goal')

    # Draw vehicle orientation at key points
    num_arrows = 10
    indices = np.linspace(0, len(times)-1, num_arrows, dtype=int)
    for idx in indices:
        x, y, theta = states[0, idx], states[1, idx], states[2, idx]
        dx = 0.5 * np.cos(theta)
        dy = 0.5 * np.sin(theta)
        ax.arrow(x, y, dx, dy, head_width=0.3, head_length=0.2,
                fc='gray', ec='gray', alpha=0.5)

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Vehicle Trajectory')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')

    # Plot 2: States over time
    ax = axes[0, 1]
    ax.plot(times, states[0, :], label='x (m)')
    ax.plot(times, states[1, :], label='y (m)')
    ax.plot(times, states[2, :], label='θ (rad)')
    ax.plot(times, states[3, :], label='α (rad)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('State')
    ax.set_title('States vs Time')
    ax.legend()
    ax.grid(True)

    # Plot 3: Controls over time
    ax = axes[1, 0]
    # Controls are piecewise constant, so repeat last value for plotting
    times_control = times[:-1]
    ax.step(times_control, controls[0, :], where='post', label='v (m/s)', linewidth=2)
    ax.step(times_control, controls[1, :], where='post', label='α̇ (rad/s)', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control')
    ax.set_title('Controls vs Time')
    ax.legend()
    ax.grid(True)

    # Plot 4: Control effort over time
    ax = axes[1, 1]
    dt = times[1] - times[0]
    effort = controls[0, :]**2 + controls[1, :]**2
    cumulative_effort = np.cumsum(effort) * dt
    ax.plot(times_control, cumulative_effort, 'r-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cumulative Control Effort')
    ax.set_title(f'Total Effort: {cumulative_effort[-1]:.2f}')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('results/trajectory_result.png', dpi=150)
    print("Plot saved to results/trajectory_result.png")
    plt.show()


def main():
    """
    Run point-to-point trajectory optimization example.
    """
    print("=" * 60)
    print("Trajectory Optimization for Bicycle Model")
    print("Point-to-Point Navigation Example")
    print("=" * 60)
    print()

    # Define bicycle parameters
    L = 2.5  # wheelbase (m)
    m = 1000.0  # mass (kg)

    print(f"Vehicle parameters:")
    print(f"  Wheelbase L = {L} m")
    print(f"  Mass m = {m} kg")
    print()

    # Create bicycle model
    if USE_PINOCCHIO:
        model = BicycleModel(model_path='robots/bicycle.urdf', L=L, m=m)
    else:
        model = BicycleModel(L=L, m=m)

    # Define start and goal states
    x_start = np.array([0.0, 0.0, 0.0, 0.0])  # origin, heading east, no steering
    x_goal = np.array([10.0, 5.0, np.pi/4, 0.0])  # 10m forward, 5m right, heading NE

    print(f"Start state: x={x_start[0]:.1f}, y={x_start[1]:.1f}, "
          f"θ={x_start[2]:.2f}, α={x_start[3]:.2f}")
    print(f"Goal state:  x={x_goal[0]:.1f}, y={x_goal[1]:.1f}, "
          f"θ={x_goal[2]:.2f}, α={x_goal[3]:.2f}")
    print()

    # Optimization parameters
    N = 100  # number of collocation points
    T = 15.0  # time horizon (seconds) - longer for 3-point turn
    w_v = 0.1  # velocity cost weight - lower to encourage movement
    w_alpha = 0.1  # steering rate cost weight

    # Create cost matrix (control_dim x control_dim)
    W = np.diag([w_v, w_alpha])  # Cost on [v, alphadot]

    # Control bounds
    u_lower = np.array([-10.0, -5.0])  # [v_min, alphadot_min]
    u_upper = np.array([10.0, 5.0])    # [v_max, alphadot_max]

    print(f"Optimization parameters:")
    print(f"  Time horizon T = {T} s")
    print(f"  Collocation points N = {N}")
    print(f"  Velocity weight w_v = {w_v}")
    print(f"  Steering rate weight w_alpha = {w_alpha}")
    print(f"  Control bounds: v in [{u_lower[0]}, {u_upper[0]}], "
          f"alphadot in [{u_lower[1]}, {u_upper[1]}]")
    print()

    # Create optimizer
    optimizer = TrajectoryOptimizer(model=model, N=N, T=T)

    # Solve
    print("Solving trajectory optimization problem...")
    print("-" * 60)
    start_time = time.time()

    success = optimizer.solve(
        x_start=x_start,
        x_goal=x_goal,
        W=W,
        u_lower=u_lower,
        u_upper=u_upper
    )

    solve_time = time.time() - start_time
    print("-" * 60)
    print()

    if success:
        print(f"✓ Optimization succeeded in {solve_time:.2f} seconds")
        print()

        # Get trajectory
        times, states, controls = optimizer.get_trajectory()

        # Compute final cost
        dt = times[1] - times[0]
        cost = np.sum(w_v * controls[0, :]**2 + w_alpha * controls[1, :]**2) * dt
        print(f"Final cost: {cost:.4f}")
        print()

        # Verify constraints
        final_state = states[:, -1]
        error = np.linalg.norm(final_state - x_goal)
        print(f"Final state error: {error:.6f}")
        print(f"  Position error: {np.linalg.norm(final_state[:2] - x_goal[:2]):.6f} m")
        print(f"  Heading error: {abs(final_state[2] - x_goal[2]):.6f} rad")
        print(f"  Steering error: {abs(final_state[3] - x_goal[3]):.6f} rad")
        print()

        # Print control statistics
        print("Control statistics:")
        print(f"  Velocity: min={controls[0, :].min():.3f}, "
              f"max={controls[0, :].max():.3f}, "
              f"mean={controls[0, :].mean():.3f} m/s")
        print(f"  Steering rate: min={controls[1, :].min():.3f}, "
              f"max={controls[1, :].max():.3f}, "
              f"mean={controls[1, :].mean():.3f} rad/s")
        print()

        # Visualize in MuJoCo
        print("Launching MuJoCo visualization...")
        print("(Close the MuJoCo window to exit)")
        print()
        visualize_trajectory(times, states, controls, model_path='robots/bicycle.xml', playback_speed=1.0)

    else:
        print(f"✗ Optimization failed after {solve_time:.2f} seconds")
        print("Try adjusting weights or increasing iteration limit.")


if __name__ == "__main__":
    main()
