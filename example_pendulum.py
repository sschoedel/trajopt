"""
Example: Swing-up control for inverted pendulum.

Demonstrates using the trajectory optimizer to swing a pendulum from
hanging down to balanced upright position.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
import time
from pendulum_model import PendulumModel
from trajopt_solver import TrajectoryOptimizer
from visualize_mujoco import visualize_trajectory


def plot_trajectory(times, states, controls, model):
    """
    Visualize the optimized pendulum trajectory.

    Args:
        times: Time array
        states: State trajectory (4 x N+1) - [x, theta, x_dot, theta_dot]
        controls: Control trajectory (1 x N) - [F]
        model: PendulumModel instance
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Cart position and pendulum angle
    ax = axes[0, 0]
    ax.plot(times, states[0, :], 'b-', linewidth=2, label='Cart position x (m)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cart Position (m)', color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax.grid(True)

    ax2 = ax.twinx()
    ax2.plot(times, states[1, :], 'r-', linewidth=2, label='Pendulum angle θ (rad)')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3, label='Upright (θ=0)')
    ax2.axhline(y=np.pi, color='k', linestyle='--', alpha=0.3, label='Down (θ=π)')
    ax2.set_ylabel('Pendulum Angle (rad)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.set_title('Cart Position and Pendulum Angle vs Time')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Plot 2: Cart and pendulum velocities
    ax = axes[0, 1]
    ax.plot(times, states[2, :], 'b-', linewidth=2, label='Cart velocity ẋ (m/s)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cart Velocity (m/s)', color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax.grid(True)

    ax2 = ax.twinx()
    ax2.plot(times, states[3, :], 'r-', linewidth=2, label='Angular velocity θ̇ (rad/s)')
    ax2.set_ylabel('Angular Velocity (rad/s)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.set_title('Velocities vs Time')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Plot 3: Control force over time
    ax = axes[1, 0]
    times_control = times[:-1]
    # Handle both 1D and 2D control arrays
    control_values = controls[0, :] if controls.ndim > 1 else controls
    ax.step(times_control, control_values, where='post',
            label='Force F (N)', linewidth=2, color='green')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (N)')
    ax.set_title('Control Force vs Time')
    ax.legend()
    ax.grid(True)

    # Plot 4: Pendulum angle in phase space (theta vs theta_dot)
    ax = axes[1, 1]
    ax.plot(states[1, :], states[3, :], 'b-', linewidth=2, alpha=0.6)
    ax.plot(states[1, 0], states[3, 0], 'go', markersize=10, label='Start')
    ax.plot(states[1, -1], states[3, -1], 'ro', markersize=10, label='Goal')
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.3, label='Upright')
    ax.axvline(x=np.pi, color='k', linestyle='--', alpha=0.3, label='Down')
    ax.axvline(x=-np.pi, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Angle θ (rad)')
    ax.set_ylabel('Angular Velocity θ̇ (rad/s)')
    ax.set_title('Phase Portrait')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('/Users/samschoedel/git/trajopt/pendulum_result.png', dpi=150)
    print("Plot saved to pendulum_result.png")
    # Don't block - just save the plot
    # plt.show()


def main():
    """
    Run pendulum swing-up trajectory optimization example.
    """
    print("=" * 60)
    print("Trajectory Optimization for Inverted Pendulum")
    print("Swing-Up Control Example")
    print("=" * 60)
    print()

    # Create pendulum model
    model = PendulumModel(m_cart=1.0, m_pole=0.1, L=0.5, g=9.81)

    print(f"Pendulum parameters:")
    print(f"  Cart mass = {model.m_cart} kg")
    print(f"  Pole mass = {model.m_pole} kg")
    print(f"  Pole length = {model.L} m")
    print(f"  Gravity = {model.g} m/s²")
    print()

    # Define start and goal states
    # State: [x, theta, x_dot, theta_dot]
    x_start = np.array([0.0, np.pi, 0.0, 0.0])  # Hanging down, at rest
    x_goal = np.array([0.0, 0.0, 0.0, 0.0])     # Upright, balanced, at rest

    print(f"Start state: x={x_start[0]:.2f} m, θ={x_start[1]:.2f} rad (down), "
          f"ẋ={x_start[2]:.2f} m/s, θ̇={x_start[3]:.2f} rad/s")
    print(f"Goal state:  x={x_goal[0]:.2f} m, θ={x_goal[1]:.2f} rad (up), "
          f"ẋ={x_goal[2]:.2f} m/s, θ̇={x_goal[3]:.2f} rad/s")
    print()

    # Optimization parameters
    N = 100  # number of collocation points
    T = 5.0  # time horizon (seconds)
    W = np.array([[1.0]])  # 1x1 cost matrix for force

    # Control bounds
    F_max = 20.0
    u_lower = np.array([-F_max])  # Max push left
    u_upper = np.array([F_max])   # Max push right

    print(f"Optimization parameters:")
    print(f"  Time horizon T = {T} s")
    print(f"  Collocation points N = {N}")
    print(f"  Force cost weight = {W[0, 0]}")
    print(f"  Force bounds: [{u_lower[0]}, {u_upper[0]}] N")
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
        # For 1D control, controls is (1, N), for general case it's (control_dim, N)
        if controls.ndim == 1:
            # 1D array for single control
            cost = np.sum(controls**2) * dt * W[0, 0]
        else:
            # 2D array (control_dim, N)
            cost = 0
            for k in range(controls.shape[1]):
                u_k = controls[:, k].reshape(-1, 1)
                cost += dt * float(u_k.T @ W @ u_k)
        print(f"Final cost: {cost:.4f}")
        print()

        # Verify constraints
        final_state = states[:, -1]
        error = np.linalg.norm(final_state - x_goal)
        print(f"Final state error: {error:.6f}")
        print(f"  Cart position error: {abs(final_state[0] - x_goal[0]):.6f} m")
        print(f"  Angle error: {abs(final_state[1] - x_goal[1]):.6f} rad")
        print(f"  Cart velocity error: {abs(final_state[2] - x_goal[2]):.6f} m/s")
        print(f"  Angular velocity error: {abs(final_state[3] - x_goal[3]):.6f} rad/s")
        print()

        # Print control statistics
        print("Control statistics:")
        if controls.ndim == 1:
            # 1D control
            print(f"  Force: min={controls.min():.3f}, "
                  f"max={controls.max():.3f}, "
                  f"mean={controls.mean():.3f} N")
        else:
            # Multi-dimensional control
            print(f"  Force: min={controls[0, :].min():.3f}, "
                  f"max={controls[0, :].max():.3f}, "
                  f"mean={controls[0, :].mean():.3f} N")
        print()

        # Visualize results
        print("Creating visualization plots...")
        plot_trajectory(times, states, controls, model)

        # Try MuJoCo visualization
        try:
            print("\nLaunching MuJoCo visualization...")
            print("(Close the MuJoCo window to exit)")
            print()
            visualize_trajectory(times, states, controls, model_path='pendulum.xml', playback_speed=1.0)
        except RuntimeError:
            print("\nMuJoCo visualization not available on this system.")
            print("Check the saved plot: pendulum_result.png")
            pass

    else:
        print(f"✗ Optimization failed after {solve_time:.2f} seconds")
        print("Try adjusting weights, time horizon, or increasing iteration limit.")


if __name__ == "__main__":
    main()
