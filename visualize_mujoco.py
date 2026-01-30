"""
Visualize trajectory optimization results using MuJoCo.

Robot-agnostic visualization that supports different robot models.
"""

import numpy as np
import mujoco
import mujoco.viewer
import time


def load_trajectory(optimizer):
    """Load trajectory from optimizer."""
    times, states, controls = optimizer.get_trajectory()
    return times, states, controls


def _setup_bicycle_markers(model, start_state, goal_state):
    """Set up start/goal markers for bicycle model."""
    start_marker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'start_marker')
    goal_marker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'goal_marker')

    # Set marker positions (x, y)
    model.body_pos[start_marker_id] = [start_state[0], start_state[1], 0.05]
    model.body_pos[goal_marker_id] = [goal_state[0], goal_state[1], 0.05]

    # Rotate markers to show heading direction (theta)
    model.body_quat[start_marker_id] = [np.cos(start_state[2]/2), 0, 0, np.sin(start_state[2]/2)]
    model.body_quat[goal_marker_id] = [np.cos(goal_state[2]/2), 0, 0, np.sin(goal_state[2]/2)]


def _setup_pendulum_markers(model, start_state, goal_state):
    """Set up start/goal markers for pendulum model."""
    start_marker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'start_marker')
    goal_marker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'goal_marker')

    # Set marker positions (cart x position only)
    model.body_pos[start_marker_id] = [start_state[0], 0, 0.05]
    model.body_pos[goal_marker_id] = [goal_state[0], 0, 0.05]


def _update_bicycle_state(data, state):
    """
    Update bicycle state in MuJoCo.

    State: [x, y, theta, alpha]
        x, y: position
        theta: heading angle
        alpha: steering angle
    """
    x, y, theta, alpha_steer = state

    # Set bicycle frame position and orientation
    data.qpos[0:3] = [x, y, 0.5]  # x, y, z

    # Orientation quaternion for rotation around z-axis
    data.qpos[3:7] = [np.cos(theta/2), 0, 0, np.sin(theta/2)]

    # Set steering angle (joint after the freejoint)
    data.qpos[7] = alpha_steer


def _update_pendulum_state(data, state):
    """
    Update pendulum state in MuJoCo.

    State: [x, theta, x_dot, theta_dot]
        x: cart position
        theta: pendulum angle (0 = up, π = down)
        x_dot, theta_dot: velocities (not used for visualization)
    """
    x_cart, theta, x_dot, theta_dot = state

    # Set cart position and pole angle
    # qpos for pendulum: [cart_position, pole_angle]
    data.qpos[0] = x_cart
    data.qpos[1] = theta


def visualize_trajectory(times, states, controls, model_path='bicycle.xml', playback_speed=1.0):
    """
    Visualize trajectory in MuJoCo viewer (robot-agnostic).

    Automatically detects robot type from model_path and applies
    appropriate state updates.

    Args:
        times: Time array (N+1,)
        states: State array (state_dim, N+1)
        controls: Control array (control_dim, N)
        model_path: Path to MuJoCo XML model file
        playback_speed: Playback speed multiplier (1.0 = real-time)

    Supported robots:
        - bicycle.xml: Bicycle model with state [x, y, theta, alpha]
        - pendulum.xml: Inverted pendulum with state [x, theta, x_dot, theta_dot]
    """
    # Detect robot type from model path
    if 'bicycle' in model_path.lower():
        robot_type = 'bicycle'
        robot_name = 'Bicycle'
    elif 'pendulum' in model_path.lower():
        robot_type = 'pendulum'
        robot_name = 'Inverted Pendulum'
    else:
        robot_type = 'unknown'
        robot_name = 'Robot'
        print(f"Warning: Unknown robot type for {model_path}")
        print("Attempting generic visualization...")

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Get start and goal states
    start_state = states[:, 0]
    goal_state = states[:, -1]

    # Set up markers based on robot type
    try:
        if robot_type == 'bicycle':
            _setup_bicycle_markers(model, start_state, goal_state)
        elif robot_type == 'pendulum':
            _setup_pendulum_markers(model, start_state, goal_state)
    except Exception as e:
        print(f"Note: Could not set up markers: {e}")

    print("=" * 60)
    print(f"MuJoCo {robot_name} Visualization")
    print("=" * 60)
    print(f"Trajectory duration: {times[-1]:.2f} seconds")
    print(f"Number of waypoints: {len(times)}")
    print(f"Playback speed: {playback_speed}x")
    print()
    print("Controls:")
    print("  - Right Click + Drag: Rotate view")
    print("  - Scroll: Zoom")
    print("  - Close window (or ESC) to exit")
    print("=" * 60)
    print()

    # Set up animation with passive viewer
    start_time = time.time()

    # Initialize to starting position
    if robot_type == 'bicycle':
        _update_bicycle_state(data, start_state)
    elif robot_type == 'pendulum':
        _update_pendulum_state(data, start_state)
    else:
        # Generic: try to set qpos directly from state
        data.qpos[:len(start_state)] = start_state

    mujoco.mj_forward(model, data)

    # Try to launch viewer
    try:
        viewer = mujoco.viewer.launch_passive(model, data)
    except RuntimeError as e:
        print("\n" + "="*60)
        print("Note: MuJoCo passive viewer requires mjpython on macOS")
        print("="*60)
        print("\nAlternative: View the saved trajectory plot")
        print("="*60)
        raise

    # Launch viewer with passive rendering
    with viewer:
        while viewer.is_running():
            # Calculate current trajectory time
            elapsed = time.time() - start_time
            traj_time = (elapsed * playback_speed) % times[-1]  # Loop

            # Interpolate state at current time
            idx = np.searchsorted(times, traj_time)
            if idx == 0:
                state = states[:, 0]
            elif idx >= len(times):
                state = states[:, -1]
            else:
                # Linear interpolation
                t0, t1 = times[idx - 1], times[idx]
                s0, s1 = states[:, idx - 1], states[:, idx]
                alpha_interp = (traj_time - t0) / (t1 - t0)
                state = s0 + alpha_interp * (s1 - s0)

            # Update robot state based on type
            if robot_type == 'bicycle':
                _update_bicycle_state(data, state)
            elif robot_type == 'pendulum':
                _update_pendulum_state(data, state)
            else:
                # Generic: try to set qpos directly
                data.qpos[:len(state)] = state

            # Update physics
            mujoco.mj_forward(model, data)

            # Sync with viewer
            viewer.sync()

            # Control frame rate
            time.sleep(0.01)
