"""
Visualize trajectory optimization results using MuJoCo.

Loads the optimized trajectory and replays it in MuJoCo's interactive viewer.
"""

import numpy as np
import mujoco
import mujoco.viewer
import time


def load_trajectory(optimizer):
    """Load trajectory from optimizer."""
    times, states, controls = optimizer.get_trajectory()
    return times, states, controls


def visualize_trajectory(times, states, controls, model_path='bicycle.xml', playback_speed=1.0):
    """
    Visualize trajectory in MuJoCo viewer.

    Args:
        times: Time array (N+1,)
        states: State array (4, N+1) - [x, y, theta, alpha]
        controls: Control array (2, N) - [v, alphadot]
        model_path: Path to MuJoCo XML model file
        playback_speed: Playback speed multiplier (1.0 = real-time)
    """
    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Get start and goal states
    start_state = states[:, 0]
    goal_state = states[:, -1]

    # Position the start and goal markers
    start_marker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'start_marker')
    goal_marker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'goal_marker')

    # Set marker positions (they are static bodies, so we modify the model)
    model.body_pos[start_marker_id] = [start_state[0], start_state[1], 0.05]
    model.body_pos[goal_marker_id] = [goal_state[0], goal_state[1], 0.05]

    # Rotate markers to show heading direction
    model.body_quat[start_marker_id] = [np.cos(start_state[2]/2), 0, 0, np.sin(start_state[2]/2)]
    model.body_quat[goal_marker_id] = [np.cos(goal_state[2]/2), 0, 0, np.sin(goal_state[2]/2)]

    print("=" * 60)
    print("MuJoCo Trajectory Visualization")
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
    state = states[:, 0]
    x, y, theta, alpha_steer = state
    data.qpos[0:3] = [x, y, 0.5]
    data.qpos[3:7] = [np.cos(theta/2), 0, 0, np.sin(theta/2)]
    data.qpos[7] = alpha_steer
    mujoco.mj_forward(model, data)

    # Try to launch viewer
    try:
        viewer = mujoco.viewer.launch_passive(model, data)
    except RuntimeError as e:
        print("\n" + "="*60)
        print("ERROR: MuJoCo viewer requires mjpython on macOS")
        print("="*60)
        print("\nTo install mjpython:")
        print("  python3 -m pip install mujoco-mjpython")
        print("\nThen run with:")
        print("  mjpython example_point_to_point.py")
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

            x, y, theta, alpha_steer = state

            # Set bicycle frame position and orientation
            data.qpos[0:3] = [x, y, 0.5]  # x, y, z

            # Orientation quaternion for rotation around z-axis
            data.qpos[3:7] = [np.cos(theta/2), 0, 0, np.sin(theta/2)]

            # Set steering angle (joint after the freejoint)
            data.qpos[7] = alpha_steer

            # Update physics
            mujoco.mj_forward(model, data)

            # Sync with viewer
            viewer.sync()

            # Control frame rate
            time.sleep(0.01)