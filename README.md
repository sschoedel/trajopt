# Robot-Agnostic Trajectory Optimization

A flexible trajectory optimization framework that supports arbitrary robot models with different state and control dimensions.

## Features

- **Robot-Agnostic Design**: Supports any robot by implementing a simple interface
- **Generic Cost Function**: Quadratic control cost `u^T * W * u` with arbitrary dimensions
- **Flexible Constraints**: Control bounds work for any control dimension
- **Direct Collocation**: Uses CasADi and IPOPT for efficient optimization
- **MuJoCo Visualization**: Interactive visualization of optimized trajectories

## Project Structure

```
trajopt/
├── examples/               # Example optimization scripts
│   ├── bicycle_point_to_point.py    # Bicycle navigation example
│   └── pendulum_swing_up.py         # Inverted pendulum swing-up
├── models/                 # Robot model implementations
│   ├── __init__.py
│   ├── robot_model_base.py          # Abstract base class
│   ├── bicycle_model_base.py        # Bicycle model base
│   ├── bicycle_model.py             # Hand-written bicycle dynamics
│   ├── bicycle_model_pinocchio.py   # Pinocchio-based bicycle
│   └── pendulum_model.py            # Cart-pole dynamics
├── robots/                 # Robot description files (URDF/XML)
│   ├── bicycle.urdf
│   ├── bicycle.xml
│   ├── pendulum.urdf
│   └── pendulum.xml
├── results/                # Output plots and visualizations
│   ├── trajectory_result.png
│   └── pendulum_result.png
├── trajopt_solver.py       # Core trajectory optimizer
└── visualize_mujoco.py     # MuJoCo visualization utilities
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install Pinocchio for additional features
pip install pin
```

## Quick Start

### Bicycle Point-to-Point Navigation

```bash
python3 examples/bicycle_point_to_point.py
```

Optimizes a trajectory for a bicycle model to navigate from a start to goal position.

**Robot**: Bicycle (4D state, 2D control)
- State: `[x, y, theta, alpha]` (position, heading, steering angle)
- Control: `[v, alphadot]` (velocity, steering rate)

### Inverted Pendulum Swing-Up

```bash
python3 examples/pendulum_swing_up.py
```

Optimizes a swing-up trajectory for an inverted pendulum from hanging down to balanced upright.

**Robot**: Cart-Pole (4D state, 1D control)
- State: `[x, theta, x_dot, theta_dot]` (cart position, pole angle, velocities)
- Control: `[F]` (horizontal force on cart)

## Requirements

- Python 3.9+
- NumPy
- CasADi
- IPOPT
- MuJoCo
- Matplotlib
- Pinocchio (optional)
