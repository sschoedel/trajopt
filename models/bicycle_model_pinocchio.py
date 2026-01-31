"""
Pinocchio-based Kinematic Bicycle Model for trajectory optimization.

Uses Pinocchio library to load the bicycle structure from URDF and
implements kinematic bicycle dynamics.
"""

import numpy as np
import pinocchio as pin
from .bicycle_model_base import BicycleModelBase


class BicycleModelPinocchio(BicycleModelBase):
    """
    Kinematic bicycle model using Pinocchio for model management.

    State vector: [x, y, theta, alpha]
        x, y: position of center of gravity (rear axle)
        theta: heading angle
        alpha: steering angle

    Control vector: [v, alphadot]
        v: velocity of rear wheels
        alphadot: steering angle rate

    Parameters:
        L: wheelbase (distance from rear to front wheel centers)
        model_path: Path to URDF file
    """

    def __init__(self, model_path='bicycle.urdf', L=2.5, m=None):
        """
        Initialize the Pinocchio-based bicycle model.

        Args:
            model_path: Path to URDF file
            L: Wheelbase length (meters)
            m: Mass (kg, optional - for reference)
        """
        # Initialize base class
        super().__init__(L, m)

        # Load URDF model
        self.pin_model = pin.buildModelFromUrdf(model_path)
        self.pin_data = self.pin_model.createData()

        # Joint info
        self.nq = self.pin_model.nq  # Configuration dimension
        self.nv = self.pin_model.nv  # Velocity dimension

        print(f"Loaded Pinocchio bicycle model:")
        print(f"  Number of joints: {self.pin_model.njoints}")
        print(f"  Configuration dim (nq): {self.nq}")
        print(f"  Velocity dim (nv): {self.nv}")
        print(f"  Wheelbase L: {self.L} m")

    def dynamics(self, state, control):
        """
        Compute state derivatives for kinematic bicycle model.

        Args:
            state: [x, y, theta, alpha] - state vector
            control: [v, alphadot] - control vector

        Returns:
            state_dot: [xdot, ydot, thetadot, alphadot] - state derivatives
        """
        # Extract state variables
        theta = state[2]
        alpha = state[3]

        # Extract control variables
        v = control[0]
        alphadot = control[1]

        # Check if we're using CasADi symbolic variables
        try:
            import casadi as ca
            is_casadi = isinstance(state, (ca.MX, ca.SX, ca.DM)) or isinstance(control, (ca.MX, ca.SX, ca.DM))
        except ImportError:
            is_casadi = False

        if is_casadi:
            # CasADi operations for optimization
            import casadi as ca
            xdot = v * ca.cos(theta)
            ydot = v * ca.sin(theta)
            thetadot = v * ca.tan(alpha) / self.L
            alphadot_out = alphadot
            return ca.vertcat(xdot, ydot, thetadot, alphadot_out)
        else:
            # NumPy operations for simulation
            xdot = v * np.cos(theta)
            ydot = v * np.sin(theta)
            thetadot = v * np.tan(alpha) / self.L
            alphadot_out = alphadot
            return np.array([xdot, ydot, thetadot, alphadot_out])

    def get_model_info(self):
        """Get information about the Pinocchio model."""
        return {
            'name': self.pin_model.name,
            'njoints': self.pin_model.njoints,
            'nq': self.nq,
            'nv': self.nv,
            'joint_names': [self.pin_model.names[i] for i in range(self.pin_model.njoints)]
        }
