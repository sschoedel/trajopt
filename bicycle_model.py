"""
Kinematic Bicycle Model for trajectory optimization.

Hand-written implementation of the kinematic bicycle dynamics.
"""

import numpy as np
from bicycle_model_base import BicycleModelBase


class BicycleModel(BicycleModelBase):
    """
    Kinematic bicycle model for vehicle dynamics.

    State vector: [x, y, theta, alpha]
        x, y: position of center of gravity
        theta: heading angle
        alpha: steering angle

    Control vector: [v, alphadot]
        v: velocity of rear wheels
        alphadot: steering angle rate

    Parameters:
        L: wheelbase (distance from rear to front wheel centers)
        m: mass (optional, for reference)
    """

    def __init__(self, L, m=None):
        """
        Initialize the hand-written bicycle model.

        Args:
            L: Wheelbase length (meters)
            m: Mass (kg, optional)
        """
        super().__init__(L, m)

    def dynamics(self, state, control):
        """
        Compute state derivatives given current state and control.

        Args:
            state: [x, y, theta, alpha] - state vector
            control: [v, alphadot] - control vector

        Returns:
            state_dot: [xdot, ydot, thetadot, alphadot] - state derivatives
        """
        # Extract state variables
        # x = state[0]  # not needed for dynamics
        # y = state[1]  # not needed for dynamics
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
            # CasADi operations
            import casadi as ca
            xdot = v * ca.cos(theta)
            ydot = v * ca.sin(theta)
            thetadot = v * ca.tan(alpha) / self.L
            alphadot_out = alphadot
            return ca.vertcat(xdot, ydot, thetadot, alphadot_out)
        else:
            # NumPy operations
            xdot = v * np.cos(theta)
            ydot = v * np.sin(theta)
            thetadot = v * np.tan(alpha) / self.L
            alphadot_out = alphadot
            return np.array([xdot, ydot, thetadot, alphadot_out])
