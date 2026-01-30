"""
Cart-Pole Inverted Pendulum Model for trajectory optimization.

Implements the dynamics of a cart-pole system (inverted pendulum).
"""

import numpy as np
from robot_model_base import RobotModelBase


class PendulumModel(RobotModelBase):
    """
    Cart-pole inverted pendulum dynamics.

    State vector: [x, theta, x_dot, theta_dot]
        x: cart position (m)
        theta: pendulum angle from upward vertical (rad)
               theta = 0 is upright (balanced)
               theta = π is hanging down
        x_dot: cart velocity (m/s)
        theta_dot: angular velocity (rad/s)

    Control vector: [F]
        F: horizontal force applied to cart (N)

    Parameters:
        m_cart: mass of cart (kg)
        m_pole: mass of pendulum pole (kg)
        L: length from pivot to center of mass of pole (m)
        g: gravitational acceleration (m/s²)

    Equations of motion derived from Lagrangian mechanics:
        (m_cart + m_pole) * x_ddot + m_pole * L * theta_ddot * cos(theta)
            - m_pole * L * theta_dot^2 * sin(theta) = F

        m_pole * L * x_ddot * cos(theta) + m_pole * L^2 * theta_ddot
            - m_pole * g * L * sin(theta) = 0
    """

    def __init__(self, m_cart=1.0, m_pole=0.1, L=0.5, g=9.81):
        """
        Initialize the pendulum model.

        Args:
            m_cart: Cart mass (kg), default 1.0
            m_pole: Pole mass (kg), default 0.1
            L: Pole length to center of mass (m), default 0.5
            g: Gravity (m/s²), default 9.81
        """
        super().__init__(state_dim=4, control_dim=1)
        self.m_cart = m_cart
        self.m_pole = m_pole
        self.L = L
        self.g = g

    def dynamics(self, state, control):
        """
        Compute state derivatives given current state and control.

        Args:
            state: [x, theta, x_dot, theta_dot] - state vector
            control: [F] - control vector (force on cart)

        Returns:
            state_dot: [x_dot, theta_dot, x_ddot, theta_ddot] - state derivatives
        """
        # Extract state variables
        x = state[0]
        theta = state[1]
        x_dot = state[2]
        theta_dot = state[3]

        # Extract control
        F = control[0]

        # Check if we're using CasADi symbolic variables
        try:
            import casadi as ca
            is_casadi = isinstance(state, (ca.MX, ca.SX, ca.DM)) or isinstance(control, (ca.MX, ca.SX, ca.DM))
        except ImportError:
            is_casadi = False

        if is_casadi:
            # CasADi operations
            import casadi as ca

            # Precompute trig functions
            sin_theta = ca.sin(theta)
            cos_theta = ca.cos(theta)

            # Total mass and pole moment
            m_total = self.m_cart + self.m_pole

            # Denominator for solving the coupled equations
            # From eliminating x_ddot: denominator = m_total * L - m_pole * L * cos²(theta)
            denom = m_total * self.L - self.m_pole * self.L * cos_theta**2

            # Solve for angular acceleration (theta_ddot)
            # From the second equation: theta_ddot = (x_ddot * cos(theta) + g * sin(theta)) / L
            # Substituting x_ddot from first equation and simplifying:
            theta_ddot = (m_total * self.g * sin_theta
                         - self.m_pole * self.L * theta_dot**2 * sin_theta * cos_theta
                         - F * cos_theta) / denom

            # Solve for cart acceleration (x_ddot)
            # From first equation:
            x_ddot = (F + self.m_pole * self.L * theta_dot**2 * sin_theta
                     - self.m_pole * self.L * theta_ddot * cos_theta) / m_total

            return ca.vertcat(x_dot, theta_dot, x_ddot, theta_ddot)
        else:
            # NumPy operations
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)

            m_total = self.m_cart + self.m_pole

            denom = m_total * self.L - self.m_pole * self.L * cos_theta**2

            theta_ddot = (m_total * self.g * sin_theta
                         - self.m_pole * self.L * theta_dot**2 * sin_theta * cos_theta
                         - F * cos_theta) / denom

            x_ddot = (F + self.m_pole * self.L * theta_dot**2 * sin_theta
                     - self.m_pole * self.L * theta_ddot * cos_theta) / m_total

            return np.array([x_dot, theta_dot, x_ddot, theta_ddot])

    def get_parameters(self):
        """Get model parameters."""
        return {
            'm_cart': self.m_cart,
            'm_pole': self.m_pole,
            'L': self.L,
            'g': self.g,
            'state_dim': self.state_dim,
            'control_dim': self.control_dim
        }

    def __repr__(self):
        """String representation of the model."""
        return (f"{self.__class__.__name__}(m_cart={self.m_cart}, "
                f"m_pole={self.m_pole}, L={self.L}, g={self.g})")
