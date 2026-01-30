"""
Abstract base class for bicycle models.

Defines the common interface that all bicycle model implementations must follow.
"""

from abc import ABC, abstractmethod
from robot_model_base import RobotModelBase


class BicycleModelBase(RobotModelBase):
    """
    Abstract base class for kinematic bicycle models.

    All bicycle model implementations must inherit from this class and
    implement the required abstract methods.

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
        Initialize the bicycle model.

        Args:
            L: Wheelbase length (meters)
            m: Mass (kg, optional)
        """
        super().__init__(state_dim=4, control_dim=2)
        self.L = L
        self.m = m

    @abstractmethod
    def dynamics(self, state, control):
        """
        Compute state derivatives given current state and control.

        This method must be implemented by all derived classes.

        Args:
            state: [x, y, theta, alpha] - state vector
            control: [v, alphadot] - control vector

        Returns:
            state_dot: [xdot, ydot, thetadot, alphadot] - state derivatives

        Note:
            Implementation must handle both NumPy arrays and CasADi symbolic
            variables for compatibility with optimization.
        """
        pass

    def get_parameters(self):
        """Get model parameters."""
        return {
            'L': self.L,
            'm': self.m,
            'state_dim': self.state_dim,
            'control_dim': self.control_dim
        }

    def __repr__(self):
        """String representation of the model."""
        return f"{self.__class__.__name__}(L={self.L}, m={self.m})"
