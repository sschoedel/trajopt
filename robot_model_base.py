"""
Abstract base class for robot models used in trajectory optimization.

Defines the interface that all robot models must implement to work with
the trajectory optimizer.
"""

from abc import ABC, abstractmethod


class RobotModelBase(ABC):
    """
    Abstract base class for all robot models.

    This class defines the interface that trajectory optimization expects.
    Concrete robot models must inherit from this class and implement the
    dynamics() method.

    Attributes:
        state_dim: Dimension of the state vector
        control_dim: Dimension of the control input vector
    """

    def __init__(self, state_dim, control_dim):
        """
        Initialize the robot model with specified dimensions.

        Args:
            state_dim: Dimension of the state vector
            control_dim: Dimension of the control input vector
        """
        self.state_dim = state_dim
        self.control_dim = control_dim

    @abstractmethod
    def dynamics(self, state, control):
        """
        Compute the time derivative of the state given current state and control.

        This method must be implemented by all concrete robot models.
        The implementation must work with both NumPy arrays and CasADi symbolic
        variables to support both simulation and optimization.

        Args:
            state: State vector (state_dim,)
            control: Control vector (control_dim,)

        Returns:
            state_dot: Time derivative of state (state_dim,)
        """
        pass

    def get_state_dim(self):
        """Return the state dimension."""
        return self.state_dim

    def get_control_dim(self):
        """Return the control dimension."""
        return self.control_dim
