"""
Robot models for trajectory optimization.

Provides base classes and implementations for different robot types.
"""

from .robot_model_base import RobotModelBase
from .bicycle_model_base import BicycleModelBase
from .bicycle_model import BicycleModel
from .pendulum_model import PendulumModel

try:
    from .bicycle_model_pinocchio import BicycleModelPinocchio
except ImportError:
    # Pinocchio not available
    BicycleModelPinocchio = None

__all__ = [
    'RobotModelBase',
    'BicycleModelBase',
    'BicycleModel',
    'BicycleModelPinocchio',
    'PendulumModel',
]
