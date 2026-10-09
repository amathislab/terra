"""TERRA reinforcement-learning components.

MuscleMimic provides the simulator and algorithms.  This package owns the
terrain-conditioned task definition used by TERRA experiments.
"""

from loco_mujoco.core.terrain import Terrain
from loco_mujoco.task_factories import TaskFactory
from terra.rl.environment import register_environments
from terra.rl.metrics import TerraMetricsHandler
from terra.rl.observations import TerraGoal
from terra.rl.rendering import TerraFullBodyTrackingGoalVisual, TerraGoalVisual
from terra.rl.rewards import TerraReward
from terra.rl.task_factory import TerraImitationFactory
from terra.rl.termination import TerraGlobalMPJPETerminalStateHandler
from terra.rl.terrain import PairedBoxTerrain
from terra.rl.tracking import TerraFullBodyTrackingGoal


def register_components() -> None:
    """Register the local task components with LocoMuJoCo."""

    register_environments()
    TerraGoal.register()
    TerraGoalVisual.register()
    TerraFullBodyTrackingGoal.register()
    TerraFullBodyTrackingGoalVisual.register()
    TerraReward.register()
    TerraGlobalMPJPETerminalStateHandler.register()
    registered_terrain = Terrain.registered.get(PairedBoxTerrain.get_name())
    if registered_terrain is None:
        PairedBoxTerrain.register()
    elif registered_terrain is not PairedBoxTerrain:
        raise ValueError("A different PairedBoxTerrain is already registered")
    registered_factory = TaskFactory.registered.get(TerraImitationFactory.get_name())
    if registered_factory is None:
        TerraImitationFactory.register()
    elif registered_factory is not TerraImitationFactory:
        raise ValueError("A different TerraImitationFactory is already registered")


__all__ = [
    "PairedBoxTerrain",
    "TerraFullBodyTrackingGoal",
    "TerraFullBodyTrackingGoalVisual",
    "TerraGlobalMPJPETerminalStateHandler",
    "TerraGoal",
    "TerraGoalVisual",
    "TerraImitationFactory",
    "TerraMetricsHandler",
    "TerraReward",
    "register_components",
]
