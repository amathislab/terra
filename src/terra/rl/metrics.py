"""Validation metrics in the trajectory world frame."""

from typing import Any

from musclemimic.utils.metrics import MetricsHandler


class TerraMetricsHandler(MetricsHandler):
    """Keep preserved trajectory roots in their original world frame."""

    def __init__(self, config: Any, env: Any):
        super().__init__(config, env)
        self._preserve_trajectory_root_xy = bool(getattr(env, "preserve_trajectory_root_xy", False))

    def _get_root_xy_offset(self, env_states):
        if self._preserve_trajectory_root_xy:
            return None
        return super()._get_root_xy_offset(env_states)
