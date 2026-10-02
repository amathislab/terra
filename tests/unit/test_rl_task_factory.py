from __future__ import annotations

import pytest

from terra.rl.task_factory import TerraImitationFactory


@pytest.mark.parametrize("method", ("terra", "omniretarget", "gmr", "smpl"))
def test_paired_terrain_config_preserves_supported_retargeting_method(method):
    config = TerraImitationFactory._amass_config(
        {
            "rel_dataset_path": ["Study/Trial"],
            "retargeting_method": method,
            "output_cache_subdir": method,
            "load_paired_terrain": True,
            "require_nonflat_terrain": False,
        }
    )

    assert config.retargeting_method == method
    assert config.output_cache_subdir == method
    assert config.load_paired_terrain is True


def test_paired_terrain_config_rejects_unknown_retargeting_method():
    with pytest.raises(ValueError, match="unknown retargeting method"):
        TerraImitationFactory._amass_config(
            {
                "rel_dataset_path": ["Study/Trial"],
                "retargeting_method": "unknown",
                "load_paired_terrain": True,
            }
        )
