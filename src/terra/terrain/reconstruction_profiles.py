"""TERRA terrain-reconstruction method and ablation profiles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from terra.terrain.family import FAMILY_EVIDENCE_HEIGHT_ONLY

TERRA_NO_PHYSICAL_CUES_PROFILE = "terra-no-physical-cues"
TERRA_FULL_PROFILE = "terra"
TERRA_RECONSTRUCTION_PROFILES = (
    TERRA_NO_PHYSICAL_CUES_PROFILE,
    TERRA_FULL_PROFILE,
)


@dataclass(frozen=True)
class TerraReconstructionProfile:
    """One immutable reconstruction-stage configuration.

    Numerical terrain parameters remain the current library defaults for every arm.
    """

    name: str
    cli_name: str
    description: str
    forced_fit_options: tuple[tuple[str, Any], ...]
    stages: tuple[str, ...]
    use_free_space_evidence: bool
    use_neutral_foot_pitch: bool
    use_physical_family_cues: bool
    use_posed_seat_surface: bool

    @property
    def unsupported_support_kinds(self) -> tuple[str, ...]:
        """Support channels that this profile structurally cannot reconstruct."""

        return () if self.use_posed_seat_surface else ("pelvis",)

    def fit_options(
        self,
        base_options: Mapping[str, Any] | None,
        *,
        neutral_foot_pitch: Mapping[str, float] | None,
        calibrated_joint_offsets: Mapping[str, float] | None,
    ) -> dict[str, Any]:
        """Resolve effective fitter options without mutating caller-owned values."""

        options = dict(base_options or {})
        effective_offsets = None if calibrated_joint_offsets is None else dict(calibrated_joint_offsets)

        options.update(
            calibrated_joint_offsets=effective_offsets,
            # The physical-cues ablation removes the calibrated normal.
            neutral_foot_pitch=(
                dict(neutral_foot_pitch) if self.use_neutral_foot_pitch and neutral_foot_pitch is not None else None
            ),
            use_free_space_evidence=self.use_free_space_evidence,
        )
        if not self.use_posed_seat_surface:
            # A config-side seat surface must not silently leak into a reduced profile.
            options.pop("seat_support_heights", None)
        options.update(self.forced_fit_options)
        return options

    def to_dict(self, effective_fit_options: Mapping[str, Any]) -> dict[str, Any]:
        """Return the complete profile settings for one output record."""

        return {
            "name": self.name,
            "cli_name": self.cli_name,
            "description": self.description,
            "cumulative_stages": list(self.stages),
            "forced_fit_options": dict(self.forced_fit_options),
            "shared_evidence": {
                "source_landmarks": "motion-derived SMPL-H/MyoFullBody landmark trajectories",
                "contact_timing": "inferred from motion trajectories",
                "sole_offsets": "motion-derived flat/self calibration when available",
            },
            "additional_evidence": {
                "calibrated_sole_offsets": True,
                "free_space_extent": self.use_free_space_evidence,
                "neutral_foot_pitch": self.use_neutral_foot_pitch,
                "physical_family_cues": self.use_physical_family_cues,
                "posed_seat_surface": self.use_posed_seat_surface,
            },
            "unsupported_support_kinds": list(self.unsupported_support_kinds),
            "effective_fit_options": dict(effective_fit_options),
        }


_PROFILES = {
    TERRA_NO_PHYSICAL_CUES_PROFILE: TerraReconstructionProfile(
        name=TERRA_NO_PHYSICAL_CUES_PROFILE,
        cli_name="no-physical-cues",
        description=(
            "Full TERRA without supported-foot normal or swing-clearance evidence for "
            "ramp-versus-step selection; the frozen contact-height residual decides."
        ),
        forced_fit_options=(("family_evidence_mode", FAMILY_EVIDENCE_HEIGHT_ONLY),),
        stages=(
            "support_events",
            "calibrated_support_heights",
            "height_level_clustering",
            "per_level_boxes",
            "free_space_extent",
            "lower_support_exclusion",
            "height_profile_family_selection",
            "ramp_primitive",
            "stair_flight_primitive",
            "posed_surface_seat",
        ),
        use_free_space_evidence=True,
        use_neutral_foot_pitch=False,
        use_physical_family_cues=False,
        use_posed_seat_surface=True,
    ),
    TERRA_FULL_PROFILE: TerraReconstructionProfile(
        name=TERRA_FULL_PROFILE,
        cli_name="full",
        description=(
            "Current TERRA reconstruction with calibrated support evidence, physical "
            "ramp/stair family cues, structured primitives, and posed-surface seats."
        ),
        forced_fit_options=(),
        stages=(
            "support_events",
            "calibrated_support_heights",
            "height_level_clustering",
            "per_level_boxes",
            "free_space_extent",
            "lower_support_exclusion",
            "ramp_primitive",
            "stair_flight_primitive",
            "neutral_foot_surface_normal",
            "swing_clearance_tie_break",
            "orientation_slope_refinement",
            "posed_surface_seat",
        ),
        use_free_space_evidence=True,
        use_neutral_foot_pitch=True,
        use_physical_family_cues=True,
        use_posed_seat_surface=True,
    ),
}
_CLI_PROFILES = {profile.cli_name: profile for profile in _PROFILES.values()}


def resolve_terra_reconstruction_profile(name: str) -> TerraReconstructionProfile:
    """Resolve a canonical or CLI profile name."""

    if not isinstance(name, str):
        raise ValueError(f"TERRA reconstruction profile must be a string, got {name!r}")
    profile = _PROFILES.get(name) or _CLI_PROFILES.get(name)
    if profile is None:
        supported = ", ".join(TERRA_RECONSTRUCTION_PROFILES)
        raise ValueError(f"unsupported TERRA reconstruction profile {name!r}; expected one of {supported}")
    return profile


__all__ = [
    "TERRA_FULL_PROFILE",
    "TERRA_NO_PHYSICAL_CUES_PROFILE",
    "TERRA_RECONSTRUCTION_PROFILES",
    "TerraReconstructionProfile",
    "resolve_terra_reconstruction_profile",
]
