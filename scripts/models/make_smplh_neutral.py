"""Build SMPLH_NEUTRAL.pkl from the SMPL-H and MANO downloads without chumpy."""

import argparse
import pickle
from pathlib import Path

import numpy as np


class _ChumpyArray:
    def __setstate__(self, state):
        self.state = state


class _ModelUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _ChumpyArray
        return super().find_class(module, name)


def convert(model_root: Path) -> Path:
    """Combine the neutral body arrays with the hand PCA arrays in licensed models."""
    with np.load(model_root / "smplh/neutral/model.npz") as archive:
        body = dict(archive)
    for side, filename in (("l", "MANO_LEFT.pkl"), ("r", "MANO_RIGHT.pkl")):
        with (model_root / "mano_v1_2/models" / filename).open("rb") as handle:
            hand = _ModelUnpickler(handle, encoding="latin1").load()
        for key in ("hands_components", "hands_coeffs", "hands_mean"):
            value = hand[key]
            if isinstance(value, _ChumpyArray):
                state = value.state
                value = state["x"] if isinstance(state, dict) and "x" in state else state
            body[f"{key}{side}"] = np.asarray(value)
    output = model_root / "SMPLH_NEUTRAL.pkl"
    with output.open("wb") as handle:
        pickle.dump(body, handle)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_root", type=Path, help="directory containing smplh/ and mano_v1_2/")
    print(convert(parser.parse_args().model_root.expanduser()))
