"""Methods used by the terrain-reconstruction benchmark."""

from .least_squares import LeastSquaresMethod
from .terra import TerraMethod
from .voronoi import VoronoiMethod

__all__ = [
    "LeastSquaresMethod",
    "TerraMethod",
    "VoronoiMethod",
]
