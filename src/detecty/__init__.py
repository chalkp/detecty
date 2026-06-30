"""detecty — automatic YOLO label generation for RoboCup@Home objects.

Recommended pipeline (ensemble): localize with Grounding DINO, classify each crop
by fusing DINOv3-L nearest-prototype + masked HSV colour + OCR brand-match.
"""
__version__ = "0.1.0"

from .pipeline import SamYolo  # noqa: E402

__all__ = ["SamYolo", "__version__"]
