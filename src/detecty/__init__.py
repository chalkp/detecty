"""detecty — object detection pipeline for RoboCup@Home objects.

Callable API (SamYolo): localize with Grounding DINO, classify each crop by
fusing DINOv3-L nearest-prototype + masked HSV colour + OCR brand-match.

    from detecty import SamYolo
    with SamYolo(device="cpu") as det:
        result = det.detect("image.jpg")   # -> dict of detections
"""
__version__ = "0.1.0"

from .pipeline import SamYolo  # noqa: E402

__all__ = ["SamYolo", "__version__"]
