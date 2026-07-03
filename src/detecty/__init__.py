"""detecty — object detection pipeline for RoboCup@Home objects.

Callable API (SamYolo): localize with Grounding DINO, classify each crop by
fusing DINOv3-L nearest-prototype + masked HSV colour + OCR brand-match.

    from detecty import SamYolo
    with SamYolo(device="cpu") as det:
        result = det.detect("image.jpg")   # -> dict of detections

BasketDetector is a discrete, lightweight sibling (Grounding DINO localize +
DINOv3 confirm, no OCR/colour/ensemble) for detecting baskets / hampers only:

    from detecty import BasketDetector
    with BasketDetector(protos="basket_prototypes.npz") as det:
        result = det.detect("frame.jpg")
"""
__version__ = "0.1.0"

from .basket import BasketDetector  # noqa: E402
from .pipeline import SamYolo  # noqa: E402

__all__ = ["SamYolo", "BasketDetector", "__version__"]
