"""Colour feature: masked HSV hue histogram (GrabCut foreground)."""
import cv2
import numpy as np

HUE_BINS = 32


def masked_hue_hist(pil):
    """HSV hue histogram over the GrabCut foreground, chromatic pixels only.

    Returns (hist[HUE_BINS] normalized, chroma_fraction). Background/table pixels
    are excluded so colour reflects the object, not the white board.
    """
    bgr = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    m = max(2, int(0.08 * min(h, w)))
    rect = (m, m, max(1, w - 2 * m), max(1, h - 2 * m))
    try:
        cv2.grabCut(bgr, mask, rect, np.zeros((1, 65), np.float64),
                    np.zeros((1, 65), np.float64), 3, cv2.GC_INIT_WITH_RECT)
        fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    except cv2.error:
        fg = np.ones((h, w), np.uint8)
    if fg.sum() < 0.02 * h * w:
        fg = np.ones((h, w), np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    chroma = fg.astype(bool) & (hsv[..., 1] > 40) & (hsv[..., 2] > 40)
    hues = hsv[..., 0][chroma]                      # OpenCV H in [0,180)
    hist = np.histogram(hues, bins=HUE_BINS, range=(0, 180))[0].astype(np.float32)
    s = hist.sum()
    return (hist / s if s > 0 else hist), float(chroma.sum()) / (h * w)


def hist_intersection(a, b) -> float:
    return float(np.minimum(a, b).sum())
