"""Discrete basket / hamper detector — Grounding DINO localize + DINOv3 confirm.

A deliberately small pipeline answering one question: *where are the baskets /
hampers in this frame?* — the container the robot drops objects into. It reuses
detecty's two accurate building blocks but **none** of the 30-class ensemble
machinery (no EasyOCR, no masked-HSV colour, no catalog prototype bank):

  1. **Localize** — Grounding DINO with basket / hamper text prompts (open-vocab,
     class-agnostic boxes).
  2. **Confirm** — each crop is embedded with the same DINOv3 backbone detecty
     uses and matched (cosine, L2-normalized) against a small per-class
     reference-prototype bank; crops below ``sim_threshold`` are rejected as
     localizer false positives (a bin, a box, a chair).

This is the more accurate re-map of the earlier YOLOE + CLIP prototype
(``BasketDetectRobocupEIC``): the DINOv3 representation separates basket / hamper
from background far more reliably than CLIP-ViT-B/32, and Grounding DINO
localizes the (often frame-filling) container better than a fixed detector.

    from detecty import BasketDetector

    with BasketDetector(protos="basket_prototypes.npz", device="cpu") as det:
        result = det.detect("frame.jpg")
    for d in result["detections"]:
        print(d["class"], round(d["score"], 3), d["bbox"])

Basket and hamper are treated as one ``basket`` class (a hamper is a basket;
the split was a labeling artifact) — "hamper" survives only as a Grounding DINO
localizer synonym to widen recall, never as an output label.

``detect()`` returns::

    {
      "width": int, "height": int, "num_detections": int,
      "detections": [
        {"class": "basket", "score": float, "det_score": float,
         "bbox": [x1, y1, x2, y2],          # pixels
         "bbox_norm": [cx, cy, w, h]},      # YOLO-normalized
        ... ]
    }

Everything defaults to CPU (a 1 GB GPU cannot fit Grounding DINO + DINOv3).
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.ops import nms

from .embedder import DEFAULT_MODEL, Embedder
from .localize_gd import _run_prompt

# Text prompts fed to Grounding DINO. Kept to the container itself — synonyms
# (incl. "hamper", which is the same class) widen recall without pulling in
# unrelated furniture. The DINOv3 confirm step decides basket vs. not, so a
# slightly loose prompt is fine.
BASKET_PROMPTS = ["basket", "laundry basket", "hamper", "wicker basket"]

# Grounding DINO checkpoint — matches detecty's localizer default (config.yaml).
GD_MODEL_ID = "IDEA-Research/grounding-dino-tiny"


class BasketDetector:
    """Load once (setup), detect baskets many times (detect), free (shutdown)."""

    def __init__(self, protos="basket_prototypes.npz", dino_model=DEFAULT_MODEL,
                 device="cpu", gd_model_id=GD_MODEL_ID, prompts=None,
                 sim_threshold=0.55, box_threshold=0.25, text_threshold=0.20,
                 nms_iou=0.5, long_side=1280, min_box=16,
                 min_area_frac=0.002, max_area_frac=1.0,
                 negative_classes=("not_basket",)):
        """
        protos: path to the basket prototype bank (.npz of ``names`` + ``vecs``,
            built by ``detecty.build_basket_prototypes`` from reference crops).
        dino_model: DINOv3 backbone for the confirm step. Use the SAME backbone
            the bank was embedded with — mixing backbones makes cosine sims
            meaningless.
        sim_threshold: minimum cosine similarity (DINOv3, not CLIP — this scale
            runs lower, ~0.5-0.9; calibrate on a known basket crop). Below it a
            localizer box is discarded.
        box_threshold / text_threshold: Grounding DINO confidence gates.
        max_area_frac: basket/hamper close-ups routinely fill most of the frame,
            so this defaults to 1.0 (unlike the ensemble's small-object cap).
        negative_classes: prototype-bank class names that count as HARD NEGATIVES
            — a box whose nearest prototype is one of these is rejected outright,
            regardless of ``sim_threshold``. Drop crops of the usual laundry-room
            confusers (front-loader washing machines, bins) into
            ``<refs>/not_basket/`` and they stop being detected as baskets even
            when their absolute basket-similarity is high. Positive classes are
            all bank classes NOT listed here.
        """
        self.protos_path = protos
        self.dino_model = dino_model
        self.device = device
        self.gd_model_id = gd_model_id
        self.prompts = list(prompts) if prompts else list(BASKET_PROMPTS)
        self.sim_threshold = sim_threshold
        self.negative_classes = set(negative_classes or ())
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_iou = nms_iou
        self.long_side = long_side
        self.min_box = min_box
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self._ready = False

    # ------------------------------------------------------------------ setup
    def setup(self):
        """Load Grounding DINO, the DINOv3 embedder and the prototype bank. Idempotent."""
        if self._ready:
            return self
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._gd_proc = AutoProcessor.from_pretrained(self.gd_model_id)
        self._gd = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(self.gd_model_id).to(self.device).eval()
        )
        self._emb = Embedder(self.dino_model, self.device)

        pz = np.load(self.protos_path, allow_pickle=True)
        self.p_names, self.p_vecs = pz["names"], pz["vecs"]
        self._by_cls = defaultdict(list)
        for i, n in enumerate(self.p_names):
            self._by_cls[str(n)].append(i)
        self._ready = True
        return self

    # --------------------------------------------------------------- localize
    def _localize(self, img):
        """Grounding DINO -> class-agnostic (x1, y1, x2, y2, det_score) boxes."""
        W, H = img.size
        scale = self.long_side / max(W, H) if max(W, H) > self.long_side else 1.0
        small = (img.resize((max(1, round(W * scale)), max(1, round(H * scale))))
                 if scale < 1.0 else img)
        prompt = " . ".join(self.prompts)
        res = _run_prompt(small, prompt, self._gd, self._gd_proc, self.device,
                          self.box_threshold, self.text_threshold, (H, W))
        boxes = [b.tolist() for b in res["boxes"]]
        scores = [float(s) for s in res["scores"]]
        if not boxes:
            return []
        keep = nms(torch.tensor(boxes, dtype=torch.float32),
                   torch.tensor(scores, dtype=torch.float32), self.nms_iou)
        area = float(W * H)
        out = []
        for i in keep.tolist():
            x1, y1, x2, y2 = boxes[i]
            x1, x2 = sorted((max(0.0, x1), min(float(W), x2)))
            y1, y2 = sorted((max(0.0, y1), min(float(H), y2)))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            af = (x2 - x1) * (y2 - y1) / area
            if af < self.min_area_frac or af > self.max_area_frac:
                continue
            out.append((x1, y1, x2, y2, scores[i]))
        return out

    # ---------------------------------------------------------------- confirm
    def _confirm(self, crop):
        """Nearest-prototype (class, cosine-sim) of ``crop`` over the whole bank.

        The winner is the argmax over ALL classes including negatives, so a
        washing-machine crop that resembles a basket still loses to the closer
        ``not_basket`` prototype and gets rejected upstream.
        """
        v = self._emb.embed(crop)                 # L2-normalized
        sims = self.p_vecs @ v                     # cosine per prototype
        best = {c: max(float(sims[i]) for i in idx)
                for c, idx in self._by_cls.items()}
        cls = max(best, key=best.get)
        return cls, best[cls]

    # ----------------------------------------------------------------- detect
    def detect(self, image, min_box=None) -> dict:
        """Detect baskets / hampers in one image (path/str or PIL.Image)."""
        if not self._ready:
            raise RuntimeError("call setup() before detect()")
        min_box = self.min_box if min_box is None else min_box
        if isinstance(image, (str, Path)):
            img = ImageOps.exif_transpose(Image.open(image).convert("RGB"))
        else:
            img = image.convert("RGB")
        W, H = img.size

        dets = []
        for x1, y1, x2, y2, det_score in self._localize(img):
            crop = img.crop((max(0, int(x1)), max(0, int(y1)),
                             min(W, int(x2)), min(H, int(y2))))
            if crop.width < min_box or crop.height < min_box:
                continue
            cls, sim = self._confirm(crop)
            if cls in self.negative_classes or sim < self.sim_threshold:
                continue
            dets.append({
                "class": cls, "score": round(sim, 4),
                "det_score": round(det_score, 4),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "bbox_norm": [round((x1 + x2) / 2 / W, 6), round((y1 + y2) / 2 / H, 6),
                              round((x2 - x1) / W, 6), round((y2 - y1) / H, 6)],
            })
        return {"width": W, "height": H, "num_detections": len(dets),
                "detections": dets}

    def detect_batch(self, images, min_box=None):
        """Detect on many images. Returns a list of per-image result dicts."""
        return [self.detect(im, min_box=min_box) for im in images]

    # --------------------------------------------------------------- shutdown
    def shutdown(self):
        """Release models and free memory."""
        for attr in ("_gd", "_gd_proc", "_emb"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self._ready = False

    def __enter__(self):
        return self.setup()

    def __exit__(self, *exc):
        self.shutdown()
        return False
