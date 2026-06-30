"""High-level one-class API wrapping the whole pipeline.

    from detecty import SamYolo

    det = SamYolo(device="cpu").setup()        # load models + prototype bank
    result = det.detect("object/img.jpg")      # -> dict (see detect())
    for d in result["detections"]:
        print(d["class"], round(d["score"], 2), d["bbox"])
    det.shutdown()                             # free models

Or as a context manager:

    with SamYolo() as det:
        result = det.detect(pil_image)

Localization = Grounding DINO (generic, class-agnostic). Classification = ensemble
(DINOv3-L nearest-prototype + masked HSV colour + OCR brand-match). Everything
defaults to CPU (1 GB GPU cannot fit the models).
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

from . import vlm_consult
from ._resources import default_config, default_ensemble
from .embedder import DEFAULT_MODEL, Embedder
from .features import hist_intersection, masked_hue_hist
from .localize_gd import detect as gd_detect
from .localize_gd import load_config

# colours cycled per class id for draw()
PALETTE = [
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
    (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190), (0, 128, 128),
    (170, 110, 40), (255, 215, 0), (128, 0, 0), (170, 255, 195), (0, 0, 128),
    (128, 128, 0), (255, 99, 71), (46, 139, 87), (30, 144, 255), (218, 112, 214),
    (160, 82, 45), (199, 21, 133), (47, 79, 79), (255, 140, 0), (0, 191, 255),
    (220, 20, 60), (34, 139, 34), (138, 43, 226), (210, 105, 30), (1, 50, 32),
]


class SamYolo:
    """Load once (setup), detect many (detect), free (shutdown)."""

    def __init__(self, config=None, ensemble=None, protos="prototypes.npz",
                 dino_model=DEFAULT_MODEL, device="cpu", use_ocr=True,
                 use_vlm=False, quantize_localizer=False):
        self.config_path = config or default_config()
        self.ensemble_path = ensemble or default_ensemble()
        self.protos_path = protos
        self.dino_model = dino_model
        self.device = device
        self.use_ocr = use_ocr
        self.use_vlm = use_vlm
        self.quantize_localizer = quantize_localizer
        self._ready = False

    # ------------------------------------------------------------------ setup
    def setup(self):
        """Load Grounding DINO, DINOv3 embedder, prototype bank, OCR. Idempotent."""
        if self._ready:
            return self
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.cfg = load_config(self.config_path)
        self.id_of = {c["name"]: c["id"] for c in self.cfg["classes"]}
        self.ecfg = yaml.safe_load(open(self.ensemble_path))
        self.F = self.ecfg["fusion"]
        self.brand_kw = self.ecfg.get("brand_keywords", {})
        self.cat_of = self.ecfg.get("category_of", {})

        mid = self.cfg["model"]["id"]
        self.long_side = self.cfg["model"]["inference_longest_side"]
        self._gd_proc = AutoProcessor.from_pretrained(mid)
        gd = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(self.device).eval()
        if self.quantize_localizer:
            import torch
            gd = torch.quantization.quantize_dynamic(gd.to("cpu"), {torch.nn.Linear}, dtype=torch.qint8)
            self.device = "cpu"
        self._gd = gd

        self._emb = Embedder(self.dino_model, self.device)

        pz = np.load(self.protos_path, allow_pickle=True)
        self.p_names, self.p_srcs = pz["names"], pz["srcs"]
        self.p_vecs, self.p_hues = pz["vecs"], pz["hues"]
        self._penalty = np.array([self.F["catalog_penalty"] if s == "catalog" else 0.0
                                  for s in self.p_srcs])
        self._by_cls = defaultdict(list)
        for i, n in enumerate(self.p_names):
            self._by_cls[str(n)].append(i)

        self._ocr = None
        if self.use_ocr:
            try:
                import easyocr
                self._ocr = easyocr.Reader(self.ecfg["ocr"]["langs"],
                                           gpu=(self.device != "cpu"), verbose=False)
            except Exception as e:
                print(f"[SamYolo] OCR disabled ({e})")

        self._ready = True
        return self

    # ------------------------------------------------------------- classify
    def _ocr_hits(self, crop):
        if self._ocr is None:
            return {}
        txt = " ".join(t.lower() for _, t, c in self._ocr.readtext(np.array(crop))
                       if c >= self.ecfg["ocr"]["min_conf"])
        return {cls: 1.0 for cls, kws in self.brand_kw.items()
                if any(kw.lower() in txt for kw in kws)}

    def _classify(self, crop):
        v = self._emb.embed(crop)
        hh = masked_hue_hist(crop)[0]
        sims = (self.p_vecs @ v) - self._penalty
        e = {c: max(sims[i] for i in idx) for c, idx in self._by_cls.items()}
        col = {c: max(hist_intersection(hh, self.p_hues[i]) for i in idx)
               for c, idx in self._by_cls.items()}
        ocr = self._ocr_hits(crop)
        rank = sorted(e, key=e.get, reverse=True)
        plausible = set(rank[:8])
        score, used_ocr = {}, set()
        for c in e:
            s = e[c] + self.F["w_color"] * col.get(c, 0.0)
            if c in ocr and (c in plausible or self.cat_of.get(c) == self.cat_of.get(rank[0])):
                s += self.F["w_ocr"] * ocr[c]
                used_ocr.add(c)
            score[c] = s
        order = sorted(score, key=score.get, reverse=True)
        top = order[0]
        margin = score[top] - (score[order[1]] if len(order) > 1 else 0.0)
        return dict(cls=top, score=float(score[top]), margin=float(margin),
                    candidates=order[:3], source=("ocr" if top in used_ocr else "ensemble"))

    # -------------------------------------------------------------- detect
    def detect(self, image, min_box=8) -> dict:
        """Detect known objects in one image.

        image: path/str or PIL.Image. Returns:
        {
          "width": int, "height": int,
          "num_detections": int, "num_review": int,
          "detections": [
            {"class": str, "class_id": int, "score": float, "margin": float,
             "candidates": [str, str, str], "bbox": [x1,y1,x2,y2],
             "bbox_norm": [cx,cy,w,h], "review": bool, "source": str},
            ... ]
        }
        Detections flagged "review": True are low-confidence (kept in the list so
        the caller can decide). Multi-view voting is a batch feature (detecty-label
        --multiview) and is not applied to single-image detect().
        """
        if not self._ready:
            raise RuntimeError("call setup() before detect()")
        if isinstance(image, (str, Path)):
            img = ImageOps.exif_transpose(Image.open(image).convert("RGB"))
        else:
            img = image.convert("RGB")
        W, H = img.size

        d = self.cfg["defaults"]
        boxes = gd_detect(img, self._gd, self._gd_proc, self.device, self.long_side,
                          self.cfg, "generic", d["nms_iou"],
                          d.get("min_box_area_frac", 0.0), d.get("max_box_area_frac", 1.0))

        dets = []
        for x1, y1, x2, y2, _bscore, _cid in boxes:
            crop = img.crop((max(0, x1), max(0, y1), min(W, x2), min(H, y2)))
            if crop.width < min_box or crop.height < min_box:
                continue
            r = self._classify(crop)
            review = r["margin"] < self.F["margin_review"] or r["score"] < self.F["min_top_score"]
            if review and self.use_vlm and vlm_consult.available():
                pick = vlm_consult.consult(crop, r["candidates"],
                                           hints="RoboCup@Home object; read the brand/logo.")
                if pick:
                    r["cls"], r["source"], review = pick, "vlm", False
            dets.append({
                "class": r["cls"], "class_id": self.id_of.get(r["cls"], -1),
                "score": round(r["score"], 4), "margin": round(r["margin"], 4),
                "candidates": r["candidates"],
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "bbox_norm": [round((x1 + x2) / 2 / W, 6), round((y1 + y2) / 2 / H, 6),
                              round((x2 - x1) / W, 6), round((y2 - y1) / H, 6)],
                "review": bool(review), "source": r["source"],
            })
        return {
            "width": W, "height": H,
            "num_detections": sum(not x["review"] for x in dets),
            "num_review": sum(x["review"] for x in dets),
            "detections": dets,
        }

    def detect_batch(self, images, min_box=8):
        """Detect on many images. Returns list of per-image result dicts (same
        schema as detect()). Each image is handled independently."""
        return [self.detect(im, min_box=min_box) for im in images]

    # --------------------------------------------------------------- draw
    def draw(self, image, result, max_side=1600):
        """Return a PIL image with `result` boxes drawn. Confident boxes are
        coloured per class; review boxes are grey and prefixed '?'."""
        if isinstance(image, (str, Path)):
            img = ImageOps.exif_transpose(Image.open(image).convert("RGB"))
        else:
            img = image.convert("RGB")
        W, H = img.size
        s = max_side / max(W, H) if max(W, H) > max_side else 1.0
        img = img.resize((round(W * s), round(H * s)))
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(13, img.size[0] // 90))
        except Exception:
            font = ImageFont.load_default()
        for x in result["detections"]:
            x1, y1, x2, y2 = (v * s for v in x["bbox"])
            if x["review"]:
                col, label = (130, 130, 130), f"?{x['class']}"
            else:
                col, label = PALETTE[x["class_id"] % len(PALETTE)], f"{x['class']} {x['score']:.2f}"
            d.rectangle([x1, y1, x2, y2], outline=col, width=3)
            tb = d.textbbox((x1, y1), label, font=font)
            d.rectangle([tb[0], tb[1], tb[2] + 4, tb[3] + 2], fill=col)
            d.text((x1 + 2, y1), label, fill=(255, 255, 255), font=font)
        return img

    # ------------------------------------------------------------- shutdown
    def shutdown(self):
        """Release models and free memory."""
        for attr in ("_gd", "_gd_proc", "_emb", "_ocr"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        try:
            import torch
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
