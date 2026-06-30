"""Grounding DINO localizer / baseline labeler.

Modes:
  generic  (default) — ONE forward pass with a combined generic object prompt;
                       writes class-agnostic boxes (class id 0). Fast; intended
                       as the localizer feeding the ensemble (detecty-label).
  classes            — one pass per class with that class's text prompt; writes
                       class ids. The text-only baseline labeler (brand-blind).

Device defaults to CPU: on a 1 GB GPU Grounding DINO will not fit.
"""
import argparse
import shutil
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageOps
from torchvision.ops import nms
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from ._resources import default_config

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GENERIC_PROMPTS = [
    "bottle", "can", "cup", "mug", "bowl", "plate", "box", "carton", "fruit",
    "snack bag", "sponge", "cube", "tube", "fork", "knife", "spoon", "object",
]


def load_config(path):
    cfg = yaml.safe_load(open(path))
    d = cfg["defaults"]
    for c in cfg["classes"]:
        c.setdefault("box_threshold", d["box_threshold"])
        c.setdefault("text_threshold", d["text_threshold"])
    return cfg


def _post_process(processor, outputs, input_ids, box_thr, text_thr, target_size):
    fn = processor.post_process_grounded_object_detection
    sizes = [target_size]
    for kwargs in (dict(box_threshold=box_thr, text_threshold=text_thr),
                   dict(threshold=box_thr, text_threshold=text_thr),
                   dict(threshold=box_thr)):
        try:
            return fn(outputs, input_ids=input_ids, target_sizes=sizes, **kwargs)[0]
        except TypeError:
            continue
    return fn(outputs, input_ids, box_thr, text_thr, sizes)[0]


@torch.no_grad()
def _run_prompt(img_small, prompt, model, processor, device, box_thr, text_thr, target_HW):
    prompt = prompt.strip().lower()
    if not prompt.endswith("."):
        prompt += "."
    inputs = processor(images=img_small, text=prompt, return_tensors="pt").to(device)
    outputs = model(**inputs)
    return _post_process(processor, outputs, inputs["input_ids"], box_thr, text_thr, target_HW)


def detect(img, model, processor, device, long_side, cfg, mode, nms_iou, min_area, max_area):
    W, H = img.size
    scale = long_side / max(W, H) if max(W, H) > long_side else 1.0
    small = img.resize((max(1, round(W * scale)), max(1, round(H * scale)))) if scale < 1.0 else img
    d = cfg["defaults"]
    boxes, scores, cls_ids = [], [], []
    if mode == "generic":
        prompt = " . ".join(GENERIC_PROMPTS)
        res = _run_prompt(small, prompt, model, processor, device,
                          d["box_threshold"], d["text_threshold"], (H, W))
        for b, s in zip(res["boxes"], res["scores"]):
            boxes.append(b.tolist()); scores.append(float(s)); cls_ids.append(0)
    else:
        for c in cfg["classes"]:
            res = _run_prompt(small, c["prompt"], model, processor, device,
                              c["box_threshold"], c["text_threshold"], (H, W))
            for b, s in zip(res["boxes"], res["scores"]):
                boxes.append(b.tolist()); scores.append(float(s)); cls_ids.append(c["id"])
    if not boxes:
        return []
    keep = nms(torch.tensor(boxes, dtype=torch.float32),
               torch.tensor(scores, dtype=torch.float32), nms_iou)
    area = float(W * H)
    out = []
    for i in keep.tolist():
        x1, y1, x2, y2 = boxes[i]
        x1, x2 = sorted((max(0.0, x1), min(float(W), x2)))
        y1, y2 = sorted((max(0.0, y1), min(float(H), y2)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        af = (x2 - x1) * (y2 - y1) / area
        if af < min_area or af > max_area:
            continue
        out.append((x1, y1, x2, y2, scores[i], cls_ids[i]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Grounding DINO localizer / baseline labeler.")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", default="dataset/raw_gd")
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", choices=["generic", "classes"], default="generic")
    ap.add_argument("--device", default="cpu", help="cpu (1 GB GPU cannot fit GD)")
    ap.add_argument("--quantize", action="store_true",
                    help="dynamic INT8 (CPU): ~2.7x smaller checkpoint, IoU~0.90, "
                         "but lower recall -> thresholds auto-lowered to compensate")
    ap.add_argument("--quant-thr-comp", type=float, default=0.05,
                    help="subtract this from box/text thresholds when --quantize")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = load_config(args.config or default_config())
    d = cfg["defaults"]
    mid = cfg["model"]["id"]
    long_side = cfg["model"]["inference_longest_side"]
    print(f"Device: {args.device} | Model: {mid} | Mode: {args.mode}"
          + (" | INT8" if args.quantize else ""))

    processor = AutoProcessor.from_pretrained(mid)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(args.device).eval()
    if args.quantize:
        if args.device != "cpu":
            print("--quantize forces CPU (qint8 kernels are CPU-only).")
        import torch as _t
        model = _t.quantization.quantize_dynamic(model.to("cpu"), {_t.nn.Linear}, dtype=_t.qint8)
        args.device = "cpu"
        comp = args.quant_thr_comp
        d["box_threshold"] = max(0.0, d["box_threshold"] - comp)
        d["text_threshold"] = max(0.0, d["text_threshold"] - comp)
        for c in cfg["classes"]:
            c["box_threshold"] = max(0.0, c["box_threshold"] - comp)
            c["text_threshold"] = max(0.0, c["text_threshold"] - comp)

    imgs = sorted(p for p in Path(args.images_dir).iterdir() if p.suffix.lower() in IMG_EXTS)
    if args.limit:
        imgs = imgs[: args.limit]
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    total = 0
    for i, p in enumerate(imgs, 1):
        img = ImageOps.exif_transpose(Image.open(p).convert("RGB"))
        W, H = img.size
        dets = detect(img, model, processor, args.device, long_side, cfg, args.mode,
                      d["nms_iou"], d.get("min_box_area_frac", 0.0), d.get("max_box_area_frac", 1.0))
        lines = []
        for x1, y1, x2, y2, _s, cid in dets:
            lines.append(f"{cid} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
        (out / "labels" / f"{p.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        dst = out / "images" / p.name
        if not dst.exists():
            shutil.copy2(p, dst)
        total += len(dets)
        print(f"[{i}/{len(imgs)}] {p.name}: {len(dets)} boxes")
    print(f"\nDone. {total} boxes across {len(imgs)} images -> {out}/")


if __name__ == "__main__":
    main()
