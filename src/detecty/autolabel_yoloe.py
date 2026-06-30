"""EXPERIMENTAL: YOLOE-26x visual-prompt few-shot labeler.

Found UNRELIABLE for cross-image labeling of cluttered scenes (low recall + large
false positives); kept for experimentation. Prefer detecty-label (ensemble).
NOTE: YOLOE needs visual_prompts['cls'] to be LOCAL sequential 0..k-1 — this
module remaps to global config ids. Device defaults to CPU (1 GB GPU too small).
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageOps
from torchvision.ops import nms

from ._resources import default_config, default_visual_prompts

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def exif_fix_to(src, dst):
    img = ImageOps.exif_transpose(Image.open(src).convert("RGB"))
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, quality=95)
    return img.size


def main(argv=None):
    ap = argparse.ArgumentParser(description="Experimental YOLOE-26x few-shot labeler.")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", default="dataset/raw_yoloe")
    ap.add_argument("--config", default=None)
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--model", default="yoloe-26x-seg.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.config or default_config()))
    names = {c["id"]: c["name"] for c in cfg["classes"]}
    d = cfg["defaults"]
    nms_iou, min_area, max_area = d["nms_iou"], d.get("min_box_area_frac", 0.0), d.get("max_box_area_frac", 1.0)
    prompts = yaml.safe_load(open(args.prompts or default_visual_prompts()))
    ref_root = Path(prompts.get("images", "object"))

    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
    print(f"Loading {args.model} on {args.device} ...")
    model = YOLOE(args.model)

    work = Path(args.out) / "_work"
    out_img, out_lbl = Path(args.out) / "images", Path(args.out) / "labels"
    out_img.mkdir(parents=True, exist_ok=True); out_lbl.mkdir(parents=True, exist_ok=True)

    src_imgs = sorted(p for p in Path(args.images_dir).iterdir() if p.suffix.lower() in IMG_EXTS)
    if args.limit:
        src_imgs = src_imgs[: args.limit]
    targets, tsize = [], {}
    for p in src_imgs:
        wp = work / "targets" / f"{p.stem}.jpg"
        tsize[p.stem] = exif_fix_to(p, wp)
        targets.append(wp)
    if not targets:
        print(f"No images in {args.images_dir}")
        return
    acc = {p.stem: [] for p in targets}

    for ri, ref in enumerate(prompts["references"]):
        rwp = work / "refs" / f"{Path(ref['image']).stem}.jpg"
        exif_fix_to(ref_root / ref["image"], rwp)
        boxes = ref["boxes"]
        local2global = [b["cls"] for b in boxes]
        vp = dict(bboxes=np.array([b["xyxy"] for b in boxes], dtype=float),
                  cls=np.arange(len(boxes), dtype=int))
        print(f"[ref {ri+1}/{len(prompts['references'])}] {ref['image']}: "
              + ", ".join(names.get(g, str(g)) for g in local2global))
        results = model.predict([str(t) for t in targets], refer_image=str(rwp), visual_prompts=vp,
                                predictor=YOLOEVPSegPredictor, imgsz=args.imgsz, conf=args.conf,
                                device=args.device, verbose=False)
        for t, res in zip(targets, results):
            if res.boxes is None or len(res.boxes) == 0:
                continue
            for (x1, y1, x2, y2), sc, lc in zip(res.boxes.xyxy.cpu().numpy(),
                                                res.boxes.conf.cpu().numpy(),
                                                res.boxes.cls.cpu().numpy().astype(int)):
                g = local2global[lc] if lc < len(local2global) else int(lc)
                acc[t.stem].append((float(x1), float(y1), float(x2), float(y2), float(sc), int(g)))

    total = 0
    for t in targets:
        W, H = tsize[t.stem]
        dets = acc[t.stem]
        kept = []
        if dets:
            keep = nms(torch.tensor([x[:4] for x in dets], dtype=torch.float32),
                       torch.tensor([x[4] for x in dets], dtype=torch.float32), nms_iou).tolist()
            ai = float(W * H)
            for i in keep:
                x1, y1, x2, y2, sc, g = dets[i]
                x1, x2 = sorted((max(0.0, x1), min(float(W), x2)))
                y1, y2 = sorted((max(0.0, y1), min(float(H), y2)))
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                af = (x2 - x1) * (y2 - y1) / ai
                if af < min_area or af > max_area:
                    continue
                kept.append((g, x1, y1, x2, y2))
        (out_lbl / f"{t.stem}.txt").write_text(
            "\n".join(f"{g} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}"
                      for g, x1, y1, x2, y2 in kept) + ("\n" if kept else ""))
        shutil.copy2(t, out_img / f"{t.stem}.jpg")
        total += len(kept)
        print(f"{t.stem}: {len(kept)} boxes")
    shutil.rmtree(work, ignore_errors=True)
    print(f"\nDone. {total} boxes across {len(targets)} images -> {args.out}/")


if __name__ == "__main__":
    main()
