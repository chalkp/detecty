"""Ensemble auto-labeler (RECOMMENDED).

Localize with Grounding DINO (detecty-localize), then classify each crop by
fusing DINOv3-L nearest-prototype + masked HSV colour + OCR brand-match.
Uncertain crops are routed to <out>/review/ instead of being guessed.

Optional, OFF by default: --vlm consults a Gemma-3n-E4B/vLLM model (see
vlm_consult.py) on the very hardest crops only. Use with care.

Device defaults to CPU (1 GB GPU cannot fit DINOv3-L / Grounding DINO).
"""
import argparse
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageOps

from . import vlm_consult
from ._resources import default_config, default_ensemble
from .embedder import DEFAULT_MODEL, Embedder
from .features import hist_intersection, masked_hue_hist

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_FRAME_RE = re.compile(r"^(?P<vid>.+)_f\d{3,}$")
PHOTO_TS_RE = re.compile(r"(?P<date>\d{8})_(?P<time>\d{6})")


def scene_key(stem):
    m = VIDEO_FRAME_RE.match(stem)
    if m:
        return f"vid:{m.group('vid')}"
    m = PHOTO_TS_RE.search(stem)
    if not m:
        return f"single:{stem}"
    secs = int(m["time"][:2]) * 3600 + int(m["time"][2:4]) * 60 + int(m["time"][4:6])
    return f"photo:{m['date']}:{secs // 120}"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ensemble (DINOv3+colour+OCR) auto-labeler.")
    ap.add_argument("--boxes-from", required=True, help="dir with images/ and labels/ (from detecty-localize)")
    ap.add_argument("--out", default="dataset/raw")
    ap.add_argument("--config", default=None)
    ap.add_argument("--ensemble", default=None)
    ap.add_argument("--protos", default="prototypes.npz")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--multiview", action="store_true", help="propagate confident scene-sibling labels")
    ap.add_argument("--vlm", action="store_true", help="consult Gemma/vLLM on hardest crops (OFF by default)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.config or default_config()))
    id_of = {c["name"]: c["id"] for c in cfg["classes"]}
    ecfg = yaml.safe_load(open(args.ensemble or default_ensemble()))
    F = ecfg["fusion"]
    brand_kw = ecfg.get("brand_keywords", {})
    cat_of = ecfg.get("category_of", {})

    pz = np.load(args.protos, allow_pickle=True)
    p_names, p_srcs, p_vecs, p_hues = pz["names"], pz["srcs"], pz["vecs"], pz["hues"]
    by_cls = defaultdict(list)
    for i, n in enumerate(p_names):
        by_cls[str(n)].append(i)
    print(f"{len(p_names)} prototypes over {len(by_cls)} classes")

    emb = Embedder(args.model, args.device)
    penalty = np.array([F["catalog_penalty"] if s == "catalog" else 0.0 for s in p_srcs])

    reader = None
    if not args.no_ocr:
        try:
            import easyocr
            reader = easyocr.Reader(ecfg["ocr"]["langs"], gpu=(args.device != "cpu"), verbose=False)
        except Exception as e:
            print(f"OCR disabled ({e})")
    if args.vlm and not vlm_consult.available():
        print("--vlm requested but 'openai' not installed; VLM consult disabled.")

    def ocr_hits(crop):
        if reader is None:
            return {}
        txt = " ".join(t.lower() for _, t, c in reader.readtext(np.array(crop))
                       if c >= ecfg["ocr"]["min_conf"])
        hits = {}
        for cls, kws in brand_kw.items():
            if any(kw.lower() in txt for kw in kws):
                hits[cls] = 1.0
        return hits

    def classify(crop):
        v = emb.embed(crop)
        hh = masked_hue_hist(crop)[0]
        sims = (p_vecs @ v) - penalty
        e = {c: max(sims[i] for i in idx) for c, idx in by_cls.items()}
        col = {c: max(hist_intersection(hh, p_hues[i]) for i in idx) for c, idx in by_cls.items()}
        ocr = ocr_hits(crop)
        rank = sorted(e, key=e.get, reverse=True)
        plausible = set(rank[:8])
        score = {}
        for c in e:
            s = e[c] + F["w_color"] * col.get(c, 0.0)
            if c in ocr and (c in plausible or cat_of.get(c) == cat_of.get(rank[0])):
                s += F["w_ocr"] * ocr[c]
            score[c] = s
        order = sorted(score, key=score.get, reverse=True)
        margin = score[order[0]] - (score[order[1]] if len(order) > 1 else 0.0)
        return dict(cls=order[0], score=score[order[0]], margin=margin,
                    top3=order[:3], vec=v)

    # ---- gather + classify crops ----
    bf = Path(args.boxes_from)
    img_dir, lbl_dir = bf / "images", bf / "labels"
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if args.limit:
        imgs = imgs[: args.limit]
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    review = out / "review"; review.mkdir(parents=True, exist_ok=True)

    dets = defaultdict(list)
    for p in imgs:
        lbl = lbl_dir / f"{p.stem}.txt"
        if not lbl.exists():
            continue
        im = ImageOps.exif_transpose(Image.open(p).convert("RGB"))
        W, H = im.size
        for line in lbl.read_text().splitlines():
            t = line.split()
            if len(t) != 5:
                continue
            _, cx, cy, w, h = t[0], *map(float, t[1:])
            box = ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)
            crop = im.crop((max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])))
            if crop.width < 8 or crop.height < 8:
                continue
            r = classify(crop)
            r.update(box=box, W=W, H=H, crop=crop)
            dets[p.stem].append(r)
        print(f"{p.stem}: {len(dets[p.stem])} crops classified")

    # ---- multi-view: propagate confident scene-sibling label to low-margin crops ----
    if args.multiview:
        scenes = defaultdict(list)
        for stem, ds in dets.items():
            for i in range(len(ds)):
                scenes[scene_key(stem)].append((stem, i))
        moved = 0
        for members in scenes.values():
            conf = [(s, i) for s, i in members if dets[s][i]["margin"] >= F["margin_review"]]
            for s, i in members:
                d = dets[s][i]
                if d["margin"] >= F["margin_review"]:
                    continue
                best = None
                for cs, ci in conf:
                    if cs == s:
                        continue
                    sim = float(dets[cs][ci]["vec"] @ d["vec"])
                    if sim > 0.6 and (best is None or sim > best[0]):
                        best = (sim, dets[cs][ci]["cls"])
                if best:
                    d["cls"], d["margin_src"] = best[1], "multiview"
                    moved += 1
        if moved:
            print(f"multiview: re-labeled {moved} low-margin crops from scene siblings")

    # ---- optional VLM consult on the hardest crops (off by default) ----
    if args.vlm and vlm_consult.available():
        hard_thr = F["margin_review"] / 2.0
        used = 0
        for stem, ds in dets.items():
            for d in ds:
                if d.get("margin_src") == "multiview" or d["margin"] >= hard_thr:
                    continue
                pick = vlm_consult.consult(d["crop"], d["top3"],
                                           hints="RoboCup@Home object; read the brand/logo.")
                if pick:
                    d["cls"], d["margin_src"] = pick, "vlm"
                    used += 1
        if used:
            print(f"vlm: resolved {used} hard crops")

    # ---- write labels + review ----
    total = nrev = 0
    for p in imgs:
        ds = dets.get(p.stem, [])
        lines = []
        for k, d in enumerate(ds):
            W, H = d["W"], d["H"]
            x1, y1, x2, y2 = d["box"]
            resolved = d.get("margin_src") in ("multiview", "vlm")
            uncertain = (not resolved) and (d["margin"] < F["margin_review"] or d["score"] < F["min_top_score"])
            if uncertain:
                d["crop"].save(review / f"{p.stem}_{k}_{'-'.join(d['top3'])}.jpg")
                nrev += 1
                continue
            cid = id_of[d["cls"]]
            lines.append(f"{cid} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
        (out / "labels" / f"{p.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        shutil.copy2(img_dir / p.name, out / "images" / p.name)
        total += len(lines)
        msg = f"{p.stem}: {len(lines)} labels"
        if len(ds) - len(lines):
            msg += f", {len(ds)-len(lines)} -> review"
        print(msg)

    print(f"\nDone. {total} labels, {nrev} crops -> {review}/  ({args.out})")


if __name__ == "__main__":
    main()
