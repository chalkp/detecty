"""Draw YOLO boxes onto images for visual QA."""
import argparse
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

from ._resources import default_config

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PALETTE = [
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
    (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190), (0, 128, 128),
    (170, 110, 40), (255, 215, 0), (128, 0, 0), (170, 255, 195), (0, 0, 128),
    (128, 128, 0), (255, 99, 71), (46, 139, 87), (30, 144, 255), (218, 112, 214),
    (160, 82, 45), (199, 21, 133), (47, 79, 79), (255, 140, 0), (0, 191, 255),
    (220, 20, 60), (34, 139, 34), (138, 43, 226), (210, 105, 30), (1, 50, 32),
]


def load_names(config):
    cfg = yaml.safe_load(open(config))
    return {c["id"]: c["name"] for c in cfg["classes"]}


def draw(img_path, label_path, names, max_side):
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    if max(W, H) > max_side:
        s = max_side / max(W, H)
        img = img.resize((round(W * s), round(H * s)))
    dw, dh = img.size
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(14, dw // 60))
    except Exception:
        font = ImageFont.load_default()
    n = 0
    if Path(label_path).exists():
        for line in Path(label_path).read_text().splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            cid, cx, cy, w, h = int(p[0]), *map(float, p[1:])
            x1, y1, x2, y2 = (cx - w / 2) * dw, (cy - h / 2) * dh, (cx + w / 2) * dw, (cy + h / 2) * dh
            color = PALETTE[cid % len(PALETTE)]
            d.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label = names.get(cid, str(cid))
            tb = d.textbbox((x1, y1), label, font=font)
            d.rectangle([tb[0], tb[1], tb[2] + 4, tb[3] + 2], fill=color)
            d.text((x1 + 2, y1), label, fill=(255, 255, 255), font=font)
            n += 1
    return img, n


def main(argv=None):
    ap = argparse.ArgumentParser(description="Visualize YOLO labels for QA.")
    ap.add_argument("--raw", required=True, help="dir with images/ and labels/")
    ap.add_argument("--out", default="qa_preview")
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-side", type=int, default=1280)
    args = ap.parse_args(argv)

    names = load_names(args.config or default_config())
    raw = Path(args.raw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in (raw / "images").iterdir() if p.suffix.lower() in IMG_EXTS)
    total = 0
    for p in imgs:
        img, n = draw(p, raw / "labels" / f"{p.stem}.txt", names, args.max_side)
        img.save(out / f"{p.stem}.png")
        total += n
        print(f"{p.name}: {n} boxes")
    print(f"\n{total} boxes drawn across {len(imgs)} images -> {out}/")


if __name__ == "__main__":
    main()
