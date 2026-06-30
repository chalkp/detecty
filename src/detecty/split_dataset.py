"""Scene-grouped train/val split (+ dataset.yaml) — no burst/video leakage."""
import argparse
import random
import re
import shutil
from pathlib import Path

import yaml

from ._resources import default_config

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_FRAME_RE = re.compile(r"^(?P<vid>.+)_f\d{3,}$")
PHOTO_TS_RE = re.compile(r"(?P<date>\d{8})_(?P<time>\d{6})")


def scene_groups(stems, gap):
    groups, photos = {}, []
    for s in stems:
        m = VIDEO_FRAME_RE.match(s)
        if m:
            groups[s] = f"vid:{m.group('vid')}"
            continue
        m = PHOTO_TS_RE.search(s)
        if m:
            secs = int(m["time"][:2]) * 3600 + int(m["time"][2:4]) * 60 + int(m["time"][4:6])
            photos.append((m["date"], secs, s))
        else:
            groups[s] = f"single:{s}"
    photos.sort()
    gi, prev = 0, None
    for date, secs, s in photos:
        if prev is None or date != prev[0] or secs - prev[1] > gap:
            gi += 1
        groups[s] = f"photo:{date}:{gi}"
        prev = (date, secs)
    return groups


def copy_pair(stem, src_img, raw, out_split):
    (out_split / "images").mkdir(parents=True, exist_ok=True)
    (out_split / "labels").mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_img, out_split / "images" / src_img.name)
    lbl = raw / "labels" / f"{stem}.txt"
    dst = out_split / "labels" / f"{stem}.txt"
    dst.write_text(lbl.read_text() if lbl.exists() else "")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Scene-grouped train/val split + dataset.yaml.")
    ap.add_argument("--raw", default="dataset/raw")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--config", default=None)
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--gap", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    raw = Path(args.raw)
    imgs = {p.stem: p for p in (raw / "images").iterdir() if p.suffix.lower() in IMG_EXTS}
    if not imgs:
        print(f"No images in {raw/'images'}")
        return
    groups = scene_groups(list(imgs), args.gap)
    by_group = {}
    for stem in imgs:
        by_group.setdefault(groups[stem], []).append(stem)

    rng = random.Random(args.seed)
    order = sorted(by_group); rng.shuffle(order)
    target, val_groups, n_val = args.val * len(imgs), set(), 0
    for g in order:
        if n_val < target:
            val_groups.add(g); n_val += len(by_group[g])

    out = Path(args.out)
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            d = out / split / sub
            if d.exists():
                shutil.rmtree(d)
    n_train = 0
    for g, stems in by_group.items():
        split = "val" if g in val_groups else "train"
        for stem in stems:
            copy_pair(stem, imgs[stem], raw, out / split)
            n_train += split == "train"

    names = [c["name"] for c in sorted(yaml.safe_load(open(args.config or default_config()))["classes"],
                                       key=lambda c: c["id"])]
    yaml.safe_dump({"path": str(out.resolve()), "train": "train/images", "val": "val/images",
                    "nc": len(names), "names": names}, open(out / "dataset.yaml", "w"), sort_keys=False)
    print(f"Scenes: {len(by_group)} -> train {n_train} / val {n_val} imgs")
    print(f"Wrote {out}/dataset.yaml ({len(names)} classes)")


if __name__ == "__main__":
    main()
