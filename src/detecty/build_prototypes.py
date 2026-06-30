"""Build the prototype bank: DINOv3 embedding + masked hue histogram per crop.

Sources:
  prototypes/<class>/*.jpg    your collected in-domain crops  (high trust)
  objects_gt/<cat>/<class>.*  official Incheon2026 photos     (fallback)
Output: prototypes.npz (consumed by detecty-label).
"""
import argparse
import glob
import os

import numpy as np
import yaml
from PIL import Image

from ._resources import default_config
from .embedder import DEFAULT_MODEL, Embedder
from .features import masked_hue_hist


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build DINOv3+colour prototype bank.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prototypes-dir", default="prototypes")
    ap.add_argument("--catalog-dir", default="objects_gt")
    ap.add_argument("--out", default="prototypes.npz")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.config or default_config()))
    valid = {c["name"] for c in cfg["classes"]}
    emb = Embedder(args.model, args.device)
    print(f"DINOv3: {emb.name} input {emb.input_size}")

    items = []
    if os.path.isdir(args.prototypes_dir):
        for cls in sorted(os.listdir(args.prototypes_dir)):
            for p in glob.glob(os.path.join(args.prototypes_dir, cls, "*")):
                if cls in valid and os.path.isfile(p):
                    items.append((cls, "indomain", p))
    for p in glob.glob(os.path.join(args.catalog_dir, "**", "*"), recursive=True):
        if os.path.isfile(p):
            cls = os.path.splitext(os.path.basename(p))[0]
            if cls in valid:
                items.append((cls, "catalog", p))
    if not items:
        print("No prototype images found.")
        return

    names, srcs, vecs, hues = [], [], [], []
    for cls, src, p in items:
        pil = Image.open(p).convert("RGB")
        vecs.append(emb.embed(pil))
        hues.append(masked_hue_hist(pil)[0])
        names.append(cls)
        srcs.append(src)
    np.savez(args.out, names=np.array(names), srcs=np.array(srcs),
             vecs=np.stack(vecs).astype(np.float32), hues=np.stack(hues).astype(np.float32))
    nin = srcs.count("indomain")
    print(f"Saved {args.out}: {len(names)} prototypes ({nin} in-domain + {len(names)-nin} catalog) "
          f"over {len(set(names))} classes")
    missing = sorted(valid - {n for n, s in zip(names, srcs) if s == "indomain"})
    if missing:
        print(f"Classes with NO in-domain crop (catalog only — weaker): {missing}")


if __name__ == "__main__":
    main()
