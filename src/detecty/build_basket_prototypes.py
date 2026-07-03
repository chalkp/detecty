"""Build the basket prototype bank: one DINOv3 embedding per reference crop.

Reads tight reference crops laid out as ``<refs>/<class>/*.jpg`` (class in
``basket`` | ``hamper``) and writes an ``.npz`` of ``names`` + L2-normalized
``vecs`` consumed by :class:`detecty.BasketDetector`.

Deliberately embed-only — no colour histogram, no catalog fallback: the basket
detector is a small, discrete pipeline, not the 30-class ensemble.

    detecty-build-basket-prototypes --refs-dir baskets --out basket_prototypes.npz
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image

from .embedder import DEFAULT_MODEL, Embedder

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build(refs_dir, out, model=DEFAULT_MODEL, device="cpu"):
    """Embed every ``<refs_dir>/<class>/*`` crop; save ``names`` + ``vecs`` to ``out``."""
    emb = Embedder(model, device)
    print(f"DINOv3: {emb.name} input {emb.input_size}")
    names, vecs = [], []
    for cls in sorted(os.listdir(refs_dir)):
        cdir = os.path.join(refs_dir, cls)
        if not os.path.isdir(cdir):
            continue
        n = 0
        for p in sorted(glob.glob(os.path.join(cdir, "*"))):
            if os.path.splitext(p)[1].lower() not in IMG_EXTS:
                continue
            vecs.append(emb.embed(Image.open(p).convert("RGB")))
            names.append(cls)
            n += 1
        print(f"  {cls}: {n} prototypes")
    if not vecs:
        raise SystemExit(f"No basket reference images under {refs_dir}/<class>/*")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez(out, names=np.array(names), vecs=np.stack(vecs).astype(np.float32))
    print(f"Saved {out}: {len(names)} basket prototypes over {len(set(names))} classes")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the DINOv3 basket prototype bank.")
    ap.add_argument("--refs-dir", default="baskets",
                    help="dir of <class>/*.jpg reference crops (basket|hamper)")
    ap.add_argument("--out", default="basket_prototypes.npz")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="DINOv3 backbone")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    build(args.refs_dir, args.out, args.model, args.device)


if __name__ == "__main__":
    main()
