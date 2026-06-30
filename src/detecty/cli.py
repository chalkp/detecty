"""detecty-detect — run the detection pipeline on an image or folder.

Thin wrapper over the callable API (detecty.SamYolo). Prints/saves the result
dict(s) as JSON and optionally an annotated image.

    detecty-detect media/ultimate_test.jpg --out result.jpg --json result.json
    detecty-detect path/to/folder --out annotated/ --json results.json
"""
import argparse
import json
from pathlib import Path

from .embedder import DEFAULT_MODEL
from .pipeline import SamYolo

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run detecty object detection on an image/folder.")
    ap.add_argument("image", help="image file or directory of images")
    ap.add_argument("--out", default=None, help="annotated image path (file) or dir (folder input)")
    ap.add_argument("--json", default=None, help="write results JSON here (else print summary)")
    ap.add_argument("--protos", default="prototypes.npz")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--vlm", action="store_true", help="consult Gemma/vLLM on hardest crops (off by default)")
    ap.add_argument("--quantize-localizer", action="store_true", help="INT8 Grounding DINO (CPU)")
    ap.add_argument("--max-side", type=int, default=1600)
    args = ap.parse_args(argv)

    src = Path(args.image)
    if src.is_dir():
        imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)
    else:
        imgs = [src]
    if not imgs:
        print(f"No images at {src}")
        return

    det = SamYolo(protos=args.protos, dino_model=args.model, device=args.device,
                  use_ocr=not args.no_ocr, use_vlm=args.vlm,
                  quantize_localizer=args.quantize_localizer).setup()

    out_is_dir = src.is_dir() or (args.out and (args.out.endswith("/") or len(imgs) > 1))
    if args.out and out_is_dir:
        Path(args.out).mkdir(parents=True, exist_ok=True)

    results = {}
    for p in imgs:
        res = det.detect(str(p))
        results[p.name] = res
        n, r = res["num_detections"], res["num_review"]
        print(f"{p.name}: {n} detections, {r} review")
        for x in res["detections"]:
            tag = "REVIEW" if x["review"] else x["source"]
            print(f"    {x['class']:16s} score={x['score']:.2f} margin={x['margin']:.2f} "
                  f"[{tag}] bbox={x['bbox']}")
        if args.out:
            annotated = det.draw(str(p), res, max_side=args.max_side)
            dst = Path(args.out) / f"{p.stem}_result.jpg" if out_is_dir else Path(args.out)
            annotated.save(dst, quality=92)
            print(f"    -> {dst}")
    det.shutdown()

    if args.json:
        payload = results if len(imgs) > 1 else next(iter(results.values()))
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"JSON -> {args.json}")


if __name__ == "__main__":
    main()
