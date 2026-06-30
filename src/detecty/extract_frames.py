"""Sample frames from .mp4 videos (ffmpeg). Frame names carry the video stem so
the train/val split keeps each video's frames on one side (no leakage)."""
import argparse
import subprocess
from pathlib import Path


def extract(video: Path, out_dir: Path, every: float) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / f"{video.stem}_f%04d.jpg")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
        "-vf", f"fps=1/{every}", "-qscale:v", "2", pattern,
    ], check=True)
    return len(list(out_dir.glob(f"{video.stem}_f*.jpg")))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sample frames from videos for labeling.")
    ap.add_argument("--videos-dir", default="object")
    ap.add_argument("--out", default="frames")
    ap.add_argument("--every", type=float, default=1.0, help="seconds between frames")
    args = ap.parse_args(argv)

    vids = sorted(Path(args.videos_dir).glob("*.mp4"))
    if not vids:
        print(f"No .mp4 files in {args.videos_dir}")
        return
    out, total = Path(args.out), 0
    for v in vids:
        n = extract(v, out, args.every)
        total += n
        print(f"{v.name}: {n} frames")
    print(f"\nExtracted {total} frames -> {out}/")


if __name__ == "__main__":
    main()
