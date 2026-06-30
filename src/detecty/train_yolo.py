"""Train a YOLO detector on the auto-labeled dataset (Ultralytics).

Note: a 1 GB GPU is far too small to train; use a bigger GPU / cluster
(see the packaged train.slurm: `python -c "import detecty,os;
print(os.path.join(os.path.dirname(detecty.__file__),'data','train.slurm'))"`).
"""
import argparse


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train YOLO on the auto-labeled dataset.")
    ap.add_argument("--data", required=True, help="path to dataset.yaml")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="'0', '0,1', or 'cpu'")
    ap.add_argument("--project", default="runs/robocup")
    ap.add_argument("--name", default="yolo_autolabel")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, project=args.project, name=args.name, workers=args.workers,
                patience=args.patience, resume=args.resume,
                hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, fliplr=0.5, mosaic=1.0, close_mosaic=10)
    metrics = model.val(data=args.data, device=args.device)
    print("mAP50-95:", getattr(metrics.box, "map", "n/a"), "| mAP50:", getattr(metrics.box, "map50", "n/a"))


if __name__ == "__main__":
    main()
