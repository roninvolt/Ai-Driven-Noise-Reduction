from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.denoiser.infer import run_inference


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the denoising pipeline on one image.")
    parser.add_argument(
        "--image",
        default=None,
        help="Path to input image. Defaults to data/sample.png if omitted.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional trained classifier checkpoint path.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device (e.g. cpu, cuda).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    fallback_path = project_root / "data" / "sample.png"
    image_path = Path(args.image).expanduser() if args.image else fallback_path

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read image: {image_path}")

    output = run_inference(image=image, checkpoint_path=args.checkpoint, device=args.device)
    print("Predicted noise:", output.predicted_noise)
