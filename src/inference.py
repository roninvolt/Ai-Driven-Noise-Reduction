"""Compatibility inference entrypoints.

Prefer src.denoiser.infer for new code.
"""

from src.denoiser.infer import run_inference, run_inference_from_path

__all__ = ["run_inference", "run_inference_from_path"]
