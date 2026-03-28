"""AI-first denoiser package with temporary legacy fallback routing."""

from .router import DENOISERS, get_denoiser

__all__ = ["DENOISERS", "get_denoiser"]
