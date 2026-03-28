"""Dataset utilities for noise classification."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from PIL import Image
from torch.utils.data import Dataset

LABELS = ["gaussian", "salt_pepper", "speckle", "periodic"]


class NoiseDataset(Dataset):
    """Image-folder dataset with fixed label order for noise classes."""

    def __init__(
        self,
        root_dir: str | Path,
        labels: Sequence[str] | None = None,
        transform: Callable | None = None,
        extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
    ) -> None:
        self.root_dir = Path(root_dir)
        self.labels = list(labels) if labels is not None else list(LABELS)
        self.transform = transform
        self.extensions = tuple(ext.lower() for ext in extensions)

        self.class_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.samples: list[tuple[Path, int]] = []

        for label in self.labels:
            class_dir = self.root_dir / label
            if not class_dir.exists() or not class_dir.is_dir():
                continue

            for image_path in sorted(class_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in self.extensions:
                    self.samples.append((image_path, self.class_to_idx[label]))

        if not self.samples:
            raise FileNotFoundError(
                f"No images found under {self.root_dir}. Expected class folders: {self.labels}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, label_index = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label_index
