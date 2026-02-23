"""Dataset registry: build datasets from config.

Lazy-imports dataset implementations so that missing optional dependencies
don't crash import time.
"""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset

from omnilatent.config import OmniLatentConfig


def build_dataset(
    name: str,
    config: OmniLatentConfig,
    **kwargs: Any,
) -> Dataset:
    """Build a dataset by name.

    Supported names:
      - "synthetic": in-memory synthetic data (no extra deps)
      - "coco": COCO Captions (requires torchvision, PIL)
      - "video": Video watching dataset (requires torchaudio, torchvision)
      - "pdf": PDF dataset (requires pymupdf)

    Args:
        name: dataset name (case-insensitive).
        config: OmniLatent config.
        **kwargs: additional arguments passed to the dataset constructor.

    Returns:
        A PyTorch Dataset instance.
    """
    name = name.lower().strip()

    if name == "synthetic":
        from omnilatent.training.data import SyntheticMultiModalDataset
        return SyntheticMultiModalDataset(config, **kwargs)

    elif name == "coco":
        try:
            from omnilatent.training.coco_dataset import COCOCaptionsDataset
        except (ImportError, OSError, RuntimeError) as e:
            raise ImportError(
                f"COCO dataset requires torchvision and PIL. "
                f"Install with: pip install 'omnilatent[coco]'. "
                f"Original error: {e}"
            ) from e
        return COCOCaptionsDataset(config=config, **kwargs)

    elif name == "video":
        try:
            from omnilatent.training.video_dataset import VideoWatchingDataset
        except (ImportError, OSError, RuntimeError) as e:
            raise ImportError(
                f"Video dataset requires torchaudio and torchvision. "
                f"Install with: pip install 'omnilatent[video]'. "
                f"Original error: {e}"
            ) from e
        return VideoWatchingDataset(config=config, **kwargs)

    elif name == "pdf":
        try:
            from omnilatent.training.pdf_dataset import PDFDataset
        except (ImportError, OSError, RuntimeError) as e:
            raise ImportError(
                f"PDF dataset requires pymupdf. "
                f"Install with: pip install 'omnilatent[pdf]'. "
                f"Original error: {e}"
            ) from e
        return PDFDataset(config=config, **kwargs)

    else:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available: synthetic, coco, video, pdf"
        )
