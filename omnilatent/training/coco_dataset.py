"""COCO Captions dataset for text <-> image training.

Loads COCO-format image-caption pairs and returns them in the standard
OmniLatent multi-modal dict format (compatible with collate_multimodal
and _train_step_synthetic).

Expected directory layout:
    data/
      train2014/
        COCO_train2014_000000000001.jpg
        ...
      annotations/
        captions_train2014.json

Download:
    wget http://images.cocodataset.org/zips/train2014.zip
    wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
    unzip train2014.zip
    unzip annotations_trainval2014.zip
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from omnilatent.config import OmniLatentConfig

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

try:
    from torchvision import transforms as T
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False


def _simple_tokenize(text: str, max_len: int, vocab_size: int) -> torch.Tensor:
    """Byte-level fallback tokenizer (same as video_dataset.py).

    Token 0 = padding, token 1 = BOS.  Content tokens are 2..vocab_size-1.
    Replace with SentencePiece / tiktoken for real training.
    """
    encoded = text.encode("utf-8")[:max_len - 1]  # leave room for BOS
    ids = [1] + [(b % (vocab_size - 2)) + 2 for b in encoded]  # BOS + content
    return torch.tensor(ids, dtype=torch.long)


class CocoCaptionsDataset(Dataset):
    """COCO Captions dataset for text <-> image training.

    Each sample returns {"image": (3, 224, 224), "text": (seq_len,)} which
    plugs directly into collate_multimodal and _train_step_synthetic.

    Args:
        image_dir: path to image directory (e.g., "data/train2014").
        annotation_file: path to captions JSON (e.g., "data/annotations/captions_train2014.json").
        config: OmniLatentConfig for image size and text settings.
        tokenizer_fn: optional (text, max_len) -> LongTensor tokenizer.
        augment: apply random horizontal flip and color jitter.
    """

    def __init__(
        self,
        image_dir: str | Path,
        annotation_file: str | Path,
        config: OmniLatentConfig,
        tokenizer_fn=None,
        augment: bool = True,
    ) -> None:
        if not _HAS_PIL:
            raise ImportError("Pillow is required: pip install Pillow")
        if not _HAS_TORCHVISION:
            raise ImportError("torchvision is required: pip install torchvision")

        self.config = config
        self.image_dir = Path(image_dir)
        self.tokenizer_fn = tokenizer_fn

        # Load annotations
        with open(annotation_file, "r") as f:
            data = json.load(f)

        # Build image_id -> filename mapping
        id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

        # Build list of (image_path, caption) pairs
        self.samples: list[tuple[Path, str]] = []
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            caption = ann["caption"].strip()
            if img_id in id_to_file and caption:
                img_path = self.image_dir / id_to_file[img_id]
                self.samples.append((img_path, caption))

        if not self.samples:
            raise RuntimeError(
                f"No valid image-caption pairs found. "
                f"Check that {image_dir} contains images and "
                f"{annotation_file} has matching annotations."
            )

        # Image transforms
        transform_list = [
            T.Resize((config.image_size, config.image_size)),
        ]
        if augment:
            transform_list += [
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ]
        transform_list += [
            T.ToTensor(),  # -> (C, H, W) float [0, 1]
        ]
        self.transform = T.Compose(transform_list)

        print(
            f"CocoCaptionsDataset: {len(self.samples)} image-caption pairs "
            f"from {self.image_dir}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _tokenize(self, text: str) -> torch.Tensor:
        c = self.config
        if self.tokenizer_fn is not None:
            return self.tokenizer_fn(text, c.text_max_len)
        return _simple_tokenize(text, c.text_max_len, c.vocab_size)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path, caption = self.samples[idx]

        # Load and transform image
        try:
            img = Image.open(img_path).convert("RGB")
            image_tensor = self.transform(img)  # (3, 224, 224)
        except Exception:
            # Fallback: black image (rare — corrupt file)
            image_tensor = torch.zeros(
                self.config.image_channels,
                self.config.image_size,
                self.config.image_size,
            )

        # Tokenize caption
        text_tensor = self._tokenize(caption)

        return {"image": image_tensor, "text": text_tensor}
