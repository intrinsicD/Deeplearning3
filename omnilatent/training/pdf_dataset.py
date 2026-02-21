"""PDF dataset for scientific literature training.

Extracts text and images from scientific PDFs, creating multi-modal
training pairs:
  * text → text  (section reconstruction, abstract completion)
  * image → text (figure → caption mapping)
  * text → image (caption → figure mapping)

Uses PyMuPDF (fitz) for fast, high-quality extraction of both text
blocks and embedded images.  Handles common scientific PDF layouts
including two-column papers, extracted figures, and section structure.

Tokenization uses the same byte-level fallback as video_dataset.py.
Replace with SentencePiece / tiktoken / HuggingFace tokenizers for
production quality.
"""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from omnilatent.config import OmniLatentConfig

# Lazy import — fitz (PyMuPDF) is an optional dependency
_fitz = None


def _get_fitz():
    global _fitz
    if _fitz is None:
        try:
            import fitz as _fitz_module

            _fitz = _fitz_module
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF processing. "
                "Install it with: pip install PyMuPDF"
            )
    return _fitz


# -------------------------------------------------------------------------
# Byte-level tokenizer (same fallback as video_dataset.py)
# -------------------------------------------------------------------------
def _byte_tokenize(text: str, max_len: int) -> torch.Tensor:
    """Encode text as byte values (1-255), 0 = padding.  Simple fallback."""
    encoded = list(text.encode("utf-8", errors="replace"))[:max_len]
    token_ids = [min(b + 1, 255) for b in encoded]  # shift so 0 = pad
    if len(token_ids) < max_len:
        token_ids += [0] * (max_len - len(token_ids))
    return torch.tensor(token_ids, dtype=torch.long)


# -------------------------------------------------------------------------
# PDF extraction helpers
# -------------------------------------------------------------------------
def extract_pdf_text_blocks(pdf_path: str | Path) -> list[str]:
    """Extract text blocks from a PDF, grouped by page.

    Returns a list of non-empty text strings (paragraphs / blocks).
    """
    fitz = _get_fitz()
    doc = fitz.open(str(pdf_path))
    blocks: list[str] = []

    for page in doc:
        page_blocks = page.get_text("blocks")  # list of (x0,y0,x1,y1, text, block_no, block_type)
        for block in page_blocks:
            # block_type 0 = text, 1 = image
            if block[6] == 0:
                text = block[4].strip()
                if len(text) > 20:  # skip tiny fragments
                    blocks.append(text)

    doc.close()
    return blocks


def extract_pdf_images(
    pdf_path: str | Path,
    min_size: int = 50,
    target_size: int = 224,
) -> list[torch.Tensor]:
    """Extract images from a PDF as normalized tensors.

    Returns list of (C, H, W) float32 tensors, resized to target_size.
    Skips images smaller than min_size in either dimension.
    """
    fitz = _get_fitz()
    doc = fitz.open(str(pdf_path))
    images: list[torch.Tensor] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        image_list = page.get_images(full=True)

        for img_info in image_list:
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue

            if base_image is None:
                continue

            width = base_image.get("width", 0)
            height = base_image.get("height", 0)
            if width < min_size or height < min_size:
                continue

            image_bytes = base_image["image"]
            try:
                # Decode image bytes to tensor using PyMuPDF's Pixmap
                pix = fitz.Pixmap(image_bytes)
                # Convert to RGB if needed
                if pix.n != 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                # Convert to torch tensor
                img_data = torch.frombuffer(
                    bytearray(pix.samples), dtype=torch.uint8
                ).reshape(pix.h, pix.w, 3)
                img_data = img_data.permute(2, 0, 1).float() / 255.0  # (C, H, W)

                # Resize to target_size
                img_data = torch.nn.functional.interpolate(
                    img_data.unsqueeze(0),
                    size=(target_size, target_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

                images.append(img_data)
            except Exception:
                continue

    doc.close()
    return images


def extract_sections(text_blocks: list[str]) -> dict[str, list[str]]:
    """Attempt to parse text blocks into named sections.

    Looks for common scientific paper headings (Abstract, Introduction,
    Methods, Results, Discussion, Conclusion, References).

    Returns dict mapping section name → list of text blocks.
    """
    section_pattern = re.compile(
        r"^(?:\d+\.?\s*)?"
        r"(abstract|introduction|background|related work|methods?|methodology|"
        r"materials?\s+and\s+methods?|experimental?\s+setup|results?|"
        r"discussion|conclusions?|summary|references?|acknowledgment|appendix)",
        re.IGNORECASE,
    )

    sections: dict[str, list[str]] = {"_preamble": []}
    current_section = "_preamble"

    for block in text_blocks:
        first_line = block.split("\n")[0].strip()
        match = section_pattern.match(first_line)
        if match:
            current_section = match.group(1).lower().strip()
            if current_section not in sections:
                sections[current_section] = []
            # Add remaining text after heading
            rest = block[match.end():].strip()
            if rest:
                sections[current_section].append(rest)
        else:
            sections[current_section].append(block)

    # Remove empty sections
    return {k: v for k, v in sections.items() if v}


def extract_figure_caption_pairs(
    text_blocks: list[str],
    images: list[torch.Tensor],
) -> list[tuple[torch.Tensor, str]]:
    """Attempt to pair extracted images with figure captions.

    Looks for "Figure N:" or "Fig. N:" patterns in text blocks and
    pairs them with images in order.  Falls back to sequential pairing.
    """
    caption_pattern = re.compile(
        r"^(?:Figure|Fig\.?)\s*(\d+)[.:]\s*(.*)", re.IGNORECASE | re.DOTALL
    )

    captions: list[str] = []
    for block in text_blocks:
        match = caption_pattern.match(block.strip())
        if match:
            caption_text = match.group(2).strip()
            if caption_text:
                captions.append(caption_text)

    # Pair images with captions (sequential matching)
    pairs: list[tuple[torch.Tensor, str]] = []
    for i, img in enumerate(images):
        if i < len(captions):
            pairs.append((img, captions[i]))

    return pairs


# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------
class ScientificPDFDataset(Dataset):
    """Dataset that loads scientific PDFs and creates multi-modal training pairs.

    Scans a directory (recursively) for .pdf files, extracts text and images,
    and generates training samples as (modality → tensor) dicts compatible
    with collate_multimodal.

    Sample types (randomly selected per __getitem__):
      1. text_reconstruction: predict masked/next section from context
      2. figure_captioning: image → text (figure + caption pair)
      3. caption_to_figure: text → image (caption → figure pair)
      4. section_completion: text prefix → text continuation

    Args:
        pdf_dir: directory containing .pdf files (searched recursively).
        config: OmniLatentConfig for shapes and tokenization params.
        max_text_len: maximum token length for text samples.
        image_size: target image size for extracted figures.
        cache_extractions: if True, cache parsed PDFs in memory.
    """

    def __init__(
        self,
        pdf_dir: str | Path,
        config: OmniLatentConfig,
        max_text_len: int | None = None,
        image_size: int | None = None,
        cache_extractions: bool = True,
    ) -> None:
        self.config = config
        self.max_text_len = max_text_len or config.text_max_len
        self.image_size = image_size or config.image_size
        self.cache = cache_extractions

        # Find all PDFs
        pdf_dir = Path(pdf_dir)
        self.pdf_paths = sorted(pdf_dir.rglob("*.pdf"))
        if not self.pdf_paths:
            raise FileNotFoundError(f"No PDF files found in {pdf_dir}")

        # Caches
        self._text_cache: dict[int, list[str]] = {}
        self._image_cache: dict[int, list[torch.Tensor]] = {}
        self._section_cache: dict[int, dict[str, list[str]]] = {}
        self._pair_cache: dict[int, list[tuple[torch.Tensor, str]]] = {}

        # Pre-scan to count available samples
        self._pdf_indices: list[int] = list(range(len(self.pdf_paths)))

    def __len__(self) -> int:
        # Each PDF generates multiple samples; estimate ~20 per PDF
        return len(self.pdf_paths) * 20

    def _load_pdf(self, pdf_idx: int) -> tuple[
        list[str], list[torch.Tensor], dict[str, list[str]],
        list[tuple[torch.Tensor, str]],
    ]:
        """Load and parse a PDF, using cache if available."""
        if pdf_idx in self._text_cache:
            return (
                self._text_cache[pdf_idx],
                self._image_cache.get(pdf_idx, []),
                self._section_cache.get(pdf_idx, {}),
                self._pair_cache.get(pdf_idx, []),
            )

        path = self.pdf_paths[pdf_idx]
        try:
            text_blocks = extract_pdf_text_blocks(path)
            images = extract_pdf_images(path, target_size=self.image_size)
            sections = extract_sections(text_blocks)
            pairs = extract_figure_caption_pairs(text_blocks, images)
        except Exception:
            text_blocks, images, sections, pairs = [], [], {}, []

        if self.cache:
            self._text_cache[pdf_idx] = text_blocks
            self._image_cache[pdf_idx] = images
            self._section_cache[pdf_idx] = sections
            self._pair_cache[pdf_idx] = pairs

        return text_blocks, images, sections, pairs

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pdf_idx = idx % len(self.pdf_paths)
        text_blocks, images, sections, pairs = self._load_pdf(pdf_idx)

        sample: dict[str, torch.Tensor] = {}

        # Decide sample type based on available data
        sample_types = ["text_reconstruction"]
        if images:
            sample_types.append("figure_captioning")
        if pairs:
            sample_types.append("caption_to_figure")
        if len(text_blocks) >= 2:
            sample_types.append("section_completion")

        sample_type = random.choice(sample_types)

        if sample_type == "text_reconstruction" and text_blocks:
            # Pick a random text block as both source and target
            block = random.choice(text_blocks)
            sample["text"] = _byte_tokenize(block, self.max_text_len)

        elif sample_type == "figure_captioning" and pairs:
            # Image → text pair
            img, caption = random.choice(pairs)
            sample["image"] = img
            sample["text"] = _byte_tokenize(caption, self.max_text_len)

        elif sample_type == "caption_to_figure" and pairs:
            # Text → image pair
            img, caption = random.choice(pairs)
            sample["text"] = _byte_tokenize(caption, self.max_text_len)
            sample["image"] = img

        elif sample_type == "section_completion" and len(text_blocks) >= 2:
            # Concatenate consecutive blocks: source = block[i], target = block[i+1]
            i = random.randint(0, len(text_blocks) - 2)
            context = text_blocks[i]
            continuation = text_blocks[i + 1]
            # Pack both into text: the model learns to predict continuation from context
            combined = context + " " + continuation
            sample["text"] = _byte_tokenize(combined, self.max_text_len)

        # Fallback: if we got nothing, generate a text sample from any block
        if not sample and text_blocks:
            block = random.choice(text_blocks)
            sample["text"] = _byte_tokenize(block, self.max_text_len)

        # If PDF was completely empty, return a minimal sample
        if not sample:
            sample["text"] = torch.zeros(self.max_text_len, dtype=torch.long)

        return sample


def collate_pdf(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Collate PDF dataset samples, handling mixed modality presence.

    Keeps only modalities present in ALL samples of the batch.
    Pads text to max length; stacks images directly.
    """
    result: dict[str, torch.Tensor] = {}

    # Find common modalities
    common = set(batch[0].keys())
    for sample in batch[1:]:
        common &= set(sample.keys())

    if "text" in common:
        max_len = max(s["text"].shape[0] for s in batch)
        padded = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, s in enumerate(batch):
            padded[i, : s["text"].shape[0]] = s["text"]
        result["text"] = padded

    if "image" in common:
        result["image"] = torch.stack([s["image"] for s in batch])

    return result
