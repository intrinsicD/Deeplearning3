"""Tests for the scientific PDF dataset pipeline.

Since we may not have PyMuPDF installed in the test environment or
actual PDF files, these tests focus on:
  1. The byte tokenizer
  2. Section extraction from text blocks
  3. Figure-caption pairing logic
  4. Dataset construction and collation
"""

from __future__ import annotations

import pytest
import torch

from omnilatent.config import OmniLatentConfig
from omnilatent.training.pdf_dataset import (
    _byte_tokenize,
    collate_pdf,
    extract_figure_caption_pairs,
    extract_sections,
)


class TestByteTokenizer:
    def test_basic_encoding(self) -> None:
        tokens = _byte_tokenize("Hello", max_len=10)
        assert tokens.shape == (10,)
        assert tokens.dtype == torch.long
        # First 5 should be non-zero, rest padding
        assert (tokens[:5] > 0).all()
        assert (tokens[5:] == 0).all()

    def test_truncation(self) -> None:
        tokens = _byte_tokenize("Hello World", max_len=5)
        assert tokens.shape == (5,)
        assert (tokens > 0).all()  # all filled, no padding

    def test_empty_string(self) -> None:
        tokens = _byte_tokenize("", max_len=10)
        assert tokens.shape == (10,)
        assert (tokens == 0).all()

    def test_unicode(self) -> None:
        tokens = _byte_tokenize("αβγ", max_len=20)
        assert tokens.shape == (20,)
        # Greek letters are multi-byte in UTF-8
        assert (tokens[:6] > 0).all()  # 3 chars × 2 bytes each

    def test_values_in_range(self) -> None:
        tokens = _byte_tokenize("Test string with various chars!", max_len=50)
        assert tokens.min() >= 0
        assert tokens.max() <= 255


class TestSectionExtraction:
    def test_basic_sections(self) -> None:
        blocks = [
            "Abstract This paper presents a novel approach.",
            "1. Introduction Deep learning has shown remarkable progress.",
            "Some more introduction text here.",
            "2. Methods We use a transformer architecture.",
            "Results Our model achieves state-of-the-art.",
        ]
        sections = extract_sections(blocks)
        assert "abstract" in sections
        assert "introduction" in sections
        assert "methods" in sections or "method" in sections

    def test_no_sections_goes_to_preamble(self) -> None:
        blocks = ["Just some text.", "More text without headings."]
        sections = extract_sections(blocks)
        assert "_preamble" in sections
        assert len(sections["_preamble"]) == 2

    def test_empty_blocks(self) -> None:
        sections = extract_sections([])
        # Empty sections should be removed
        assert len(sections) == 0 or all(len(v) > 0 for v in sections.values())

    def test_numbered_sections(self) -> None:
        blocks = [
            "3. Discussion The results indicate...",
            "4. Conclusion We have demonstrated...",
        ]
        sections = extract_sections(blocks)
        assert "discussion" in sections
        # "conclusion" or "conclusions" depending on parsing
        assert any("conclusi" in k for k in sections)


class TestFigureCaptionPairing:
    def test_basic_pairing(self) -> None:
        blocks = [
            "Some text about the experiment.",
            "Figure 1: Architecture of our proposed model.",
            "More text here.",
            "Fig. 2: Training loss over epochs.",
        ]
        # Create dummy image tensors
        images = [torch.randn(3, 32, 32), torch.randn(3, 32, 32)]
        pairs = extract_figure_caption_pairs(blocks, images)
        assert len(pairs) == 2
        assert pairs[0][1] == "Architecture of our proposed model."
        assert pairs[1][1] == "Training loss over epochs."

    def test_more_images_than_captions(self) -> None:
        blocks = ["Figure 1: Only one caption."]
        images = [torch.randn(3, 32, 32)] * 3
        pairs = extract_figure_caption_pairs(blocks, images)
        assert len(pairs) == 1  # only 1 caption matched

    def test_no_captions(self) -> None:
        blocks = ["No figure captions here."]
        images = [torch.randn(3, 32, 32)]
        pairs = extract_figure_caption_pairs(blocks, images)
        assert len(pairs) == 0

    def test_no_images(self) -> None:
        blocks = ["Figure 1: A caption without an image."]
        pairs = extract_figure_caption_pairs(blocks, [])
        assert len(pairs) == 0


class TestCollation:
    def test_text_only_batch(self) -> None:
        batch = [
            {"text": torch.tensor([1, 2, 3])},
            {"text": torch.tensor([4, 5])},
        ]
        result = collate_pdf(batch)
        assert "text" in result
        assert result["text"].shape == (2, 3)  # padded to max len
        assert result["text"][1, 2] == 0  # padding

    def test_mixed_modalities(self) -> None:
        batch = [
            {"text": torch.tensor([1, 2]), "image": torch.randn(3, 32, 32)},
            {"text": torch.tensor([3, 4, 5]), "image": torch.randn(3, 32, 32)},
        ]
        result = collate_pdf(batch)
        assert "text" in result
        assert "image" in result
        assert result["text"].shape == (2, 3)
        assert result["image"].shape == (2, 3, 32, 32)

    def test_uncommon_modalities_dropped(self) -> None:
        """If a modality isn't in ALL samples, it's dropped."""
        batch = [
            {"text": torch.tensor([1, 2]), "image": torch.randn(3, 32, 32)},
            {"text": torch.tensor([3, 4])},
        ]
        result = collate_pdf(batch)
        assert "text" in result
        assert "image" not in result  # not common to all
