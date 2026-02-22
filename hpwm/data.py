"""
Data loading for HPWM Phase -1.

Provides:
1. SyntheticMovingShapes: Procedural dataset with colored shapes moving
   on a background. Provides ground-truth segmentation masks for slot
   binding stability evaluation (Signal 2). Zero download cost.

2. SSv2Dataset: Something-Something-v2 loader for real object interaction
   videos. Requires dataset download.

3. ConcatenatedClipDataset: Kinetics-style long clips constructed by
   concatenating short clips, for temporal state retention testing (Signal 3).
"""

import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SyntheticMovingShapes(Dataset):
    """
    Procedural dataset: colored shapes moving on a background.

    Each clip contains N_objects shapes (circles, rectangles) that move
    with random velocities. Provides:
    - frames: [T, 3, H, W] video frames
    - masks: [T, N_objects, H, W] per-object binary masks

    For immediate testing of all 3 validation signals without data download.
    """

    def __init__(
        self,
        n_clips: int = 1000,
        n_frames: int = 120,
        resolution: int = 128,
        n_objects: int = 4,
        fps: int = 2,
        seed: int = 42,
    ):
        self.n_clips = n_clips
        self.n_frames = n_frames
        self.resolution = resolution
        self.n_objects = n_objects
        self.fps = fps
        self.seed = seed

        # Pre-generate clip seeds for reproducibility
        rng = random.Random(seed)
        self.clip_seeds = [rng.randint(0, 2**31) for _ in range(n_clips)]

    def __len__(self) -> int:
        return self.n_clips

    def _generate_clip(self, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a single video clip with moving shapes.

        Returns:
            frames: [T, 3, H, W] float32 in [0, 1]
            masks: [T, N_objects, H, W] binary float32
        """
        rng = np.random.RandomState(seed)
        H = W = self.resolution
        T = self.n_frames

        # Background color (dark)
        bg_color = rng.uniform(0.05, 0.2, size=(3,)).astype(np.float32)

        # Object properties
        shapes = []
        for _ in range(self.n_objects):
            shape_type = rng.choice(["circle", "rectangle"])
            color = rng.uniform(0.4, 1.0, size=(3,)).astype(np.float32)
            size = rng.uniform(8, 24)  # radius or half-width
            x = rng.uniform(size + 2, W - size - 2)
            y = rng.uniform(size + 2, H - size - 2)
            vx = rng.uniform(-1.5, 1.5)
            vy = rng.uniform(-1.5, 1.5)
            shapes.append({
                "type": shape_type,
                "color": color,
                "size": size,
                "x": x, "y": y,
                "vx": vx, "vy": vy,
            })

        frames = np.zeros((T, 3, H, W), dtype=np.float32)
        masks = np.zeros((T, self.n_objects, H, W), dtype=np.float32)

        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

        for t in range(T):
            # Background
            frame = np.ones((3, H, W), dtype=np.float32)
            for c in range(3):
                frame[c] = bg_color[c]

            for obj_idx, shape in enumerate(shapes):
                # Random velocity perturbation each frame (Brownian-ish motion)
                shape["vx"] += rng.normal(0, 0.3)
                shape["vy"] += rng.normal(0, 0.3)
                # Clamp velocity to prevent runaway
                shape["vx"] = np.clip(shape["vx"], -3.0, 3.0)
                shape["vy"] = np.clip(shape["vy"], -3.0, 3.0)

                # Update position with bouncing
                shape["x"] += shape["vx"]
                shape["y"] += shape["vy"]

                s = shape["size"]
                if shape["x"] <= s or shape["x"] >= W - s:
                    shape["vx"] *= -1
                    shape["x"] = np.clip(shape["x"], s, W - s)
                if shape["y"] <= s or shape["y"] >= H - s:
                    shape["vy"] *= -1
                    shape["y"] = np.clip(shape["y"], s, H - s)

                # Generate mask
                if shape["type"] == "circle":
                    dist = np.sqrt((xx - shape["x"])**2 + (yy - shape["y"])**2)
                    mask = (dist < shape["size"]).astype(np.float32)
                else:  # rectangle
                    mask = (
                        (np.abs(xx - shape["x"]) < shape["size"])
                        & (np.abs(yy - shape["y"]) < shape["size"])
                    ).astype(np.float32)

                masks[t, obj_idx] = mask

                # Draw shape on frame
                for c in range(3):
                    frame[c] = frame[c] * (1 - mask) + shape["color"][c] * mask

            frames[t] = frame

        return torch.from_numpy(frames), torch.from_numpy(masks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        frames, masks = self._generate_clip(self.clip_seeds[idx])

        # Normalize frames to ImageNet stats (for DINO compatibility)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        frames_norm = (frames - mean) / std

        return {
            "frames": frames_norm,     # [T, 3, H, W] normalized
            "frames_raw": frames,      # [T, 3, H, W] original [0,1]
            "masks": masks,            # [T, N_objects, H, W]
            "clip_id": idx,
        }


class ConcatenatedClipDataset(Dataset):
    """
    Long clips by concatenating short clips from SyntheticMovingShapes.

    Used for Signal 3 (Mamba state retention): tests whether the temporal
    state degrades gracefully over longer sequences.

    Creates clips of variable length (10s, 30s, 60s) at the configured fps.
    """

    def __init__(
        self,
        base_dataset: SyntheticMovingShapes,
        target_lengths_s: list[int] = [10, 30, 60],
        fps: int = 2,
    ):
        self.base = base_dataset
        self.target_lengths_s = target_lengths_s
        self.fps = fps

    def __len__(self) -> int:
        return len(self.base) * len(self.target_lengths_s)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        length_idx = idx % len(self.target_lengths_s)
        base_idx = idx // len(self.target_lengths_s)

        target_s = self.target_lengths_s[length_idx]
        target_frames = target_s * self.fps

        item = self.base[base_idx % len(self.base)]
        frames = item["frames"]
        T = frames.shape[0]

        if target_frames <= T:
            frames = frames[:target_frames]
        else:
            # Concatenate clips to reach target length
            all_frames = [frames]
            current_len = T
            extra_idx = base_idx + 1
            while current_len < target_frames:
                extra_item = self.base[extra_idx % len(self.base)]
                all_frames.append(extra_item["frames"])
                current_len += extra_item["frames"].shape[0]
                extra_idx += 1
            frames = torch.cat(all_frames, dim=0)[:target_frames]

        return {
            "frames": frames,
            "target_length_s": target_s,
            "clip_id": base_idx,
        }


class SSv2Dataset(Dataset):
    """
    Something-Something-v2 dataset loader.

    Expects the dataset to be downloaded and extracted at data_dir.
    Structure:
        data_dir/
            20bn-something-something-v2/
                1.webm
                2.webm
                ...
            labels/
                train.json
                validation.json
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        n_frames: int = 120,
        resolution: int = 128,
        fps: int = 2,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.n_frames = n_frames
        self.resolution = resolution
        self.fps = fps
        self.samples = []

        # Try to load labels
        label_file = self.data_dir / "labels" / f"{split}.json"
        if label_file.exists():
            import json
            with open(label_file) as f:
                self.samples = json.load(f)
        else:
            print(f"[WARN] SSv2 labels not found at {label_file}")
            print("       Use SyntheticMovingShapes for testing.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if not self.samples:
            raise RuntimeError(
                "SSv2 dataset not available. Use SyntheticMovingShapes instead."
            )

        sample = self.samples[idx]
        video_id = sample["id"]
        label = sample.get("template", "unknown")

        # Load video frames
        video_path = self.data_dir / "20bn-something-something-v2" / f"{video_id}.webm"
        frames = self._load_video(video_path)

        # Normalize for DINO
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        frames_norm = (frames - mean) / std

        return {
            "frames": frames_norm,
            "frames_raw": frames,
            "label": label,
            "clip_id": idx,
        }

    def _load_video(self, path: Path) -> torch.Tensor:
        """Load and preprocess video frames."""
        try:
            import torchvision.io as tvio
            video, _, _ = tvio.read_video(str(path), pts_unit="sec")
            # video: [T, H, W, 3] uint8
            video = video.float() / 255.0
            video = video.permute(0, 3, 1, 2)  # [T, 3, H, W]

            # Resize
            video = torch.nn.functional.interpolate(
                video, size=(self.resolution, self.resolution),
                mode="bilinear", align_corners=False,
            )

            # Subsample to target fps/frame count
            if video.shape[0] > self.n_frames:
                indices = torch.linspace(0, video.shape[0] - 1, self.n_frames).long()
                video = video[indices]
            elif video.shape[0] < self.n_frames:
                # Repeat last frame
                pad = self.n_frames - video.shape[0]
                video = torch.cat(
                    [video, video[-1:].expand(pad, -1, -1, -1)], dim=0,
                )

            return video

        except Exception as e:
            print(f"[WARN] Failed to load video {path}: {e}")
            return torch.zeros(self.n_frames, 3, self.resolution, self.resolution)


def create_dataloaders(
    config,
    use_synthetic: bool = True,
    data_dir: str | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders.

    Args:
        config: HPWMConfig
        use_synthetic: if True, use SyntheticMovingShapes
        data_dir: path to SSv2 dataset (if use_synthetic=False)

    Returns:
        train_loader, val_loader
    """
    if use_synthetic:
        train_dataset = SyntheticMovingShapes(
            n_clips=800,
            n_frames=config.n_frames,
            resolution=config.resolution,
            n_objects=4,
            fps=config.fps,
            seed=42,
        )
        val_dataset = SyntheticMovingShapes(
            n_clips=200,
            n_frames=config.n_frames,
            resolution=config.resolution,
            n_objects=4,
            fps=config.fps,
            seed=123,
        )
    else:
        train_dataset = SSv2Dataset(
            data_dir=data_dir or config.data_dir,
            split="train",
            n_frames=config.n_frames,
            resolution=config.resolution,
            fps=config.fps,
        )
        val_dataset = SSv2Dataset(
            data_dir=data_dir or config.data_dir,
            split="validation",
            n_frames=config.n_frames,
            resolution=config.resolution,
            fps=config.fps,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader
