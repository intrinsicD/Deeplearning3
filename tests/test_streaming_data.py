from __future__ import annotations

import io
import json
import tarfile
import types
from pathlib import Path

import pytest
import torch

from omnilatent.data import DataManifest, MultiModalSample, SourceSpec, StreamingMultiModalDataset, stream_samples
from omnilatent.data.sources.hf import iter_hf_stream
from omnilatent.data.sources.local import iter_local_files
from omnilatent.data.sources.minari import iter_minari_transitions
from omnilatent.data.sources.webdataset import iter_webdataset_shards


def test_manifest_json_roundtrip(tmp_path: Path) -> None:
    manifest = DataManifest([
        SourceSpec(name="local", type="local", path=str(tmp_path), max_samples=2),
    ])
    path = tmp_path / "manifest.json"

    manifest.to_json(path)
    restored = DataManifest.from_json(path)

    assert restored.to_dict() == manifest.to_dict()


def test_local_folder_text_image_pdf_smoke(tmp_path: Path) -> None:
    from PIL import Image

    (tmp_path / "note.txt").write_text("hello local text", encoding="utf-8")
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4\nplain-ish pdf bytes")
    Image.new("RGB", (3, 2), color=(255, 0, 0)).save(tmp_path / "image.png")

    samples = list(iter_local_files(SourceSpec(name="local", type="local", path=str(tmp_path))))

    assert all(isinstance(sample, MultiModalSample) for sample in samples)
    assert any(sample.text == "hello local text" for sample in samples)
    assert any(sample.pdf_page is not None and "PDF" in sample.pdf_page for sample in samples)
    image_samples = [sample for sample in samples if sample.image is not None]
    assert len(image_samples) == 1
    assert image_samples[0].image.shape == (3, 2, 3)


def test_hf_stream_smoke_with_monkeypatched_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.SimpleNamespace(
        load_dataset=lambda *args, **kwargs: [
            {"text": "hello hf", "id": "row0", "embedding": [1.0, 2.0]},
            {"caption": "second", "url": "https://example.invalid"},
        ]
    )
    monkeypatch.setitem(__import__("sys").modules, "datasets", fake_module)

    samples = list(iter_hf_stream(SourceSpec(name="hf", type="hf_stream", dataset="fake/ds", split="train")))

    assert samples[0].text == "hello hf"
    assert samples[0].vector is not None
    assert torch.equal(samples[0].vector, torch.tensor([1.0, 2.0]))
    assert samples[0].metadata["id"] == "row0"
    assert samples[1].text == "second"


def test_webdataset_tar_shard_smoke(tmp_path: Path) -> None:
    shard = tmp_path / "sample.tar"
    with tarfile.open(shard, "w") as tar:
        _add_bytes(tar, "0001.txt", b"hello shard")
        _add_bytes(tar, "0001.json", json.dumps({"license": "unit"}).encode("utf-8"))
        buf = io.BytesIO()
        torch.save(torch.tensor([3.0, 4.0]), buf)
        _add_bytes(tar, "0001.pt", buf.getvalue())

    samples = list(iter_webdataset_shards(SourceSpec(name="wds", type="webdataset", path=str(shard))))

    assert len(samples) == 1
    sample = samples[0]
    assert sample.text == "hello shard"
    assert sample.vector is not None
    assert torch.equal(sample.vector, torch.tensor([3.0, 4.0]))
    assert sample.metadata["license"] == "unit"
    assert sample.metadata["key"] == "0001"


def test_minari_vector_smoke_with_dataset_object() -> None:
    episode = {
        "observations": [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]],
        "actions": [[0.5], [1.5]],
        "rewards": [1.0, 2.0],
    }
    dataset_obj = [episode]

    samples = list(iter_minari_transitions(SourceSpec(
        name="minari",
        type="minari",
        dataset="fake-control-v0",
        options={"dataset_obj": dataset_obj},
    )))

    assert len(samples) == 2
    assert samples[0].vector is not None
    assert samples[0].action is not None
    assert samples[0].next_observation is not None
    assert isinstance(samples[0].next_observation, torch.Tensor)
    assert torch.equal(samples[0].vector, torch.tensor([0.0, 1.0]))
    assert torch.equal(samples[0].action, torch.tensor([0.5]))
    assert samples[0].reward == 1.0
    assert torch.equal(samples[0].next_observation, torch.tensor([1.0, 2.0]))


def test_streaming_dataset_dispatches_manifest_sources(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    manifest = DataManifest([
        SourceSpec(name="local_text", type="local", path=str(tmp_path), max_samples=1),
    ])

    samples = list(StreamingMultiModalDataset(manifest))
    samples_via_helper = list(stream_samples(manifest))

    assert len(samples) == 1
    assert samples[0].text in {"a", "b"}
    assert samples[0].metadata["source"] == "local_text"
    assert len(samples_via_helper) == 1


def test_all_source_smokes_produce_multimodal_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "local.txt").write_text("local", encoding="utf-8")
    local_sample = next(iter_local_files(SourceSpec(name="local", type="local", path=str(tmp_path))))

    fake_module = types.SimpleNamespace(load_dataset=lambda *args, **kwargs: [{"text": "hf"}])
    monkeypatch.setitem(__import__("sys").modules, "datasets", fake_module)
    hf_sample = next(iter_hf_stream(SourceSpec(name="hf", type="hf", dataset="fake")))

    shard = tmp_path / "wds.tar"
    with tarfile.open(shard, "w") as tar:
        _add_bytes(tar, "x.txt", b"wds")
    wds_sample = next(iter_webdataset_shards(SourceSpec(name="wds", type="webdataset", path=str(shard))))

    minari_sample = next(iter_minari_transitions(SourceSpec(
        name="minari",
        type="minari",
        options={"dataset_obj": [{"observations": [[0], [1]], "actions": [[0]], "rewards": [0]}]},
    )))

    assert all(isinstance(s, MultiModalSample) for s in [local_sample, hf_sample, wds_sample, minari_sample])


def _add_bytes(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


