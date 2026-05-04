#!/usr/bin/env python
"""Local control backend for index.html.

Run with:
    python server.py

Then open:
    http://localhost:8000

The server intentionally supports one active training job at a time. It launches
repo training scripts as subprocesses and exposes simple status endpoints for the
static dashboard.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from pydantic import BaseModel, Field
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "FastAPI backend dependencies are missing. Install them with:\n"
        "  pip install fastapi uvicorn pydantic\n"
        "or:\n"
        "  pip install -e '.[server,datasets]'"
    ) from exc

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parents[1]
STATUS_DIR = ROOT / "runs" / "control_center_backend"
STATUS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Deeplearning3 Control Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,  # type: ignore[arg-type]
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class MinariDownloadRequest(BaseModel):
    dataset_id: str
    force_download: bool = False


class TrainStartRequest(BaseModel):
    model_family: str = "mmwm"
    dataset: Dict[str, Any] = Field(default_factory=dict)
    common: Dict[str, Any] = Field(default_factory=dict)
    mmwm: Dict[str, Any] = Field(default_factory=dict)
    hpwm: Dict[str, Any] = Field(default_factory=dict)
    omnilatent: Dict[str, Any] = Field(default_factory=dict)
    lgq: Dict[str, Any] = Field(default_factory=dict)
    gaussian: Dict[str, Any] = Field(default_factory=dict)


class UseCaseRequest(BaseModel):
    command: str = ""
    use_case: str = ""
    checkpoint: str = ""
    source: str = "image"
    target: str = "text"
    input_file: str = ""
    output_path: str = ""
    notes: str = ""
    config: Dict[str, Any] = Field(default_factory=dict)


class JobState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.proc: Optional[subprocess.Popen[str]] = None
        self.status_file: Optional[Path] = None
        self.status: Dict[str, Any] = {
            "state": "idle",
            "step": 0,
            "total": 0,
            "loss": None,
            "device": "—",
            "metrics": {},
            "log": ["Backend ready."],
        }
        self.stdout_thread: Optional[threading.Thread] = None

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            status = dict(self.status)
            if self.status_file and self.status_file.exists():
                try:
                    file_status = json.loads(self.status_file.read_text(encoding="utf-8"))
                    status.update(file_status)
                except Exception as exc:  # pragma: no cover - defensive
                    status.setdefault("log", []).append(f"Could not read status file: {exc}")
            if self.proc is not None and self.proc.poll() is not None and status.get("state") == "running":
                status["state"] = "finished" if self.proc.returncode == 0 else "error"
                status.setdefault("log", []).append(f"Process exited with code {self.proc.returncode}")
                self.status = status
            return status

    def append_log(self, line: str) -> None:
        with self.lock:
            log = self.status.setdefault("log", [])
            log.append(line.rstrip())
            self.status["log"] = log[-200:]

    def start(self, cmd: List[str], status_file: Optional[Path] = None, total_steps: int = 0, kind: str = "job") -> Dict[str, Any]:
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                raise HTTPException(status_code=409, detail="A job is already running")
            self.status_file = status_file
            if status_file is not None and status_file.exists():
                status_file.unlink()
            self.status = {
                "state": "running",
                "kind": kind,
                "step": 0,
                "total": total_steps,
                "loss": None,
                "device": "auto",
                "metrics": {},
                "command": " ".join(cmd),
                "log": ["Launching: " + " ".join(cmd)],
            }
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )
            self.stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
            self.stdout_thread.start()
            return self.snapshot()

    def _read_stdout(self) -> None:
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            self.append_log(line)
        code = proc.wait()
        with self.lock:
            if self.status.get("state") == "running":
                self.status["state"] = "finished" if code == 0 else "error"
            self.status.setdefault("log", []).append(f"Process exited with code {code}")

    def stop(self) -> Dict[str, Any]:
        with self.lock:
            proc = self.proc
            if proc is None or proc.poll() is not None:
                self.status["state"] = "idle"
                self.status.setdefault("log", []).append("No active process to stop.")
                return self.status
            self.status["state"] = "stopping"
            self.status.setdefault("log", []).append("Stopping process...")
        try:
            if hasattr(os, "killpg") and proc.pid:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
        except Exception:
            proc.terminate()
        return self.snapshot()


JOB = JobState()


def _require_minari():
    try:
        import minari  # type: ignore
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="Minari is not installed. Run: pip install minari or pip install -e '.[datasets]'",
        ) from exc
    return minari


@app.get("/")
def index() -> FileResponse:
    return FileResponse(APP_DIR / "index.html")


@app.get("/api/status")
def status() -> Dict[str, Any]:
    return {"ok": True, "root": str(ROOT), "time": time.time()}


@app.get("/api/jobs")
def jobs() -> Dict[str, Any]:
    return JOB.snapshot()


@app.get("/api/minari/datasets")
def minari_datasets() -> Dict[str, Any]:
    minari = _require_minari()
    try:
        datasets = minari.list_remote_datasets()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    names = list(datasets.keys()) if hasattr(datasets, "keys") else list(datasets)
    names = sorted(str(name) for name in names)
    curated_keywords = ["pendulum", "pointmaze", "hopper", "walker", "halfcheetah", "antmaze", "kitchen"]
    curated = [name for name in names if any(k in name.lower() for k in curated_keywords)]
    return {"count": len(names), "datasets": names, "curated": curated[:200]}


@app.post("/api/minari/download")
def minari_download(req: MinariDownloadRequest) -> Dict[str, Any]:
    minari = _require_minari()
    try:
        try:
            minari.download_dataset(req.dataset_id, force_download=req.force_download)
        except TypeError:
            minari.download_dataset(req.dataset_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"ok": True, "dataset_id": req.dataset_id}


def _start_mmwm_minari(req: TrainStartRequest) -> Dict[str, Any]:
    dataset = req.dataset or {}
    common = req.common or {}
    mmwm = req.mmwm or {}
    dataset_id = dataset.get("minari_dataset_id") or dataset.get("path") or dataset.get("dataset_id")
    if not dataset_id:
        raise HTTPException(status_code=400, detail="Select or enter a Minari dataset id first")
    steps = int(common.get("steps") or 1000)
    batch_size = int(common.get("batch_size") or 128)
    lr = str(common.get("learning_rate") or "3e-4")
    run_dir = str(common.get("run_dir") or "runs/mmwm_minari")
    latent_dim = int(mmwm.get("latent_dim") or 64)
    hidden_dim = int(mmwm.get("hidden_dim") or 128)
    transition = str(mmwm.get("transition") or "mlp")
    max_transitions = int(dataset.get("max_transitions") or 50_000)
    download = bool(dataset.get("download", True))

    status_file = STATUS_DIR / "mmwm_minari_status.json"
    cmd = [
        sys.executable,
        "train_mmwm_minari.py",
        "--dataset-id", str(dataset_id),
        "--steps", str(steps),
        "--batch-size", str(batch_size),
        "--lr", lr,
        "--latent-dim", str(latent_dim),
        "--hidden-dim", str(hidden_dim),
        "--transition", transition,
        "--max-transitions", str(max_transitions),
        "--run-dir", run_dir,
        "--status-file", str(status_file),
    ]
    if download:
        cmd.append("--download")
    return JOB.start(cmd, status_file=status_file, total_steps=steps, kind="train:mmwm_minari")


@app.post("/api/train/start")
def train_start(req: TrainStartRequest) -> Dict[str, Any]:
    dataset_type = (req.dataset or {}).get("type")
    if req.model_family == "mmwm" and dataset_type == "minari":
        return _start_mmwm_minari(req)
    raise HTTPException(
        status_code=400,
        detail="This backend currently starts real jobs for model_family='mmwm' with dataset.type='minari'. Use generated commands for other workflows.",
    )


@app.post("/api/train/stop")
def train_stop() -> Dict[str, Any]:
    return JOB.stop()


def _safe_text(value: str, fallback: str = "") -> str:
    return str(value or fallback).strip()


def _write_usecase_config(req: UseCaseRequest) -> Path:
    target = STATUS_DIR / "usecase_config.json"
    target.write_text(json.dumps(req.config, indent=2), encoding="utf-8")
    return target


def _default_output(req: UseCaseRequest, suffix: str) -> str:
    raw = _safe_text(req.output_path)
    if raw:
        return raw
    return str(STATUS_DIR / f"{req.use_case}{suffix}")


def _require_checkpoint(req: UseCaseRequest) -> str:
    checkpoint = _safe_text(req.checkpoint)
    if not checkpoint:
        raise HTTPException(status_code=400, detail="Checkpoint path is required for this use case")
    path = Path(checkpoint)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Checkpoint does not exist: {checkpoint}")
    return checkpoint


def _require_input_file(path_text: str) -> str:
    input_file = _safe_text(path_text)
    if not input_file:
        raise HTTPException(status_code=400, detail="Input file is required for this use case")
    path = Path(input_file)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Input file does not exist: {input_file}")
    return input_file


def _build_usecase_command(req: UseCaseRequest) -> List[str]:
    use_case = _safe_text(req.use_case)
    checkpoint = _safe_text(req.checkpoint)
    output_base = _default_output(req, "_output")

    if use_case in {"mmwm_predict", "mmwm_rollout", "mmwm_tool"}:
        ckpt = _require_checkpoint(req)
        cfg_path = _write_usecase_config(req)
        mode = {"mmwm_predict": "predict", "mmwm_rollout": "rollout", "mmwm_tool": "tool"}[use_case]
        return [
            sys.executable,
            "mmwm_usecase.py",
            "--mode", mode,
            "--checkpoint", ckpt,
            "--config-json", str(cfg_path),
            "--steps", "5",
            "--output", str(STATUS_DIR / f"{use_case}.json"),
        ]

    if use_case == "omni_reconstruct":
        return [sys.executable, "evaluate.py", "--checkpoint", _require_checkpoint(req), "--mode", "reconstruct"]
    if use_case == "omni_translate":
        return [sys.executable, "evaluate.py", "--checkpoint", _require_checkpoint(req), "--mode", "translate"]
    if use_case == "omni_file":
        input_file = _require_input_file(req.input_file)
        return [
            sys.executable,
            "evaluate.py",
            "--checkpoint", _require_checkpoint(req),
            "--input-file", input_file,
            "--source", _safe_text(req.source, "image"),
            "--target", _safe_text(req.target, "text"),
            "--mode", "file",
            "--output-dir", output_base,
        ]

    if use_case == "hpwm_eval":
        return [sys.executable, "-m", "hpwm.train", "--eval-only", "--resume", _require_checkpoint(req)]
    if use_case == "lgq_eval":
        return [
            sys.executable,
            "-m", "lgq.evaluate",
            "--checkpoint", _require_checkpoint(req),
            "--output", _default_output(req, "_lgq_eval.json"),
        ]
    if use_case == "gauss_recon":
        return [sys.executable, "-m", "gaussian_encoder.train", "--epochs", "5", "--latent-dim", "32"]
    if use_case == "benchmark":
        cmd = [sys.executable, "benchmark.py", "--output", _default_output(req, "_benchmark.json")]
        if checkpoint:
            cmd.extend(["--checkpoint", checkpoint])
        return cmd

    raise HTTPException(status_code=400, detail=f"Unknown use case: {use_case!r}")


@app.post("/api/usecase/run")
def usecase_run(req: UseCaseRequest) -> Dict[str, Any]:
    cmd = _build_usecase_command(req)
    return JOB.start(cmd, status_file=None, total_steps=1, kind=f"usecase:{req.use_case}")


if __name__ == "__main__":
    try:
        import uvicorn
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("Install uvicorn with: pip install uvicorn") from exc
    uvicorn.run("apps.control_center.server:app", host="127.0.0.1", port=8000, reload=False)

