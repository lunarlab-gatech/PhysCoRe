"""
Checkpoints, logs and plots written by an MfM training run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from omegaconf import OmegaConf


STATE_CHECKPOINT_FORMAT = "physcore.mfm.state.v2"


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.ndim == 0:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


@dataclass(frozen=True)
class MfMRunArtifacts:
    output_dir: Path
    checkpoints_dir: Path
    states_dir: Path
    metrics_dir: Path
    plots_dir: Path
    logs_dir: Path

    @classmethod
    def create(cls, output_dir: str | Path) -> "MfMRunArtifacts":
        root = Path(output_dir)
        artifacts = cls(
            output_dir=root,
            checkpoints_dir=root / "checkpoints",
            states_dir=root / "states",
            metrics_dir=root / "metrics",
            plots_dir=root / "plots",
            logs_dir=root / "logs",
        )
        for path in (
            artifacts.output_dir,
            artifacts.checkpoints_dir,
            artifacts.states_dir,
            artifacts.metrics_dir,
            artifacts.plots_dir,
            artifacts.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return artifacts

    @property
    def train_log_path(self) -> Path:
        return self.logs_dir / "train.log"

    @property
    def history_jsonl_path(self) -> Path:
        return self.metrics_dir / "history.jsonl"

    @property
    def details_jsonl_path(self) -> Path:
        return self.metrics_dir / "details.jsonl"

    @property
    def history_json_path(self) -> Path:
        return self.metrics_dir / "history.json"

    @property
    def summary_json_path(self) -> Path:
        return self.metrics_dir / "summary.json"

    @property
    def latest_state_path(self) -> Path:
        return self.states_dir / "latest_state.pt"

    @property
    def latest_model_path(self) -> Path:
        return self.checkpoints_dir / "latest.pt"

    def best_model_path(self, metric_name: str) -> Path:
        return self.checkpoints_dir / f"best_{metric_name}.pt"

    def epoch_model_path(self, epoch: int) -> Path:
        return self.checkpoints_dir / f"epoch_{int(epoch):04d}.pt"

    def write_config(self, cfg) -> None:
        (self.output_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))

    def append_epoch_metrics(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        clean = _jsonable(row)
        with self.history_jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(clean, sort_keys=True) + "\n")
        history = self.load_history()
        self.history_json_path.write_text(json.dumps(history, indent=2))
        return history

    def append_detail_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with self.details_jsonl_path.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")

    def load_history(self) -> list[dict[str, Any]]:
        if not self.history_jsonl_path.exists():
            return []
        rows = []
        for line in self.history_jsonl_path.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

    def write_summary(self, summary: dict[str, Any]) -> None:
        self.summary_json_path.write_text(json.dumps(_jsonable(summary), indent=2))


def save_model_checkpoint(path: str | Path, model: torch.nn.Module, epoch: int) -> None:
    # Weights only; `epoch` is used by the caller for the filename.
    torch.save({"model": model.state_dict()}, Path(path))


def save_training_state(
    path: str | Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_metric: float,
    best_epoch: int,
    best_checkpoint: Optional[str],
) -> None:
    torch.save(
        {
            "format": STATE_CHECKPOINT_FORMAT,
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_metric": float(best_metric),
            "best_epoch": int(best_epoch),
            "best_checkpoint": best_checkpoint,
        },
        Path(path),
    )


def load_checkpoint(path: str | Path, map_location=None) -> dict[str, Any]:
    obj = torch.load(str(path), map_location=map_location, weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint format at {path}")
    return obj


def model_state_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "model" in checkpoint:
        return checkpoint["model"]
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    raise KeyError("checkpoint is missing model weights")


def load_model_weights(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    strict: bool,
    map_location=None,
):
    ckpt = load_checkpoint(checkpoint_path, map_location=map_location)
    state = model_state_from_checkpoint(ckpt)
    result = model.load_state_dict(state, strict=strict)
    return result, ckpt

