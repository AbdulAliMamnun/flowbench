"""Training loop with early stopping and validation-based checkpoint selection.

The loop is deliberately plain: AdamW, MSE in normalised units, one validation pass per
epoch, keep the parameters with the lowest validation MSE, stop after
``early_stopping_patience`` epochs without improvement. A model with no trainable
parameters (persistence) skips optimisation and is checkpointed as-is so every model
shares one checkpoint format.
"""

from __future__ import annotations

import copy
import time
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from flowbench.data import dataset as ds
from flowbench.data.normalize import NormalizationStats
from flowbench.data.split import Split
from flowbench.logging import get_logger
from flowbench.models.registry import build_model
from flowbench.training.checkpoint import save_checkpoint
from flowbench.training.seed import configure_determinism, resolve_device, seed_everything

if TYPE_CHECKING:
    from pathlib import Path

    from torch.utils.data import DataLoader

    from flowbench.config import FlowBenchConfig
    from flowbench.models.base import Predictor

log = get_logger(__name__)

Loader = "DataLoader[tuple[torch.Tensor, torch.Tensor]]"


@torch.no_grad()
def evaluate_mse(
    model: Predictor, loader: DataLoader[tuple[torch.Tensor, torch.Tensor]], device: torch.device
) -> float:
    """Mean squared error of ``model`` over ``loader`` in normalised units."""
    model.eval()
    total = 0.0
    count = 0
    for x, y in loader:
        xb = x.to(device)
        yb = y.to(device)
        total += float(nn.functional.mse_loss(model(xb), yb, reduction="sum"))
        count += yb.numel()
    return total / max(count, 1)


def train_one_epoch(
    model: Predictor,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_every: int,
    epoch: int,
) -> float:
    """Run one optimisation pass and return the mean training loss."""
    model.train()
    running = 0.0
    n_batches = 0
    for step, (x, y) in enumerate(loader, start=1):
        xb = x.to(device)
        yb = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(xb), yb)
        torch.autograd.backward(loss)
        optimizer.step()
        loss_value = float(loss.detach())
        running += loss_value
        n_batches += 1
        if step % log_every == 0:
            log.info("train step", epoch=epoch, step=step, loss=loss_value)
    return running / max(n_batches, 1)


def fit(
    model: Predictor,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    cfg: FlowBenchConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Train ``model`` in place and return the training record.

    The model's parameters are restored to the best validation epoch before returning.

    Args:
        model: Predictor on ``device``.
        train_loader: Shuffled training batches (normalised units).
        val_loader: Validation batches.
        cfg: Full configuration (training section is used).
        device: Compute device.

    Returns:
        Metrics dictionary with per-epoch history and the selection criterion.
    """
    tcfg = cfg.training
    history: list[dict[str, float | int]] = []
    val_before = evaluate_mse(model, val_loader, device)
    best_val = val_before
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_run = 0
    stopped_early = False

    if not model.is_trainable:
        log.info("model has no trainable parameters; skipping optimisation", val_mse=best_val)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay
        )
        since_improvement = 0
        for epoch in range(1, tcfg.epochs + 1):
            t0 = time.perf_counter()
            train_loss = train_one_epoch(
                model, train_loader, optimizer, device, tcfg.log_every_n_steps, epoch
            )
            val_loss = evaluate_mse(model, val_loader, device)
            elapsed = time.perf_counter() - t0
            epochs_run = epoch
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                since_improvement = 0
            else:
                since_improvement += 1
            history.append(
                {
                    "epoch": epoch,
                    "train_mse": train_loss,
                    "val_mse": val_loss,
                    "seconds": elapsed,
                }
            )
            log.info(
                "epoch done",
                epoch=epoch,
                train_mse=train_loss,
                val_mse=val_loss,
                best_val_mse=best_val,
                improved=improved,
                seconds=round(elapsed, 2),
            )
            if since_improvement >= tcfg.early_stopping_patience:
                stopped_early = True
                log.info("early stopping", epoch=epoch, patience=tcfg.early_stopping_patience)
                break

    model.load_state_dict(best_state)
    model.eval()
    return {
        "selection_metric": "val_mse_normalized",
        "best_val_mse": best_val,
        "best_epoch": best_epoch,
        "val_mse_before_training": val_before,
        "epochs_run": epochs_run,
        "epochs_budget": tcfg.epochs,
        "stopped_early": stopped_early,
        "history": history,
        "n_train": len(train_loader.dataset),  # type: ignore[arg-type]
        "n_val": len(val_loader.dataset),  # type: ignore[arg-type]
    }


def run_train(cfg: FlowBenchConfig) -> Path:
    """Train the configured model and write a checkpoint directory.

    Requires ``flowbench prepare`` to have produced the cache, split and statistics.

    Args:
        cfg: Full pipeline configuration.

    Returns:
        Path to the checkpoint directory (``<run_dir>/checkpoints/<model>``).
    """
    seed_everything(cfg.run.seed)
    device = resolve_device(cfg.run.device)
    determinism = configure_determinism(cfg.run.deterministic, device)
    log.info("training", model=cfg.model.name, device=device.type, seed=cfg.run.seed)

    split = Split.from_json(cfg.data.split_path)
    stats = NormalizationStats.from_json(cfg.data.normalization_path)
    pairs = ds.load_prepared(cfg.data, "train")
    train_ids, val_ids = split.train_ids, split.val_ids
    if cfg.data.max_train_samples is not None:
        train_ids, val_ids = _cap_by_simulation(split, cfg.data.max_train_samples)
    train_pairs = pairs.subset(train_ids)
    val_pairs = pairs.subset(val_ids)

    train_loader = ds.make_loader(
        train_pairs,
        stats,
        cfg.training.batch_size,
        shuffle=True,
        seed=cfg.run.seed,
        num_workers=cfg.data.num_workers,
    )
    val_loader = ds.make_loader(
        val_pairs,
        stats,
        cfg.evaluation.batch_size,
        shuffle=False,
        seed=cfg.run.seed,
        num_workers=cfg.data.num_workers,
    )

    model = build_model(cfg.model).to(device)
    log.info("model built", name=cfg.model.name, n_parameters=model.n_parameters)
    metrics = fit(model, train_loader, val_loader, cfg, device)
    metrics["train_ids_used"] = len(train_ids)
    metrics["val_ids_used"] = len(val_ids)
    return save_checkpoint(
        cfg.checkpoint_dir, model, cfg, stats, split, metrics, device, determinism
    )


def _cap_by_simulation(split: Split, max_samples: int) -> tuple[list[int], list[int]]:
    """Take the first ``max_samples`` train+val instances, preserving the val fraction.

    Instances are already grouped by simulation inside each partition, so truncating a
    prefix of each list keeps whole trajectories together wherever the cap falls between
    simulations; used only by smoke runs.
    """
    n_val = max(1, round(split.val_fraction * max_samples))
    n_train = max(1, max_samples - n_val)
    return split.train_ids[:n_train], split.val_ids[:n_val]
