"""Weights & Biases logger builder."""

from __future__ import annotations

from lightning.pytorch.loggers import WandbLogger


def wandb_logger(
    project: str,
    entity: str | None = None,
    run_name: str | None = None,
    resume_id: str | None = None,
    tags: list[str] | None = None,
    save_dir: str = "outputs/wandb",
    offline: bool = False,
    log_model: bool = False,
) -> WandbLogger:
    """Build a Lightning ``WandbLogger`` from config kwargs.

    No ``**extra`` on purpose: a typo in the config should raise, not silently
    forward. Add a new keyword here explicitly when you need it.
    """
    return WandbLogger(
        project=project,
        entity=entity,
        name=run_name,
        id=resume_id,
        tags=tags,
        save_dir=save_dir,
        offline=offline,
        log_model=log_model,
    )
