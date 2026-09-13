"""Local CSV logger builder.

For smoke tests and offline runs where a wandb project is overkill: writes
``<save_dir>/<name>/version_N/metrics.csv``, which is directly greppable and
plottable without syncing anything.
"""

from __future__ import annotations

from lightning.pytorch.loggers import CSVLogger


def csv_logger(
    save_dir: str,
    run_name: str = "csv",
    version: str | int | None = None,
    flush_logs_every_n_steps: int = 100,
) -> CSVLogger:
    """Build a Lightning ``CSVLogger`` from config kwargs.

    The kwarg is ``run_name``, not ``name``, for the same reason ``wandb_logger``
    uses ``run_name``: ``build_from_registry`` pops ``cfg["name"]`` as the
    dispatch key, so a component builder can never expose its own ``name``.

    No ``**extra`` on purpose — a typo in the config should raise, not silently
    forward. Add a new keyword here explicitly when you need it.
    """
    return CSVLogger(
        save_dir=save_dir,
        name=run_name,
        version=version,
        flush_logs_every_n_steps=flush_logs_every_n_steps,
    )
