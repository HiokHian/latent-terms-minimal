"""Training entry point."""

from __future__ import annotations

import argparse

import lightning as L
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint

from latent_terms.data import build_collator, build_datamodule
from latent_terms.loggers import build_logger
from latent_terms.model.lit_module import LatentTermsLitModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latent Terms SAE training entry point.")
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config.")
    parser.add_argument("--fast-dev-run", action="store_true", default=False,
                        help="Lightning fast-dev-run: 1 train + 1 val batch.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint to resume from (weights + optim + scheduler + loop).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["seed"])
    L.seed_everything(seed, workers=True)

    module = LatentTermsLitModule(cfg)

    data_cfg = dict(cfg["data"])
    collator = build_collator(data_cfg.pop("collator"))
    dm = build_datamodule(data_cfg, collator, seed)

    logger = build_logger(cfg["logger"])

    reserved = {"logger", "default_root_dir", "fast_dev_run"}
    overlap = reserved & cfg["trainer"].keys()
    if overlap:
        raise KeyError(
            f"cfg['trainer'] contains reserved key(s) {sorted(overlap)}; "
            "these are set from other config blocks or CLI flags."
        )

    checkpoint_cb = (
        ModelCheckpoint(**cfg["checkpoint"])
        if "checkpoint" in cfg
        else ModelCheckpoint(
            dirpath="outputs/checkpoints",
            filename="epoch={epoch:02d}",
            save_top_k=-1,
            every_n_epochs=1,
        )
    )

    trainer = L.Trainer(
        default_root_dir="outputs/",
        logger=logger,
        callbacks=[checkpoint_cb],
        **cfg["trainer"],
        fast_dev_run=args.fast_dev_run,
    )
    trainer.fit(module, dm, ckpt_path=args.ckpt)


if __name__ == "__main__":
    main()
