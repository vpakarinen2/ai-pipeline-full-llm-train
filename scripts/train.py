"""Training entrypoint."""

from __future__ import annotations

import argparse
import sys
import os

from pathlib import Path
from pprint import pprint

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")


def _add_src_to_path() -> None:
    """Ensure `src/` is on sys.path."""
    root_dir = Path(__file__).resolve().parents[1]
    src_dir = root_dir / "src"
    if src_dir.is_dir():
        src_str = str(src_dir)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)

_add_src_to_path()

from ai_pipeline.config.loader import load_config  # noqa: E402
from ai_pipeline.training.trainer import Trainer  # noqa: E402
from ai_pipeline.utils.logging import get_logger  # noqa: E402
from ai_pipeline.utils.seed import set_seed  # noqa: E402


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Full LLM fine-tune training")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to YAML/JSON config file",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Override run.seed (optional)",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        help="Override run.device (e.g. 'cuda', 'cpu', 'auto')",
    )
    parser.add_argument(
        "-r",
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from (optional)",
    )
    return parser.parse_args()


def main() -> None:
    """Main training entrypoint."""
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path)

    if args.seed is not None:
        cfg.run.seed = args.seed
    if args.device is not None:
        cfg.run.device = args.device

    set_seed(cfg.run.seed, cfg.run.deterministic)

    logger.info("Loaded config from %s", config_path)
    pprint(cfg)

    resume_path = Path(args.resume_from) if args.resume_from is not None else None

    trainer = Trainer(cfg, resume_from=resume_path)
    trainer.train()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception in training script")
        sys.exit(1)
