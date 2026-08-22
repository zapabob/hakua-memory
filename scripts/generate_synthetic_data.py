from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

try:
    from synthetic_dataset import generate_business_dataset
except ModuleNotFoundError:
    from scripts.synthetic_dataset import generate_business_dataset

LOGGER = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("synthetic_business_dataset.json"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    records = generate_business_dataset(args.seed, args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Generated %d records at %s", len(records), args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
