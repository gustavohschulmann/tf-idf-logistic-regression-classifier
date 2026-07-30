#!/usr/bin/env python3
"""Simple one-message CLI demonstrating the required predict interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import load_router


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route one support message.")
    parser.add_argument("text", help="The support message to classify.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/ticket_router.joblib"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction = load_router(args.model).predict(args.text)
    print(json.dumps(prediction.as_dict(), indent=2))


if __name__ == "__main__":
    main()
