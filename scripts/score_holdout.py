import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from constants import (
    CONFIDENCE_COLUMN,
    PREDICTION_COLUMN,
    TEXT_COLUMN,
)
from model import load_router
from validation import validate_scoring_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a CSV containing a text column."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/ticket_router.joblib"),
    )
    return parser.parse_args()


def score_file(input_path: Path, output_path: Path, model_path: Path) -> None:
    """Score rows and preserve all original columns."""
    frame = validate_scoring_frame(pd.read_csv(input_path))
    router = load_router(model_path)

    predictions = router.predict_many(frame[TEXT_COLUMN].tolist())

    output = frame.copy()
    output[PREDICTION_COLUMN] = [item.label for item in predictions]
    output[CONFIDENCE_COLUMN] = [
        round(item.confidence, 6) for item in predictions
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    score_file(args.input, args.output, args.model)
    print(f"wrote predictions to {args.output}")


if __name__ == "__main__":
    main()
