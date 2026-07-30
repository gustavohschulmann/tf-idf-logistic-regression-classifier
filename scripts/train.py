import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import TicketRouter
from validation import validate_training_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the support-ticket router on all labeled rows."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/train.csv"),
        help="Labeled CSV containing text and label columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ticket_router.joblib"),
        help="Destination for the fitted model pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validation happens before fitting so schema/data problems fail early.
    frame = validate_training_frame(pd.read_csv(args.input))

    router = TicketRouter()
    router.fit(frame)
    router.save(args.output)

    print(f"trained on {len(frame)} rows")
    print(f"saved model to {args.output}")


if __name__ == "__main__":
    main()
