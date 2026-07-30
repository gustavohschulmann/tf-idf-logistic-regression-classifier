import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from constants import LABEL_COLUMN
from evaluation import evaluate_pipeline
from validation import validate_training_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate candidate pipelines.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/train.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = validate_training_frame(pd.read_csv(args.input))

    # Stratification preserves the approximate label proportions in both sets.
    train_frame, validation_frame = train_test_split(
        frame,
        test_size=0.20,
        stratify=frame[LABEL_COLUMN],
        random_state=42,
    )

    candidates = {
        "balanced": "balanced",
        "unweighted": None,
    }

    results: dict[str, dict] = {}
    for name, class_weight in candidates.items():
        summary = evaluate_pipeline(
            train_frame=train_frame,
            validation_frame=validation_frame,
            class_weight=class_weight,
        )
        results[name] = summary.as_dict()

    # Selection rule:
    # 1) Highest validation macro F1.
    # 2) If tied, highest fraud recall.
    # 3) If still tied, prefer balanced because its objective explicitly
    #    represents the known class imbalance.
    selected = "balanced"
    best = results["balanced"]
    for name in candidates:
        current = results[name]
        if current["macro_f1"] > best["macro_f1"]:
            selected, best = name, current
        elif current["macro_f1"] == best["macro_f1"]:
            if current["fraud_recall"] > best["fraud_recall"]:
                selected, best = name, current
            elif (
                current["fraud_recall"] == best["fraud_recall"]
                and name == "balanced"
            ):
                selected, best = name, current


    payload = {
        "split": {
            "random_state": 42,
            "validation_fraction": 0.20,
            "stratified": True,
            "train_rows": len(train_frame),
            "validation_rows": len(validation_frame),
        },
        "primary_metric": "macro_f1",
        "safety_metric": "fraud_recall",
        "selected_candidate": selected,
        "candidates": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
