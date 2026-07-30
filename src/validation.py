import pandas as pd

from constants import EXPECTED_LABELS, LABEL_COLUMN, TEXT_COLUMN


def validate_text(text: object) -> str:
    """Validate and normalize a single prediction input."""

    if not isinstance(text, str):
        raise TypeError("It must be a string")

    normalized = text.strip()
    if not normalized:
        raise ValueError("Text must not be empty")

    return normalized


def validate_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate training data and return a safe copy."""

    required = {TEXT_COLUMN, LABEL_COLUMN}

    # catch column difference, good practice
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"training data is missing required columns")

    checked = df.copy()

    has_null_text = checked[TEXT_COLUMN].isna().any()
    has_null_label = checked[LABEL_COLUMN].isna().any()
    if has_null_text or has_null_label:
        raise ValueError("training data contains null text or label values")

    checked[TEXT_COLUMN] = checked[TEXT_COLUMN].map(validate_text)
    checked[LABEL_COLUMN] = checked[LABEL_COLUMN].astype(str).str.strip()

    # cleain unknown labels
    unknown_labels = sorted(set(checked[LABEL_COLUMN]) - set(EXPECTED_LABELS))
    if unknown_labels:
        raise ValueError("training data contains unknown labels")

    # verify duplicate texts
    seen_texts = set()
    duplicate_count = 0
    for text in checked[TEXT_COLUMN]:
        if text in seen_texts:
            duplicate_count += 1
        else:
            seen_texts.add(text)

    if duplicate_count:
        raise ValueError("training data contains duplicate text rows, deduplicate before splitting to reduce leakage risk")

    return checked


def validate_scoring_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate unlabeled scoring data and return a safe copy."""

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"scoring data is missing required column: {TEXT_COLUMN}")

    checked = df.copy()

    if checked[TEXT_COLUMN].isna().any():
        raise ValueError("scoring data contains null text values")

    checked[TEXT_COLUMN] = checked[TEXT_COLUMN].map(validate_text)
    return checked
