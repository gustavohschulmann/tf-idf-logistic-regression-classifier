import pandas as pd
import pytest

from validation import (
    validate_scoring_frame,
    validate_text,
    validate_training_frame,
)


@pytest.mark.parametrize("value", [None, 123, [], {}])
def test_validate_text_rejects_non_strings(value):
    """Service boundaries should reject unsupported input types clearly."""
    with pytest.raises(TypeError, match="must be a string"):
        validate_text(value)


@pytest.mark.parametrize("value", ["", "   ", "\n\t"])
def test_validate_text_rejects_blank_strings(value):
    with pytest.raises(ValueError, match="must not be empty"):
        validate_text(value)


def test_training_validation_rejects_unknown_label():
    frame = pd.DataFrame(
        {
            "text": ["Please help me"],
            "label": ["unexpected-route"],
        }
    )

    with pytest.raises(ValueError, match="unknown labels"):
        validate_training_frame(frame)


def test_scoring_validation_requires_text_column():
    with pytest.raises(ValueError, match="text"):
        validate_scoring_frame(pd.DataFrame({"message": ["hello"]}))
