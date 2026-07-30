from pathlib import Path

import pandas as pd

from constants import EXPECTED_LABELS
from model import TicketRouter, load_router


def small_training_frame() -> pd.DataFrame:
    """Small but separable fixture used to test behavior, not model quality."""

    # Samples generated with AI
    return pd.DataFrame(
        {
            "text": [
                "I cannot log in to my account",
                "My password reset link does not work",
                "I did not authorize this transaction",
                "Someone stole funds from my wallet",
                "My withdrawal completed but never arrived",
                "Please reverse the duplicated card charge",
                "How does staking work",
                "What are your withdrawal fees",
            ],
            "label": [
                "account-access",
                "account-access",
                "fraud-report",
                "fraud-report",
                "transaction-dispute",
                "transaction-dispute",
                "general",
                "general",
            ],
        }
    )


def test_predict_returns_a_known_label_and_probability():
    router = TicketRouter().fit(small_training_frame())

    result = router.predict("Someone made a transaction that I did not approve")

    assert result.label in EXPECTED_LABELS
    assert 0.0 <= result.confidence <= 1.0


def test_saved_model_preserves_predictions(tmp_path: Path):
    router = TicketRouter().fit(small_training_frame())
    model_path = tmp_path / "router.joblib"
    text = "I cannot access my account"

    before = router.predict(text)
    router.save(model_path)
    after = load_router(model_path).predict(text)

    assert after.label == before.label
    assert after.confidence == before.confidence
