from pathlib import Path

import pandas as pd

from scripts.score_holdout import score_file
from constants import CONFIDENCE_COLUMN, PREDICTION_COLUMN
from model import TicketRouter


def test_score_file_preserves_rows_and_adds_predictions(tmp_path: Path):

    # Samples generated with AI
    train_frame = pd.DataFrame(
        {
            "text": [
                "cannot login account",
                "password account blocked",
                "unauthorized wallet transfer",
                "someone stole crypto",
                "withdrawal was duplicated",
                "charge should be reversed",
                "how does staking work",
                "what fees do you charge",
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
    model_path = tmp_path / "model.joblib"
    TicketRouter().fit(train_frame).save(model_path)

    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    pd.DataFrame(
        {
            "ticket_id": [101, 102],
            "text": ["I forgot my password", "How does Ethereum staking work?"],
        }
    ).to_csv(input_path, index=False)

    score_file(input_path, output_path, model_path)
    scored = pd.read_csv(output_path)

    assert len(scored) == 2
    assert scored["ticket_id"].tolist() == [101, 102]
    assert PREDICTION_COLUMN in scored.columns
    assert CONFIDENCE_COLUMN in scored.columns
