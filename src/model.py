
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from constants import EXPECTED_LABELS, LABEL_COLUMN, TEXT_COLUMN
from validation import validate_text, validate_training_frame


def build_pipeline(*, class_weight: str | None = "balanced") -> Pipeline:
    """Create the complete text-classification pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight=class_weight,
                    max_iter=1_000,
                    random_state=42, #seed
                    solver="lbfgs", # algorithmds "liblinear", "saga", "newton-cg"
                ),
            ),
        ]
    )

@dataclass
class Prediction:
    label: str
    confidence: float

    def as_dict(self) -> dict[str, str | float]:
        return {"label": self.label, "confidence": self.confidence}

# Small service wrapper
class TicketRouter:
    def __init__(self, pipeline: Pipeline | None = None) -> None:
        self.pipeline = pipeline or build_pipeline()
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> "TicketRouter":
        checked = validate_training_frame(df)
        self.pipeline.fit(checked[TEXT_COLUMN], checked[LABEL_COLUMN])
        self._is_fitted = True
        return self

    def predict(self, text: object) -> Prediction:
        """Predict one validated message.

        Returning confidence is useful for monitoring and for a future policy
        such as: auto route confident cases and send uncertain cases to manual
        human loop (or potential LLM). 
        """
        self._require_fitted()
        checked_text = validate_text(text)

        probabilities = self.pipeline.predict_proba([checked_text])[0]
        classes = self.pipeline.named_steps["classifier"].classes_
        winning_index = int(np.argmax(probabilities))

        return Prediction(
            label=str(classes[winning_index]),
            confidence=float(probabilities[winning_index])
        )

    def predict_many(self, texts: list[object]) -> list[Prediction]:
        self._require_fitted()
        checked_texts = [validate_text(text) for text in texts]

        probabilities = self.pipeline.predict_proba(checked_texts)
        classes = self.pipeline.named_steps["classifier"].classes_

        results: list[Prediction] = []
        for row in probabilities:
            winning_index = int(np.argmax(row))
            results.append(
                Prediction(
                    label=str(classes[winning_index]),
                    confidence=float(row[winning_index]),
                )
            )
        return results

    def save(self, path: str | Path) -> None:
        """Persist vectorizer and classifier together to prevent skew."""
        self._require_fitted()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, output_path)

    @classmethod
    def from_fitted_pipeline(cls, pipeline: Pipeline) -> "TicketRouter":
        router = cls(pipeline=pipeline)
        router._is_fitted = True
        return router

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("model is not fitted; call fit() or load_router() first")

def load_router(path: str | Path) -> TicketRouter:
    """Load a previously trained router from disk."""
    pipeline: Any = joblib.load(Path(path))
    if not isinstance(pipeline, Pipeline):
        raise TypeError("model artifact does not contain a scikit-learn Pipeline")
    return TicketRouter.from_fitted_pipeline(pipeline)