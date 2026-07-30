
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

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
