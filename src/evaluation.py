from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from constants import EXPECTED_LABELS, LABEL_COLUMN, TEXT_COLUMN
from model import build_pipeline


@dataclass(frozen=True)
class EvaluationSummary:
    """Compact JSON summary for report"""

    accuracy: float
    macro_f1: float
    fraud_precision: float
    fraud_recall: float
    fraud_f1: float
    confusion_matrix: list[list[int]]
    classification_report: dict
    cross_validation_macro_f1_mean: float
    cross_validation_macro_f1_std: float
    cross_validation_fraud_recall_mean: float
    cross_validation_fraud_recall_std: float

    def as_dict(self) -> dict:
        return asdict(self)



def evaluate_pipeline(
    *,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    class_weight: str | None,
) -> EvaluationSummary:
    """Fit on train only and evaluate once on untouched validation data.

    Macro F1 is the primary aggregate metric because it gives each route equal
    weight despite class imbalance. Fraud recall is reported separately because
    missing a true fraud report is the highest stakes error.
    """
    pipeline = build_pipeline(class_weight=class_weight)
    pipeline.fit(train_frame[TEXT_COLUMN], train_frame[LABEL_COLUMN])

    predictions = pipeline.predict(validation_frame[TEXT_COLUMN])

    fraud_precision, fraud_recall, fraud_f1, _ = (
        precision_recall_fscore_support(
            validation_frame[LABEL_COLUMN],
            predictions,
            labels=["fraud-report"],
            average=None,
            zero_division=0,
        )
    )

    # Cross-validation is a stability check. With only 50 fraud samples in the
    # full dataset, a single split can produce an optimistic or pessimistic result, so I shuffle
    # it in 5 splits.
    all_development_data = pd.concat(
        [train_frame, validation_frame],
        ignore_index=True,
    )
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        build_pipeline(class_weight=class_weight),
        all_development_data[TEXT_COLUMN],
        all_development_data[LABEL_COLUMN],
        cv=splitter,
        scoring={
            "macro_f1": "f1_macro",
            "fraud_recall": _fraud_recall_scorer,
        },
        n_jobs=None,
        return_train_score=False,
    )

    return EvaluationSummary(
        accuracy=float(
            accuracy_score(validation_frame[LABEL_COLUMN], predictions)
        ),
        macro_f1=float(
            f1_score(
                validation_frame[LABEL_COLUMN],
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        fraud_precision=float(fraud_precision[0]),
        fraud_recall=float(fraud_recall[0]),
        fraud_f1=float(fraud_f1[0]),
        confusion_matrix=confusion_matrix(
            validation_frame[LABEL_COLUMN],
            predictions,
            labels=list(EXPECTED_LABELS),
        ).tolist(),
        classification_report=classification_report(
            validation_frame[LABEL_COLUMN],
            predictions,
            labels=list(EXPECTED_LABELS),
            output_dict=True,
            zero_division=0,
        ),
        cross_validation_macro_f1_mean=float(
            np.mean(cv_results["test_macro_f1"])
        ),
        cross_validation_macro_f1_std=float(
            np.std(cv_results["test_macro_f1"])
        ),
        cross_validation_fraud_recall_mean=float(
            np.mean(cv_results["test_fraud_recall"])
        ),
        cross_validation_fraud_recall_std=float(
            np.std(cv_results["test_fraud_recall"])
        ),
    )


def _fraud_recall_scorer(estimator, x, y) -> float:
    """scikit-learn-compatible custom scorer for fraud-report recall."""
    predictions = estimator.predict(x)
    _, recall, _, _ = precision_recall_fscore_support(
        y,
        predictions,
        labels=["fraud-report"],
        average=None,
        zero_division=0,
    )
    return float(recall[0])
