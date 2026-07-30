# Support-ticket router

A small, production-shaped text-classification service for routing crypto/fintech
support messages into:

- `account-access`
- `transaction-dispute`
- `fraud-report`
- `general`

The core model is **TF-IDF word features plus logistic regression**. This is a
deliberate choice for a dataset of roughly 400 labeled messages: it is fast,
reproducible, interpretable, inexpensive to serve, and difficult to beat
responsibly with a much more complex system under the assessment's time budget.

## Quick start

This project is managed with [uv](https://docs.astral.sh/uv/). Install uv first
if you do not already have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1. Sync the environment

From the repository root:

```bash
uv sync
```

This creates `.venv`, installs runtime dependencies from `uv.lock`, and includes
the `dev` dependency group (pytest). Python `3.11+` is required
(see `.python-version`).

Prefer `uv run ...` below so you do not need to activate the virtualenv. If you
prefer activation:

```bash
source .venv/bin/activate   # macOS/Linux
```

### 2. Run the tests

```bash
uv run pytest
```

The tests cover:

- invalid single-message input;
- invalid CSV schemas and labels;
- prediction output shape;
- model save/load reproducibility;
- end-to-end holdout CSV scoring and row preservation.

### 3. Evaluate the candidate models

```bash
uv run python scripts/evaluate.py \
  --input data/train.csv \
  --output artifacts/evaluation.json
```

This command:

1. validates the dataset;
2. creates a reproducible 80/20 stratified split;
3. compares balanced and unweighted logistic regression;
4. reports accuracy, macro F1, fraud precision/recall/F1, confusion matrix,
   and a full classification report per class;
5. runs five stratified cross-validation as a stability check;
6. writes all results to artifacts/evaluation.json.

### 4. Train the final artifact

After evaluation and model selection, train on all labeled development rows:

```bash
uv run python scripts/train.py \
  --input data/train.csv \
  --output artifacts/ticket_router.joblib
```

### 5. Predict one message

```bash
uv run python scripts/predict.py \
  "Someone transferred BTC from my wallet without my permission"
```

### 6. Score a hidden-holdout CSV

The input CSV must contain a `text` column. Other columns, such as a ticket ID,
are preserved.

```bash
uv run python scripts/score_holdout.py \
  --input data/holdout.csv \
  --output artifacts/holdout_predictions.csv \
  --model artifacts/ticket_router.joblib
```

Output columns include all original columns plus:

- `prediction`
- `confidence`