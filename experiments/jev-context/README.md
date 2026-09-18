# Richer email evidence for Jev Choice

This experiment reads the repository's source datasets and original predictions and writes results in this directory. It compares the original text-only Choice with enriched metadata and evidence-focused wording, followed by matched-evidence logistic regression.

## Reproduce from the repository root

```sh
uv sync
./fetch_dataset.sh
./fetch_nazario.sh
uv run python experiments/jev-context/evaluate.py --check
uv run python experiments/jev-context/report.py
uv run python experiments/jev-context/regression.py
```

The checks, reports, and regression analysis make no API calls. The existing mailbox loader opens files with `rb+` although this experiment only reads them; a restrictive sandbox may need permission for that operation.

## New API runs

The runner uses the repository's existing `TYPESAFE_API_KEY` loader. It retains successful requests and retries failed ones. The included predictions already cover the complete experiment, so choose a new output directory for an independent replication:

```sh
JEV_CONTEXT_OUTPUT_DIR=experiments/jev-context/rerun uv run python experiments/jev-context/evaluate.py --limit 3
JEV_CONTEXT_OUTPUT_DIR=experiments/jev-context/rerun uv run python experiments/jev-context/evaluate.py
JEV_CONTEXT_OUTPUT_DIR=experiments/jev-context/rerun uv run python experiments/jev-context/report.py
JEV_CONTEXT_OUTPUT_DIR=experiments/jev-context/rerun uv run python experiments/jev-context/regression.py
```

Use the same output override for evaluation and reports. Run manifests record the model and script hash. Scripts are expected to remain in this directory even when outputs go elsewhere.

## Design

All 9,886 saved main/fresh/recent messages are evaluated with `jev-1.13.0`. The text-only control receives the original defined Choice. A second request receives enriched state and two independent Choice questions: the original question and evidence-focused wording with the same definitions. The two questions do not consume one another's answers. Labels and source paths are not passed to Jev.

Code adds Reply-To, visible link text paired with destinations and hostnames, and attachment filenames/content types. It preserves the original subject/from/body exactly. HTML entities and MIME transfer encodings are decoded; link pairs are deduplicated. Plain-text URLs are included. No links are visited and no suspiciousness score is constructed. Relative destinations are retained without inventing a host.

Bodies remain capped at 6,000 characters. Link extraction uses full textual MIME parts, bounded to 80 records, roughly 24,000 serialized characters, 2,000 characters per destination, and 500 per visible label. Attachment metadata is capped at 40 records. These bounds and best-effort parsing of malformed historical mail remain limitations.

Regression verifies all reconstructed Jev input hashes before fitting. Both variants use the same original-text near-duplicate groups, folds, word-vectorizer settings, and regression hyperparameters. Fresh evaluation uses main training only; recent evaluation uses main + fresh. A common fresh subset excludes near-copies of main. The fixed 50/50 ensembles use the original defined Choice, not the evidence-focused variant. Jev classification uses its returned Choice label; rounded probabilities are normalized for ensemble averaging and probability metrics.

## Artifacts and provenance

- `REPORT.md`, `metrics.json`: the paired Jev comparison.
- `REGRESSION_REPORT.md`, `regression_metrics.json`: matched-evidence comparison.
- `predictions.jsonl`: Jev answers, probabilities, input checksums, usage, and timings; no email bodies or credentials.
- `regression_predictions.jsonl`, `regression_splits.jsonl`: model predictions and shared main-set folds.
- `manifest.json`, `regression_manifest.json`: original-run settings and hashes.
- `provenance/evaluate.py`, `provenance/regression.py`: exact original runner snapshots matching the saved manifests. These retain the original local paths and are archival references, not the reproduction entry points.

The active `evaluate.py` was adapted for a repository-relative source path and an optional output-directory override when this experiment was packaged. Its parsing and question logic match the archived runner. Active `regression.py` is unchanged from the original run. Re-running an active script writes a new manifest for that run.

The source labels contain known noise, and corpus sources and campaigns are confounded with classes. These previously inspected datasets are exploratory. The recent set contains phishing only. Equal input evidence does not imply equal training supervision.
