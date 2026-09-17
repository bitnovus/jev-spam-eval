"""How many labeled emails does a TF-IDF classifier need to match TypeSafe, which uses none?

Uses the same unique emails, near-duplicate clusters and grouped 5-fold splits as tfidf_baseline.py
(spam or ham) and phish.py (the three-way main test). The only change is that each training fold is
cut down to n emails, keeping the class shares, before the classifier is fitted. Small training sets
are drawn five times with different seeds and the scores averaged; the test folds never change.
Writes one row per run to results/learning_curve_<dataset>.jsonl. No API calls.

Usage: uv run learning_curve.py [--dataset email|lingspam|phish]
"""

import argparse
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB

import phish
from spam_noul import RESULTS_DIR, THRESHOLD
from tfidf_baseline import load_unique_rows, near_duplicate_groups

SIZES = [25, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
SEEDS = 5
MODELS = {
    "logistic regression": lambda: LogisticRegression(max_iter=2000, C=10),
    "naive Bayes": lambda: MultinomialNB(alpha=0.1),
}


def load(dataset: str) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    """(texts, class of each email, TypeSafe's answer to each question)."""
    if dataset == "phish":
        rows = [r for r in map(json.loads, (RESULTS_DIR / "phish_main_all.jsonl").open()) if "choices" in r]
        y = np.array([phish.CLASSES.index(r["label"]) for r in rows])
        answers = {q: np.array([phish.CLASSES.index(phish.predicted(r, q)) for r in rows]) for q in phish.DESCRIBED}
        return phish.texts_of(rows), y, answers
    rows, texts = load_unique_rows(dataset)
    y = np.array([int(r["label"] == "spam") for r in rows])
    answers = {q: np.array([int(r["nouls"][q] >= THRESHOLD) for r in rows])
               for q in ("spam_plain", "spam_structured_criteria")}
    return texts, y, answers


def scores(guess: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(accuracy, balanced accuracy: the average over classes of the share of each class guessed right)."""
    return float((guess == y).mean()), float(np.mean([(guess[y == c] == c).mean() for c in np.unique(y)]))


def out_of_fold(texts, y, splits, model_factory, n: int | None, seed: int) -> np.ndarray:
    guess = np.zeros(len(texts), dtype=int)
    for train, test in splits:
        if n:
            train, _ = train_test_split(train, train_size=n, stratify=y[train], random_state=seed)
        vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=200000)
        model = model_factory().fit(vec.fit_transform([texts[i] for i in train]), y[train])
        guess[test] = model.predict(vec.transform([texts[i] for i in test]))
    return guess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["email", "lingspam", "phish"], default="email")
    args = parser.parse_args()
    out = RESULTS_DIR / f"learning_curve_{args.dataset}.jsonl"

    texts, y, answers = load(args.dataset)
    groups = near_duplicate_groups(texts, 0.8)
    splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(texts, y, groups))
    full = min(len(train) for train, _ in splits)
    print(f"{len(texts)} emails, about {full} in each training fold")
    for question, guess in answers.items():
        accuracy, balanced = scores(guess, y)
        print(f"  TypeSafe {question:28s} labels      0  accuracy {accuracy:.4f}  balanced {balanced:.4f}")

    models = MODELS if args.dataset != "phish" else {"logistic regression": MODELS["logistic regression"]}
    with out.open("w") as f:
        for n in [s for s in SIZES if s < full] + [None]:
            for name, factory in models.items():
                runs = [scores(out_of_fold(texts, y, splits, factory, n, seed), y) for seed in range(SEEDS if n else 1)]
                for seed, (accuracy, balanced) in enumerate(runs):
                    f.write(json.dumps({"model": name, "labels": n or full, "seed": seed, "accuracy": accuracy,
                                        "balanced_accuracy": balanced}) + "\n")
                accuracy, balanced = np.mean(runs, axis=0)
                low, high = min(r[0] for r in runs), max(r[0] for r in runs)
                print(f"  TF-IDF {name:30s} labels {n or full:6d}  accuracy {accuracy:.4f} ({low:.4f}-{high:.4f})  "
                      f"balanced {balanced:.4f}", flush=True)
    print(f"wrote {out.relative_to(out.parent.parent)}")


if __name__ == "__main__":
    main()
