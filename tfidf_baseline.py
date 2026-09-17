"""Compare TypeSafe against TF-IDF classifiers trained on the dataset's labels.

Reads TypeSafe scores from results/results_criteria_all.jsonl, re-reads the exact email text TypeSafe
saw from the dataset, removes exact duplicates, clusters near-duplicates, and scores every email
out-of-fold with 5-fold cross-validation in which each near-duplicate cluster stays entirely on one
side of the split. Writes per-email scores to results/tfidf_oof.jsonl.

Usage: uv run tfidf_baseline.py [--similarity 0.8]
"""

import argparse
import json
import random
from collections import Counter
from math import comb

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

from spam_noul import RESULTS_DIR, THRESHOLD, auc, read_email, require_dataset

RESULTS = RESULTS_DIR / "results_criteria_all.jsonl"
OUT = RESULTS_DIR / "tfidf_oof.jsonl"

TS_BARE = "TypeSafe: bare question (0 labels)"
TS = "TypeSafe: structured criteria (0 labels)"
LR = "TF-IDF logreg, grouped CV (~14.8K labels)"
NB = "TF-IDF naive Bayes, grouped CV"
LR_LEAKY = "TF-IDF logreg, ungrouped CV (leaky)"
AVERAGE = "Average of structured criteria + logreg"


def load_unique_rows() -> tuple[list[dict], list[str]]:
    """Rows with their state text, keeping the first file (by name) of each exact-duplicate text."""
    require_dataset()
    rows = sorted((json.loads(line) for line in RESULTS.open()), key=lambda r: r["file"])
    seen, unique, texts = set(), [], []
    for r in rows:
        if "nouls" not in r:
            continue
        e = read_email(r["file"])[1]
        text = f"{e['subject']}\n{e.get('from', '')}\n{e['body']}"
        if text not in seen:
            seen.add(text)
            unique.append({**r, "subject": e["subject"], "body": e["body"]})
            texts.append(text)
    return unique, texts


def near_duplicate_groups(texts: list[str], threshold: float) -> np.ndarray:
    """Connected components of the cosine >= threshold graph (unsupervised, no labels)."""
    X = TfidfVectorizer(sublinear_tf=True, min_df=2, dtype=np.float32).fit_transform(texts)
    parent = list(range(len(texts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for start in range(0, len(texts), 500):
        sims = (X[start:start + 500] @ X.T).tocoo()
        for i, j, v in zip(sims.row, sims.col, sims.data):
            if v >= threshold and start + i != j:
                a, b = find(start + i), find(j)
                if a != b:
                    parent[a] = b
    return np.array([find(i) for i in range(len(texts))])


def out_of_fold(texts, y, splits, model_factory) -> np.ndarray:
    probs = np.zeros(len(texts))
    for train, test in splits:
        vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=200000)
        X_train = vec.fit_transform([texts[i] for i in train])
        model = model_factory().fit(X_train, y[train])
        probs[test] = model.predict_proba(vec.transform([texts[i] for i in test]))[:, 1]
    return probs


def metrics_line(name: str, p, y) -> None:
    pred = p >= THRESHOLD
    tp, tn = int((y & pred).sum()), int((~y & ~pred).sum())
    fp, fn = int((~y & pred).sum()), int((y & ~pred).sum())
    balanced = (tp / (tp + fn) + tn / (tn + fp)) / 2
    print(f"  {name:42s} acc {(tp + tn) / len(y):.4f}  balanced {balanced:.4f}  prec {tp / (tp + fp):.3f}  "
          f"rec {tp / (tp + fn):.3f}  AUC {auc(list(p), list(y)):.4f}  FN {fn:4d}  FP {fp:4d}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--similarity", type=float, default=0.8)
    args = parser.parse_args()

    rows, texts = load_unique_rows()
    y = np.array([r["label"] == "spam" for r in rows])
    print(f"{len(rows)} unique emails (ham {int((~y).sum())}, spam {int(y.sum())})")

    groups = near_duplicate_groups(texts, args.similarity)
    sizes = Counter(groups)
    print(f"near-duplicate clusters at cosine >= {args.similarity}: {len(sizes)} clusters, "
          f"{sum(1 for s in sizes.values() if s == 1)} singletons, "
          f"{sum(s for s in sizes.values() if s > 1)} emails in multi-email clusters")
    for gid, size in sizes.most_common(5):
        print(f"  cluster size {size:4d} ({size / len(rows):.1%} of data), spam share {y[groups == gid].mean():.2f}")

    grouped = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(texts, y, groups))
    leaky = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(texts, y))
    scores = {
        TS_BARE: np.array([r["nouls"]["spam_plain"] for r in rows]),
        TS: np.array([r["nouls"]["spam_structured_criteria"] for r in rows]),
    }
    print("training TF-IDF models (5 folds each)...", flush=True)
    scores[LR] = out_of_fold(texts, y, grouped, lambda: LogisticRegression(max_iter=2000, C=10))
    scores[NB] = out_of_fold(texts, y, grouped, lambda: MultinomialNB(alpha=0.1))
    scores[LR_LEAKY] = out_of_fold(texts, y, leaky, lambda: LogisticRegression(max_iter=2000, C=10))
    scores[AVERAGE] = (scores[TS] + scores[LR]) / 2

    with OUT.open("w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"file": r["file"], "label": r["label"], "cluster": int(groups[i]),
                                **{name: float(s[i]) for name, s in scores.items()}}) + "\n")
    print(f"wrote {OUT.relative_to(OUT.parent.parent)}")

    print(f"\nall {len(rows)} unique emails (threshold {THRESHOLD}):")
    for name, p in scores.items():
        metrics_line(name, p, y)

    print("\nconfusion matrices (rows actual, columns predicted):")
    for name in (TS_BARE, TS, LR, NB, AVERAGE):
        pred = scores[name] >= THRESHOLD
        tn, fp = int((~y & ~pred).sum()), int((~y & pred).sum())
        fn, tp = int((y & ~pred).sum()), int((y & pred).sum())
        print(f"  {name}\n    actual ham : {tn:6d} ham  {fp:5d} spam ({fp / (tn + fp):.2%} of ham flagged)"
              f"\n    actual spam: {fn:6d} ham  {tp:5d} spam ({fn / (fn + tp):.2%} of spam missed)")

    ts, lr, nb = scores[TS], scores[LR], scores[NB]
    print("\naccuracy by format (structured criteria / logreg grouped / naive Bayes grouped):")
    for bucket in sorted({r["bucket"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["bucket"] == bucket]
        accs = [np.mean([(s[i] >= THRESHOLD) == y[i] for i in idx]) for s in (ts, lr, nb)]
        print(f"  {bucket:13s} n={len(idx):5d}  " + "  ".join(f"{a:.4f}" for a in accs))

    print("\nreview the emails scored closest to 0.5, at the same review rate for each:")
    for name in (TS, LR, AVERAGE):
        p = scores[name]
        distance = np.abs(p - 0.5)
        for rate in (0.025, 0.046):
            keep = distance > np.quantile(distance, rate)
            wrong = int(((p[keep] >= THRESHOLD) != y[keep]).sum())
            print(f"  {name:42s} review {rate:.1%}: accuracy on the rest {1 - wrong / keep.sum():.4f} ({wrong} errors)")

    ts_right, lr_right = (ts >= THRESHOLD) == y, (lr >= THRESHOLD) == y
    print("\nwho gets which emails right (structured criteria vs logreg grouped):")
    for name, mask in (("both right", ts_right & lr_right), ("only TypeSafe right", ts_right & ~lr_right),
                       ("only logreg right", ~ts_right & lr_right), ("both wrong", ~ts_right & ~lr_right)):
        print(f"  {name:20s} {int(mask.sum()):6d}   (ham {int((mask & ~y).sum())}, spam {int((mask & y).sum())})")
    b, c = int((ts_right & ~lr_right).sum()), int((~ts_right & lr_right).sum())
    p_value = min(1.0, 2 * sum(comb(b + c, i) for i in range(min(b, c) + 1)) / 2 ** (b + c))
    print(f"  exact McNemar test on the {b + c} disagreements: p = {p_value:.2f}")

    rng = random.Random(0)
    for name, mask in (("only TypeSafe right", ts_right & ~lr_right), ("only logreg right", ~ts_right & lr_right),
                       ("both wrong", ~ts_right & ~lr_right)):
        idx = rng.sample(list(np.flatnonzero(mask)), min(12, int(mask.sum())))
        print(f"\nexamples, {name}:")
        for i in idx:
            r = rows[i]
            print(f"  {r['file']:12s} {r['label']:4s} ts={ts[i]:.2f} lr={lr[i]:.2f}  "
                  f"subj={r['subject'][:40]!r}  body={r['body'][:110]!r}")


if __name__ == "__main__":
    main()
