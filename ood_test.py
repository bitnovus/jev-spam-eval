"""Out-of-distribution tests: TypeSafe, which is never trained, against TF-IDF classifiers trained on this
repository's data and then shown email unlike anything they were trained on.

1. Ling-Spam (2000, a linguistics mailing list): spam or ham. TypeSafe's saved answers from the Ling-Spam
   run against TF-IDF logistic regression trained on all unique email-dataset emails.
2. Recent phishing (2024-25): TypeSafe's saved three-way answers against a three-way TF-IDF classifier
   trained on the phishing experiment's main and fresh tests (2005-07 phishing, ham and spam).
3. Modern mail (2026): posts to python-list and python-announce-list (legitimate) and a sample of the
   August 2026 spam-trap archive from untroubled.org (spam or phishing; the archive doesn't separate
   them). New TypeSafe calls with the three-way questions against the same three-way TF-IDF classifier.

Usage:
  ./fetch_ood.sh
  uv run ood_test.py --concurrency 16   # classifies the modern mail (about 630 requests), then reports
  uv run ood_test.py --report           # report from saved results, no API calls
"""

import argparse
import asyncio
import functools
import json
import mailbox
import random
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import phish
from spam_noul import CONCURRENCY, RESULTS_DIR, ROOT, THRESHOLD, parse_email, read_email
from tfidf_baseline import load_unique_rows

OOD = ROOT / "ood-corpus"
SPAM_TRAP_SAMPLE = 300
SEED = 7


@functools.cache
def list_messages(name: str) -> list[str]:
    box = mailbox.mbox(OOD / "lists" / name, create=False)
    return [box.get_bytes(key).decode("utf-8", errors="replace") for key in box.iterkeys()]


def read_ood(file: str) -> dict:
    """Parsed state for "lists/<mbox>#<index>" or "spamtrap/<path>", the same way as every other test."""
    if file.startswith("lists/"):
        name, _, index = file.removeprefix("lists/").partition("#")
        raw = list_messages(name)[int(index)]
    else:
        raw = (OOD / file).read_text(errors="replace")
    return parse_email(raw, "raw-headers")


def modern_items() -> list[dict]:
    if not (OOD / "lists").is_dir() or not (OOD / "spamtrap" / "2026" / "08").is_dir():
        raise SystemExit("Modern mail not found; run ./fetch_ood.sh first.")
    legit = [f"lists/{p.name}#{i}" for p in sorted((OOD / "lists").glob("*.mbox")) for i in range(len(list_messages(p.name)))]
    trap = [f"spamtrap/2026/08/{p.name}" for p in sorted((OOD / "spamtrap" / "2026" / "08").iterdir())]

    def unique(files: list[str], label: str) -> list[dict]:
        seen, items = set(), []
        for file in files:
            e = read_ood(file)
            text = phish.state_text(e)
            if text in seen or len(e["subject"] + e["body"]) < 20:
                continue
            seen.add(text)
            items.append({"file": file, "label": label, "bucket": "raw-headers", "email": e})
        return items

    legit_items, trap_items = unique(legit, "ham"), unique(trap, "not_ham")
    trap_items = random.Random(SEED).sample(trap_items, SPAM_TRAP_SAMPLE)
    print(f"modern legitimate posts: {len(legit_items)}; spam-trap sample: {len(trap_items)}")
    return legit_items + trap_items


def text_of(file: str) -> str:
    if file.startswith(("lists/", "spamtrap/")):
        return phish.state_text(read_ood(file))
    return phish.state_text(read_email(file, phish.dataset_of(file))[1])


def fit(texts: list[str], labels: list[int]):
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=200000)
    model = LogisticRegression(max_iter=2000, C=10).fit(vec.fit_transform(texts), labels)
    return vec, model


def binary_line(name: str, spam_scores: np.ndarray, is_spam: np.ndarray) -> None:
    pred = spam_scores >= THRESHOLD
    tp, tn = int((is_spam & pred).sum()), int((~is_spam & ~pred).sum())
    fp, fn = int((~is_spam & pred).sum()), int((is_spam & ~pred).sum())
    balanced = (tp / (tp + fn) + tn / (tn + fp)) / 2
    print(f"  {name:48s} accuracy {(tp + tn) / len(pred):.4f}  balanced {balanced:.4f}  "
          f"missed spam {fn:4d}  ham flagged {fp:4d}")


def test_lingspam() -> list[dict]:
    train_rows, train_texts = load_unique_rows("email")
    vec, model = fit(train_texts, [r["label"] == "spam" for r in train_rows])
    rows, texts = load_unique_rows("lingspam")
    is_spam = np.array([r["label"] == "spam" for r in rows])
    print(f"\n=== 1. Ling-Spam: {len(rows)} unique messages ({int(is_spam.sum())} spam); "
          f"TF-IDF trained on {len(train_rows):,} email-dataset emails ===")
    binary_line("TypeSafe, plain question", np.array([r["nouls"]["spam_plain"] for r in rows]), is_spam)
    binary_line("TypeSafe, detailed criteria", np.array([r["nouls"]["spam_structured_criteria"] for r in rows]), is_spam)
    spam_probability = model.predict_proba(vec.transform(texts))[:, 1]
    binary_line("TF-IDF logistic regression (email-dataset)", spam_probability, is_spam)
    print("  (for reference, TF-IDF trained on Ling-Spam itself scored 0.9857 accuracy in cross-validation)")
    return [{"set": "lingspam", "file": r["file"], "label": r["label"], "spam_probability": round(float(p), 6)}
            for r, p in zip(rows, spam_probability)]


def three_way_model():
    rows = [json.loads(line) for name in ("phish_main_all.jsonl", "phish_fresh.jsonl")
            for line in (RESULTS_DIR / name).open()]
    rows = [r for r in rows if "choices" in r]
    vec, model = fit([text_of(r["file"]) for r in rows], [phish.CLASSES.index(r["label"]) for r in rows])
    print(f"three-way TF-IDF trained on {len(rows):,} emails from the phishing experiment's main and fresh tests")
    return vec, model


def calls_line(name: str, calls: list[str]) -> None:
    counts = Counter(calls)
    n = len(calls)
    print(f"  {name:48s} legitimate {counts['ham'] / n:6.1%}  spam {counts['spam'] / n:6.1%}  phishing {counts['phish'] / n:6.1%}")


def test_recent_phishing(vec, model) -> list[dict]:
    rows = [r for r in map(json.loads, (RESULTS_DIR / "phish_recent.jsonl").open()) if "choices" in r]
    predictions = model.predict(vec.transform([text_of(r["file"]) for r in rows]))
    print(f"\n=== 2. Recent phishing (2024-25): {len(rows)} messages, all phishing ===")
    for question in phish.QUESTIONS:
        calls_line(f"TypeSafe, {question}", [phish.predicted(r, question) for r in rows])
    calls_line("TF-IDF (trained on 2005-07 mail)", [phish.CLASSES[k] for k in predictions])
    return [{"set": "recent_phishing", "file": r["file"], "label": r["label"], "tfidf_call": phish.CLASSES[k]}
            for r, k in zip(rows, predictions)]


def test_modern(rows: list[dict], vec, model) -> list[dict]:
    rows = [r for r in rows if "choices" in r]
    predictions = [phish.CLASSES[k] for k in model.predict(vec.transform([text_of(r["file"]) for r in rows]))]
    print(f"\n=== 3. Modern mail (2026): {len(rows)} messages ===")
    for label, title in (("ham", "legitimate list posts"), ("not_ham", "spam-trap sample")):
        idx = [i for i, r in enumerate(rows) if r["label"] == label]
        print(f"\n  {title} ({len(idx)}):")
        for question in phish.QUESTIONS:
            calls_line(f"TypeSafe, {question}", [phish.predicted(rows[i], question) for i in idx])
        calls_line("TF-IDF (trained on 2005-07 mail)", [predictions[i] for i in idx])

    print("\n  legitimate vs not legitimate (spam or phishing):")
    is_bad = np.array([r["label"] == "not_ham" for r in rows])
    for question in phish.QUESTIONS:
        binary_line(f"TypeSafe, {question}", np.array([phish.predicted(r, question) != "ham" for r in rows], float), is_bad)
    binary_line("TF-IDF (trained on 2005-07 mail)", np.array([p != "ham" for p in predictions], float), is_bad)

    question = "category_urgency_authority"
    for label, ts_wrong, lr_wrong, title in (
        ("ham", lambda r: phish.predicted(r, question) != "ham", lambda p: p == "ham", "legitimate, TypeSafe wrong, TF-IDF right"),
        ("ham", lambda r: phish.predicted(r, question) == "ham", lambda p: p != "ham", "legitimate, TF-IDF wrong, TypeSafe right"),
        ("not_ham", lambda r: phish.predicted(r, question) == "ham", lambda p: p != "ham", "spam trap, TypeSafe wrong, TF-IDF right"),
        ("not_ham", lambda r: phish.predicted(r, question) != "ham", lambda p: p == "ham", "spam trap, TF-IDF wrong, TypeSafe right"),
    ):
        idx = [i for i, r in enumerate(rows) if r["label"] == label and ts_wrong(r) and lr_wrong(predictions[i])]
        print(f"\n  {title}: {len(idx)}")
        for i in idx[:6]:
            e = read_ood(rows[i]["file"])
            print(f"    TypeSafe {phish.predicted(rows[i], question):5s} TF-IDF {predictions[i]:5s}  "
                  f"subject={e['subject'][:50]!r}  from={e.get('from', '')[:30]!r}")
    return [{"set": "modern", "file": r["file"], "label": r["label"], "tfidf_call": p}
            for r, p in zip(rows, predictions)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report", action="store_true", help="report from saved results, no API calls")
    args = parser.parse_args()
    out = RESULTS_DIR / "ood_modern.jsonl"

    if args.report:
        modern = [json.loads(line) for line in out.open()]
    else:
        modern = asyncio.run(phish.run(modern_items(), out, args.concurrency, args.resume))
    errors = sum("error" in r for r in modern)
    tokens = sum(r.get("input_tokens") or 0 for r in modern)
    print(f"modern mail: {len(modern)} rows, {errors} errors, {tokens:,} input tokens (${tokens * phish.PRICE_PER_INPUT_TOKEN:.2f})")

    saved = test_lingspam()
    vec, model = three_way_model()
    saved += test_recent_phishing(vec, model)
    saved += test_modern(modern, vec, model)
    out_predictions = RESULTS_DIR / "ood_tfidf_predictions.jsonl"
    with out_predictions.open("w") as f:
        for row in saved:
            f.write(json.dumps(row) + "\n")
    print(f"\nwrote {out_predictions.name}")


if __name__ == "__main__":
    main()
