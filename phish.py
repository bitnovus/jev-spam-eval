"""Ham, spam or phishing: a three-way TypeSafe Choice question, compared with a TF-IDF classifier.

Ham and spam come from email-dataset, using only emails that keep their full headers. Phishing comes
from Jose Nazario's phishing corpus (CC-BY-4.0, see fetch_nazario.sh).

- main: the 2006-07 mailbox, closest in time to the ham and spam, with equal numbers of each category.
- fresh: the 2005-06 mailbox with ham and spam not used in the main test. The phishing description with
  urgency and authority was written after reading the main test's mistakes, so this set checks it on
  messages that played no part in writing it.
- recent: the 2024 and 2025 mailboxes, a TypeSafe-only check; there is no ham or spam from those years.

Every message is asked three Choice questions in one request: the original descriptions, the same
descriptions with phishing's appeals to urgency and authority spelled out, and category names only.

Usage:
  uv run phish.py --per-class 50 --seed 7      # small balanced pilot on the main pool
  uv run phish.py --all --concurrency 16       # all three sets (add --resume to retry failures)
  uv run phish.py --report                     # report on saved results, no API calls
"""

import argparse
import asyncio
import json
import random
import time
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from typesafe_sdk import AsyncTypeSafeClient, Choice

from spam_noul import (
    CONCURRENCY,
    RESULTS_DIR,
    SPAM_BOUNDARY,
    auc,
    classify,
    list_files,
    load_api_key,
    mbox_messages,
    read_email,
    require_dataset,
)
from tfidf_baseline import near_duplicate_groups

CLASSES = ["ham", "spam", "phish"]
OPTION_TO_CLASS = {"legitimate": "ham", "spam": "spam", "phishing": "phish"}
MAIN_PHISH = ["phishing3.mbox"]
FRESH_PHISH = ["phishing2.mbox"]
RECENT_PHISH = ["phishing-2024", "phishing-2025"]
PRICE_PER_INPUT_TOKEN = 0.042 / 1e6
INSTRUCTIONS = "Is `email` legitimate, spam, or phishing?"

LEGITIMATE = {"what": SPAM_BOUNDARY["false"]["what"], "includes": SPAM_BOUNDARY["false"]["includes"]}
SPAM = {
    "what": "Unsolicited bulk email the recipient never signed up for, selling or promoting something, "
    "or running a scam that does not pretend to be a real organization or person the recipient deals with",
    "includes": [
        "advertising or offers from senders the recipient has no relationship with",
        "adult content, pills, cheap software, mortgage, debt, and get-rich-quick offers",
        "advance-fee, lottery, and inheritance scams from strangers",
        "mailings that announce the recipient was added to a list or given a subscription they did not request",
        "text padded with random words or character strings to get past filters",
    ],
}
PHISHING = {
    "what": "Email that pretends to come from a real bank, company, service, or colleague to trick the "
    "recipient into giving up passwords, account or payment details, or personal information, or into "
    "opening a malicious link or attachment",
    "includes": [
        "fake account alerts, suspensions, or requests to verify or update account details",
        "fake password resets, login warnings, or mailbox storage notices",
        (
            "fake invoices, payment failures, refunds, or delivery problems that ask the recipient to "
            "log in or open an attachment"
        ),
        "fake shared documents or signature requests",
    ],
}
PHISHING_URGENCY_AUTHORITY = {
    "what": PHISHING["what"],
    "common_tactics": {
        "authority": (
            "It poses as someone the recipient is expected to trust or obey, such as a bank, a well-known "
            "company or marketplace, the recipient's IT department or email administrator, a manager or "
            "executive, or a government agency, often with official-looking names, logos, reference "
            "numbers, or security wording."
        ),
        "urgency": (
            "It pressures the recipient to act right away to avoid a loss or miss out, such as an account "
            "about to be suspended or closed, unauthorized activity, a failed payment, an unanswered buyer "
            "question or unpaid item, an expiring password or full mailbox, a legal or tax penalty, or a "
            "short deadline."
        ),
    },
    "includes": PHISHING["includes"],
}

QUESTIONS = {
    "category": Choice(
        instructions=INSTRUCTIONS,
        criteria={"legitimate": LEGITIMATE, "spam": SPAM, "phishing": PHISHING},
    ),
    "category_urgency_authority": Choice(
        instructions=INSTRUCTIONS,
        criteria={"legitimate": LEGITIMATE, "spam": SPAM, "phishing": PHISHING_URGENCY_AUTHORITY},
    ),
    "category_names_only": Choice(
        instructions=INSTRUCTIONS,
        criteria={"legitimate": None, "spam": None, "phishing": None},
    ),
}
DESCRIBED = ["category", "category_urgency_authority"]


def state_text(e: dict) -> str:
    return f"{e['subject']}\n{e.get('from', '')}\n{e['body']}"


def dataset_of(file: str) -> str:
    return "nazario" if "#" in file else "email"


def mailbox_items(names: list[str]) -> list[dict]:
    items = []
    for name in names:
        for index in range(len(mbox_messages(name))):
            file = f"{name}#{index}"
            bucket, e = read_email(file, "nazario")
            items.append({"file": file, "label": "phish", "bucket": bucket, "email": e})
    return items


def unique_items(items: list[dict], exclude: set[str]) -> list[dict]:
    seen, unique = set(), []
    for item in items:
        text = state_text(item["email"])
        if text in seen or text in exclude or len(item["email"]["subject"] + item["email"]["body"]) < 20:
            continue
        seen.add(text)
        unique.append(item)
    return unique


def load_pools() -> dict[str, list[dict]]:
    """Unique, non-empty messages per pool. Exact texts shared between the original pools are dropped,
    and fresh-set phishing that repeats the text of any original pool message is dropped too."""
    require_dataset("email")
    require_dataset("nazario")
    pools = {"ham": [], "spam": [], "phish_main": [], "phish_recent": []}
    for file, label in list_files("email"):
        bucket, e = read_email(file, "email")
        if bucket == "raw-headers":
            pools[label].append({"file": file, "label": label, "bucket": bucket, "email": e})
    pools["phish_main"] = mailbox_items(MAIN_PHISH)
    pools["phish_recent"] = mailbox_items(RECENT_PHISH)

    sources = Counter()
    for pool in pools.values():
        sources.update({state_text(item["email"]) for item in pool})
    shared = {text for text, n in sources.items() if n > 1}
    for name, pool in pools.items():
        pools[name] = unique_items(pool, shared)
        print(f"{name}: {len(pool)} messages, {len(pools[name])} unique and non-empty")
    print(f"texts found in more than one pool (dropped): {len(shared)}")

    fresh = mailbox_items(FRESH_PHISH)
    pools["phish_fresh"] = unique_items(fresh, set(sources))
    print(f"phish_fresh: {len(fresh)} messages, {len(pools['phish_fresh'])} unique, non-empty and not in another pool")
    return pools


def build_sets(per_class: int | None, seed: int) -> dict[str, list[dict]]:
    pools = load_pools()
    rng = random.Random(seed)
    n = per_class or len(pools["phish_main"])
    main = [item for key in ("ham", "spam", "phish_main") for item in rng.sample(pools[key], n)]
    rng.shuffle(main)
    sets = {"main": main}
    if per_class is None:
        used = {item["file"] for item in main}
        n_fresh = len(pools["phish_fresh"])
        fresh = [
            item
            for key in ("ham", "spam")
            for item in rng.sample([i for i in pools[key] if i["file"] not in used], n_fresh)
        ] + pools["phish_fresh"]
        rng.shuffle(fresh)
        sets["fresh"] = fresh
        sets["recent"] = pools["phish_recent"]
    return sets


def predicted(row: dict, question: str) -> str:
    return OPTION_TO_CLASS[row["choices"][question]["choice"]]


def class_probabilities(row: dict, question: str) -> list[float]:
    return [row["choices"][question]["probabilities"][option] for option in OPTION_TO_CLASS]


def confusion_3way(actual: list[str], guess: list[str]) -> np.ndarray:
    return np.array([[sum(a == ac and g == gc for a, g in zip(actual, guess)) for gc in CLASSES] for ac in CLASSES])


def print_confusion(name: str, m: np.ndarray) -> None:
    f1s = []
    print(f"\n  {name}: accuracy {np.trace(m) / m.sum():.4f}")
    print(f"    {'actual, predicted':20s}" + "".join(f"{c:>8s}" for c in CLASSES) + "   precision  recall")
    for i, c in enumerate(CLASSES):
        precision = m[i, i] / m[:, i].sum() if m[:, i].sum() else 0.0
        recall = m[i, i] / m[i].sum()
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        print(f"    {c:20s}" + "".join(f"{v:8d}" for v in m[i]) + f"   {precision:9.3f}  {recall:6.3f}")
    print(f"    macro F1 {np.mean(f1s):.4f}")


def texts_of(rows: list[dict]) -> list[str]:
    return [state_text(read_email(r["file"], dataset_of(r["file"]))[1]) for r in rows]


def new_model():
    return TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=200000), LogisticRegression(max_iter=2000, C=10)


def tfidf_cross_validated(rows: list[dict]) -> np.ndarray:
    """Out-of-fold class probabilities from 5-fold CV with near-duplicate clusters kept in one fold."""
    texts = texts_of(rows)
    y = np.array([CLASSES.index(r["label"]) for r in rows])
    groups = near_duplicate_groups(texts, 0.8)
    probs = np.zeros((len(rows), len(CLASSES)))
    for train, test in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(texts, y, groups):
        vec, model = new_model()
        model.fit(vec.fit_transform([texts[i] for i in train]), y[train])
        probs[np.ix_(test, model.classes_)] = model.predict_proba(vec.transform([texts[i] for i in test]))
    return probs


def tfidf_trained_on(train_rows: list[dict], test_rows: list[dict]) -> np.ndarray:
    vec, model = new_model()
    model.fit(vec.fit_transform(texts_of(train_rows)), [CLASSES.index(r["label"]) for r in train_rows])
    probs = np.zeros((len(test_rows), len(CLASSES)))
    probs[:, model.classes_] = model.predict_proba(vec.transform(texts_of(test_rows)))
    return probs


def near_copies(rows: list[dict], reference: list[dict], threshold: float = 0.8) -> np.ndarray:
    """Whether each row has a reference message at cosine similarity >= threshold."""
    texts, ref = texts_of(rows), texts_of(reference)
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2).fit(texts + ref)
    X, R = vec.transform(texts), vec.transform(ref)
    best = np.zeros(len(rows))
    for start in range(0, len(rows), 500):
        best[start:start + 500] = (X[start:start + 500] @ R.T).max(axis=1).toarray().ravel()
    return best >= threshold


def report_set(title: str, rows: list[dict], tfidf_probs: np.ndarray, tfidf_label: str, subset=None) -> None:
    if subset is not None:
        rows, tfidf_probs = [r for r, keep in zip(rows, subset) if keep], tfidf_probs[subset]
    actual = [r["label"] for r in rows]
    y = np.array([CLASSES.index(a) for a in actual])
    is_phish = [a == "phish" for a in actual]
    print(f"\n=== {title}: {len(rows)} messages {dict(Counter(actual))} ===")

    for question in QUESTIONS:
        print_confusion(f"TypeSafe, {question}", confusion_3way(actual, [predicted(r, question) for r in rows]))
        phish_probs = [r["choices"][question]["probabilities"]["phishing"] for r in rows]
        print(f"    phishing vs the rest, AUC of P(phishing): {auc(phish_probs, is_phish):.4f}")
    print_confusion(tfidf_label, confusion_3way(actual, [CLASSES[k] for k in tfidf_probs.argmax(axis=1)]))
    print(f"    phishing vs the rest, AUC of P(phishing): {auc(list(tfidf_probs[:, 2]), is_phish):.4f}")

    before = [predicted(r, "category") for r in rows]
    after = [predicted(r, "category_urgency_authority") for r in rows]
    moves = Counter((a, b, c) for a, b, c in zip(actual, before, after) if b != c)
    print("\n  answers that changed with urgency and authority added (actual: before -> after):")
    for (a, b, c), n in moves.most_common():
        print(f"    {a}: {b} -> {c}  {n}")

    print("\n  accuracy on the rest after reviewing the least confident answers:")
    candidates = (
        ("category", np.array([CLASSES.index(p) for p in before]),
         np.array([r["choices"]["category"]["confidence"] for r in rows])),
        ("urgency_authority", np.array([CLASSES.index(p) for p in after]),
         np.array([r["choices"]["category_urgency_authority"]["confidence"] for r in rows])),
        ("TF-IDF", tfidf_probs.argmax(axis=1), tfidf_probs.max(axis=1)),
    )
    for rate in (0.05, 0.10, 0.20, 0.34):
        parts = []
        for name, pred, conf in candidates:
            keep = conf > np.quantile(conf, rate)
            parts.append(f"{name} {np.mean(pred[keep] == y[keep]):.4f}")
        print(f"    review {rate:.0%}: " + " | ".join(parts))

    for question in DESCRIBED:
        average = (np.array([class_probabilities(r, question) for r in rows]) + tfidf_probs) / 2
        print_confusion(f"average of {question} and TF-IDF",
                        confusion_3way(actual, [CLASSES[k] for k in average.argmax(axis=1)]))

    question = "category_urgency_authority"
    confidence = [r["choices"][question]["confidence"] for r in rows]
    for actual_class, guess_class in (("spam", "phish"), ("phish", "spam"), ("phish", "ham"), ("ham", "phish")):
        idx = sorted((i for i in range(len(rows)) if actual[i] == actual_class and after[i] == guess_class),
                     key=lambda i: -confidence[i])
        print(f"\n  {question}: labeled {actual_class}, called {guess_class}: {len(idx)}; most confident:")
        for i in idx[:6]:
            e = read_email(rows[i]["file"], dataset_of(rows[i]["file"]))[1]
            print(f"    {rows[i]['file']:20s} conf {confidence[i]:.2f}  subject={e['subject'][:55]!r}")


def report_recent(rows: list[dict]) -> None:
    print(f"\n=== recent phishing (TypeSafe only): {len(rows)} messages ===")
    for year in ("2024", "2025", ""):
        subset = [r for r in rows if year in r["file"]]
        for question in QUESTIONS:
            calls = Counter(predicted(r, question) for r in subset)
            print(f"  {year or 'both':5s} {question:27s} n={len(subset):4d}  phishing {calls['phish'] / len(subset):.1%}  "
                  f"spam {calls['spam'] / len(subset):.1%}  legitimate {calls['ham'] / len(subset):.1%}")


def report(results: dict[str, list[dict]]) -> None:
    for name, rows in results.items():
        errors = sum("error" in r for r in rows)
        tokens = sum(r.get("input_tokens") or 0 for r in rows)
        print(f"{name}: {len(rows)} rows, {errors} errors, {tokens:,} input tokens (${tokens * PRICE_PER_INPUT_TOKEN:.2f})")
    ok = {name: [r for r in rows if "choices" in r] for name, rows in results.items()}
    print(f"model {ok['main'][0]['model']}")

    report_set("main test", ok["main"], tfidf_cross_validated(ok["main"]),
               "TF-IDF logistic regression (5-fold CV, near-copies kept together)")
    if "fresh" in ok:
        probs = tfidf_trained_on(ok["main"], ok["fresh"])
        label = "TF-IDF logistic regression (trained on the main test)"
        report_set("fresh test", ok["fresh"], probs, label)
        copies = near_copies(ok["fresh"], ok["main"])
        print(f"\nfresh messages with a near-copy (cosine >= 0.8) in the main test: {int(copies.sum())} "
              f"{dict(Counter(r['label'] for r, c in zip(ok['fresh'], copies) if c))}")
        report_set("fresh test without near-copies of main-test messages", ok["fresh"], probs, label, ~copies)
    if "recent" in ok:
        report_recent(ok["recent"])


async def run(items: list[dict], out, concurrency: int, resume: bool) -> list[dict]:
    """Classify items, writing rows as they arrive; with resume, keep rows that already succeeded."""
    kept = {}
    if resume and out.exists():
        kept = {r["file"]: r for r in map(json.loads, out.open()) if "choices" in r}
    todo = [item for item in items if item["file"] not in kept]
    sem = asyncio.Semaphore(concurrency)
    start = time.monotonic()
    results = list(kept.values())
    with out.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
        async with AsyncTypeSafeClient(api_key=load_api_key()) as client:
            pending = [classify(client, sem, item, QUESTIONS) for item in todo]
            for count, next_result in enumerate(asyncio.as_completed(pending), 1):
                r = await next_result
                results.append(r)
                f.write(json.dumps(r) + "\n")
                if count % 2000 == 0:
                    print(f"  {count}/{len(todo)} in {time.monotonic() - start:.0f}s", flush=True)
    errors = sum("error" in r for r in results)
    print(f"{out.name}: {len(todo)} requests in {time.monotonic() - start:.1f}s ({len(kept)} kept, {errors} errors)")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--per-class", type=int, default=50)
    size.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--resume", action="store_true", help="keep saved rows that succeeded, retry the rest")
    parser.add_argument("--report", action="store_true", help="report on saved results, no API calls")
    args = parser.parse_args()

    tag = "all" if args.all or args.report else f"{args.per_class}x3_seed{args.seed}"
    outs = {
        "main": RESULTS_DIR / f"phish_main_{tag}.jsonl",
        "fresh": RESULTS_DIR / "phish_fresh.jsonl",
        "recent": RESULTS_DIR / "phish_recent.jsonl",
    }
    if args.report:
        report({name: [json.loads(line) for line in out.open()] for name, out in outs.items() if out.exists()})
        return

    sets = build_sets(None if args.all else args.per_class, args.seed)
    RESULTS_DIR.mkdir(exist_ok=True)
    report({name: asyncio.run(run(items, outs[name], args.concurrency, args.resume)) for name, items in sets.items()})


if __name__ == "__main__":
    main()
