"""Ham/spam classification of email-dataset or Ling-Spam with TypeSafe Noul questions.

Usage: uv run spam_noul.py [--dataset email|lingspam] [--questions single|multi|criteria]
                          [--per-class 100 | --all] [--seed 42] [--exclude-from results/a.jsonl ...]
                          [--concurrency 8] [--resume] [--report]

  single:   one "Is this email spam?" Noul, with and without criteria.
  multi:    the spam judgment split into independent signal Nouls in one request,
            combined in code (hand-written rule and cross-validated logistic regression).
  criteria: the single spam Noul with no criteria, generic criteria, and structured criteria
            that spell out where solicited bulk mail falls.
  --exclude-from skips emails already present in earlier results files, for a fresh holdout.
  --all classifies every email; results are written as they arrive and --resume retries only
  emails that are missing or errored in the existing results file.
  --report re-prints the report for an existing results file without calling the API.

Results go to results/results_<questions>_<size>.jsonl (results_lingspam_... for Ling-Spam) and
hold scores, not email text; the text is re-read from the dataset (see fetch_dataset.sh and
fetch_lingspam.sh) whenever it is needed.
"""

import argparse
import asyncio
import email
import functools
import html
import json
import mailbox
import math
import os
import quopri
import random
import re
import time
from collections import Counter
from email import policy
from pathlib import Path

from typesafe_sdk import AsyncTypeSafeClient, Noul

ROOT = Path(__file__).parent
DATASETS = {
    "email": ROOT / "email-dataset" / "dataset",
    "lingspam": ROOT / "lingspam" / "lingspam_public" / "bare",
    "nazario": ROOT / "phishing-corpus",
}
FETCH_SCRIPTS = {"email": "./fetch_dataset.sh", "lingspam": "./fetch_lingspam.sh", "nazario": "./fetch_nazario.sh"}
RESULTS_DIR = ROOT / "results"
LABELS = {"1": "ham", "2": "spam"}
MAX_BODY_CHARS = 6000
CONCURRENCY = 8
THRESHOLD = 0.5

QUESTION_SETS = {
    "single": {
        "spam_criteria": Noul(
            instructions="Is this email spam?",
            criteria={
                "true": "Unsolicited bulk or commercial email: advertising, scams, phishing, "
                "get-rich-quick offers, or mass mailings the recipient did not ask for.",
                "false": "Legitimate email the recipient would expect: personal or work "
                "correspondence, or newsletters and mailing lists they subscribed to.",
            },
        ),
        "spam_plain": Noul(instructions="Is this email spam?"),
    },
    "multi": {
        "spam_plain": Noul(instructions="Is this email spam?"),
        "commercial_offer": Noul(
            instructions="Does `email` advertise or try to sell a product, service, investment, or deal?"
        ),
        "scam_or_phishing": Noul(
            instructions="Is `email` a scam or phishing attempt, such as an advance-fee offer, a fake prize "
            "or lottery win, a fake account alert, or a request for passwords, bank details, or money?"
        ),
        "adult_content": Noul(instructions="Does `email` promote pornography, sexual services, or adult dating?"),
        "drugs_or_health_products": Noul(
            instructions="Does `email` promote prescription drugs, sexual enhancement pills, "
            "or miracle health or weight-loss products?"
        ),
        "bulk_generic": Noul(
            instructions="Was `email` written for mass distribution to many recipients rather than for one "
            "specific recipient?",
            criteria={
                "true": "Generic or no greeting, nothing specific to the recipient, templated marketing copy, "
                "or unsubscribe/remove-me instructions.",
                "false": "Written to a specific person or small group, referencing names, shared context, "
                "or an ongoing conversation.",
            },
        ),
        "filter_evasion": Noul(
            instructions="Does `email` contain random character strings, deliberately misspelled or broken-up "
            "words, or hidden text that look designed to get past spam filters?"
        ),
        "personal_or_work_thread": Noul(
            instructions="Is `email` personal or work correspondence between people who know each other, "
            "or a reply in an ongoing conversation?"
        ),
        "subscribed_list_or_newsletter": Noul(
            instructions="Is `email` from a discussion mailing list, forum digest, news feed, or newsletter "
            "that the recipient plausibly signed up for?",
            criteria={
                "true": "Mailing-list discussion, list footers, digests, news or feed items, or a newsletter "
                "from a service the recipient uses.",
                "false": "Not a list or newsletter, or a mass mailing the recipient never signed up for.",
            },
        ),
    },
}

SPAM_BOUNDARY = {
    "true": {
        "what": "Unsolicited bulk email the recipient never signed up for and has no relationship with the sender",
        "includes": [
            "advertising or offers from senders the recipient has no relationship with",
            "scams, advance-fee fraud, fake prizes, and phishing",
            "adult content, pills, cheap software, mortgage, debt, and get-rich-quick offers",
            "mailings that announce the recipient was added to a list or given a subscription they did not request",
            "text padded with random words or character strings to get past filters",
        ],
    },
    "false": {
        "what": "Email the recipient expects, even when it is automated, commercial, or sent to many people",
        "includes": [
            "personal and work correspondence, including forwards and replies",
            "mailing-list discussions and digests",
            "newsletters, news-feed items, and promotions from sites or stores the recipient signed up for",
            "automated notices such as delivery failures, receipts, and system alerts",
        ],
    },
}

QUESTION_SETS["criteria"] = {
    "spam_plain": QUESTION_SETS["single"]["spam_plain"],
    "spam_generic_criteria": QUESTION_SETS["single"]["spam_criteria"],
    "spam_structured_criteria": Noul(instructions="Is `email` spam?", criteria=SPAM_BOUNDARY),
    "spam_structured_focus": Noul(
        instructions={
            "question": "Is `email` spam?",
            "focus": "Judge whether the recipient asked for or expects this kind of email, "
            "not whether it is commercial or sent to many people.",
        },
        criteria=SPAM_BOUNDARY,
    ),
}

# Signals where a yes points toward ham rather than spam.
HAM_SIGNALS = {"personal_or_work_thread", "subscribed_list_or_newsletter"}


def load_api_key() -> str:
    """TYPESAFE_API_KEY (or TYPESAFEAI_API_KEY) from the environment, then from .env."""
    names = ("TYPESAFE_API_KEY", "TYPESAFEAI_API_KEY")
    values = dict(os.environ)
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            name, _, value = line.partition("=")
            values.setdefault(name.strip(), value.strip().strip("\"'"))
    for name in names:
        if values.get(name):
            return values[name]
    raise SystemExit("Set TYPESAFE_API_KEY in the environment or in .env (see .env.example)")


def format_bucket(raw: str) -> str:
    if raw.startswith("Subject:"):
        return "subject-line"
    if raw.startswith(("From ", "Return-Path", "Received")):
        return "raw-headers"
    return "other"


def html_to_text(text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return html.unescape(text)


def decode_qp(text: str) -> str:
    if re.search(r"=(3D|3C|3E|2C|20)", text):
        return quopri.decodestring(text.encode("ascii", "replace")).decode("ascii", "replace")
    return text


def parse_email(raw: str, bucket: str) -> dict:
    """Turn a raw file into {subject, from?, body} state for the model."""
    if bucket == "raw-headers":
        msg = email.message_from_string(raw, policy=policy.default)
        parts = []
        for part in msg.walk():
            if part.get_content_maintype() != "text":
                continue
            try:
                content = part.get_content()
            except Exception:  # noqa: BLE001 - malformed MIME parts fall back to the raw payload
                content = part.get_payload()
            if not isinstance(content, str):
                continue
            parts.append(html_to_text(content) if part.get_content_subtype() == "html" else content)
        fields = {"subject": str(msg.get("Subject", "")), "from": str(msg.get("From", ""))}
        body = "\n".join(parts)
    else:
        first, _, rest = raw.partition("\n")
        if first.startswith("Subject:"):
            fields = {"subject": first.removeprefix("Subject:").strip()}
            body = rest
        else:
            fields = {"subject": ""}
            body = raw
        body = decode_qp(body)
        if re.search(r"(?i)<(html|body|table|font|p|br|div)\b", body):
            body = html_to_text(body)
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n\s*\n+", "\n\n", body).strip()
    return {**fields, "body": body[:MAX_BODY_CHARS]}


@functools.cache
def mbox_messages(name: str) -> list[str]:
    """Raw messages of a phishing-corpus mailbox, in file order."""
    box = mailbox.mbox(DATASETS["nazario"] / name, create=False)
    return [box.get_bytes(key).decode("utf-8", errors="replace") for key in box.iterkeys()]


def read_email(file: str, dataset: str = "email") -> tuple[str, dict]:
    """Format bucket and parsed state for a message such as "2/0001.eml", "part1/3-1msg1.txt" or
    "phishing3.mbox#12" (the 13th message of a phishing-corpus mailbox)."""
    if dataset == "nazario":
        name, _, index = file.partition("#")
        return "raw-headers", parse_email(mbox_messages(name)[int(index)], "raw-headers")
    raw = (DATASETS[dataset] / file).read_text(errors="replace")
    bucket = format_bucket(raw)
    return bucket, parse_email(raw, bucket)


def require_dataset(dataset: str = "email") -> None:
    root = DATASETS[dataset]
    if dataset == "nazario":
        present = all((root / name).is_file() for name in ("phishing2.mbox", "phishing3.mbox", "phishing-2024", "phishing-2025"))
    else:
        folders = LABELS if dataset == "email" else [f"part{i}" for i in range(1, 11)]
        present = all((root / folder).is_dir() for folder in folders)
    if not present:
        raise SystemExit(f"Dataset not found at {root}; run {FETCH_SCRIPTS[dataset]} first.")


def list_files(dataset: str = "email") -> list[tuple[str, str]]:
    """(file, label) for every message, in a fixed order."""
    root = DATASETS[dataset]
    if dataset == "lingspam":
        # Ling-Spam keeps the 10 parts its authors used for 10-fold tests; spam files start with "spmsg".
        parts = sorted(root.iterdir(), key=lambda part: int(part.name.removeprefix("part")))
        return [(f"{part.name}/{f.name}", "spam" if f.name.startswith("spmsg") else "ham")
                for part in parts for f in sorted(part.iterdir())]
    return [(f"{folder}/{f.name}", label) for folder, label in LABELS.items() for f in sorted((root / folder).iterdir())]


def sample(per_class: int | None, seed: int, exclude: set[str] = frozenset(), dataset: str = "email") -> list[dict]:
    """A balanced random sample, or every email when per_class is None."""
    require_dataset(dataset)
    rng = random.Random(seed)
    files = list_files(dataset)
    items = []
    for label in ("ham", "spam"):
        pool = [file for file, file_label in files if file_label == label and file not in exclude]
        for file in pool if per_class is None else rng.sample(pool, per_class):
            bucket, parsed = read_email(file, dataset)
            items.append({"file": file, "label": label, "bucket": bucket, "email": parsed})
    rng.shuffle(items)
    return items


async def classify(client: AsyncTypeSafeClient, sem: asyncio.Semaphore, item: dict, questions: dict) -> dict:
    """One request per email; the stored row keeps scores and metadata but not the email text."""
    row = {key: item[key] for key in ("file", "label", "bucket")}
    async with sem:
        try:
            response = await client.system_one({"email": item["email"]}, questions)
        except Exception as error:  # noqa: BLE001 - record the failure so --resume can retry this email
            return {**row, "error": repr(error)}
    result = {
        **row,
        "nouls": {name: answer.noul for name, answer in response.nouls.items()},
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "model": response.model,
    }
    if response.choices:
        result["choices"] = {
            name: {"choice": answer.choice, "confidence": answer.confidence, "probabilities": answer.probabilities}
            for name, answer in response.choices.items()
        }
    return result


def auc(scores: list[float], y: list[bool]) -> float:
    """Mann-Whitney AUC with tied scores sharing their average rank."""
    order = sorted(range(len(scores)), key=scores.__getitem__)
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[start]]:
            end += 1
        for k in range(start, end + 1):
            ranks[order[k]] = (start + end) / 2 + 1
        start = end + 1
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    rank_sum = sum(r for r, t in zip(ranks, y) if t)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def logit(p: float) -> float:
    p = min(max(p, 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))


def hand_rule(n: dict) -> float:
    """Any clear spam signal, suppressed by evidence the email is expected correspondence."""
    spam_like = max(
        n["scam_or_phishing"],
        n["adult_content"],
        n["drugs_or_health_products"],
        n["filter_evasion"],
        n["commercial_offer"] * n["bulk_generic"],
    )
    legit = max(n["personal_or_work_thread"], n["subscribed_list_or_newsletter"])
    return spam_like * (1 - legit)


def cross_validated_logreg(ok: list[dict], features: list[str], y: list[bool]) -> tuple[list[float], dict]:
    """5-fold out-of-fold probabilities, plus coefficients from a fit on all rows."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    X = [[logit(r["nouls"][f]) for f in features] for r in ok]
    model = LogisticRegression(max_iter=1000)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    probs = cross_val_predict(model, X, y, cv=folds, method="predict_proba")[:, 1]
    model.fit(X, y)
    return list(probs), dict(zip(features, model.coef_[0]))


def predictors(ok: list[dict], question_set: str, y: list[bool]) -> dict[str, list[float]]:
    if question_set != "multi":
        return {q: [r["nouls"][q] for r in ok] for q in QUESTION_SETS[question_set]}
    signals = [q for q in QUESTION_SETS["multi"] if q != "spam_plain"]
    only_plain, _ = cross_validated_logreg(ok, ["spam_plain"], y)
    only_signals, _ = cross_validated_logreg(ok, signals, y)
    everything, coefs = cross_validated_logreg(ok, ["spam_plain", *signals], y)
    print("\nlogistic regression coefficients (all features, logit inputs, fit on all rows):")
    for name, coef in sorted(coefs.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {name:31s} {coef:+.2f}")
    return {
        "spam_plain": [r["nouls"]["spam_plain"] for r in ok],
        "hand_rule": [hand_rule(r["nouls"]) for r in ok],
        "logreg_cv[spam_plain]": only_plain,
        "logreg_cv[signals]": only_signals,
        "logreg_cv[all]": everything,
    }


def summary_line(name: str, p: list[float], y: list[bool]) -> list[bool]:
    pred = [v >= THRESHOLD for v in p]
    tp = sum(a and b for a, b in zip(y, pred))
    tn = sum(not a and not b for a, b in zip(y, pred))
    fp = sum(not a and b for a, b in zip(y, pred))
    fn = sum(a and not b for a, b in zip(y, pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"  {name:25s} acc {(tp + tn) / len(y):.3f}  prec {precision:.3f}  rec {recall:.3f}  "
          f"F1 {f1:.3f}  AUC {auc(p, y):.3f}  FN {fn:3d}  FP {fp:3d}")
    return pred


def subject_of(file: str, dataset: str) -> str:
    try:
        return read_email(file, dataset)[1]["subject"]
    except FileNotFoundError:
        return "(dataset not fetched)"


def details(name: str, ok: list[dict], p: list[float], y: list[bool], show_signals: list[str], dataset: str) -> None:
    pred = [v >= THRESHOLD for v in p]
    print(f"\n=== {name} ===")
    print("accuracy by format:")
    for bucket, count in sorted(Counter(r["bucket"] for r in ok).items()):
        idx = [i for i, r in enumerate(ok) if r["bucket"] == bucket]
        correct = sum(pred[i] == y[i] for i in idx)
        print(f"  {bucket:13s} n={count:4d} (spam {sum(y[i] for i in idx):3d})  acc {correct / count:.3f}")
    print("calibration (score bucket -> observed spam rate):")
    for lo in [i / 10 for i in range(10)]:
        idx = [i for i, v in enumerate(p) if lo <= v < lo + 0.1 or (lo == 0.9 and v == 1.0)]
        if idx:
            print(f"  [{lo:.1f}, {lo + 0.1:.1f})  n={len(idx):4d}  mean {sum(p[i] for i in idx) / len(idx):.2f}"
                  f"  spam rate {sum(y[i] for i in idx) / len(idx):.2f}")
    worst = sorted((i for i in range(len(ok)) if pred[i] != y[i]), key=lambda i: -abs(p[i] - THRESHOLD))[:10]
    if worst:
        print("most confident mistakes:")
        for i in worst:
            r = ok[i]
            signals = " ".join(f"{s[:6]}={r['nouls'][s]:.2f}" for s in show_signals)
            print(f"  {r['file']:12s} {r['label']:4s} score={p[i]:.2f}  {signals}  "
                  f"subject={subject_of(r['file'], dataset)[:40]!r}")


def report(results: list[dict], question_set: str, dataset: str = "email") -> None:
    ok = [r for r in results if "nouls" in r]
    errors = [r for r in results if "error" in r]
    print(f"\n{len(ok)} classified, {len(errors)} errors")
    for r in errors[:5]:
        print("  error:", r["file"], r["error"][:200])
    if not ok:
        return
    print(f"model: {ok[0]['model']}  tokens in/out: "
          f"{sum(r['input_tokens'] or 0 for r in ok)}/{sum(r['output_tokens'] or 0 for r in ok)}")
    y = [r["label"] == "spam" for r in ok]

    if question_set == "multi":
        print("\nper-question AUC for spam (ham-direction signals flipped):")
        for q in QUESTION_SETS["multi"]:
            scores = [r["nouls"][q] for r in ok]
            flipped = [1 - s for s in scores] if q in HAM_SIGNALS else scores
            mean_ham = sum(s for s, t in zip(scores, y) if not t) / (len(y) - sum(y))
            mean_spam = sum(s for s, t in zip(scores, y) if t) / sum(y)
            print(f"  {q:31s} AUC {auc(flipped, y):.3f}   mean yes: ham {mean_ham:.2f}  spam {mean_spam:.2f}")

    preds = predictors(ok, question_set, y)
    print(f"\nclassifiers (threshold {THRESHOLD}; logreg_cv = 5-fold out-of-fold probabilities):")
    for name, p in preds.items():
        summary_line(name, p, y)

    show = list(QUESTION_SETS[question_set]) if question_set == "multi" else []
    for name in (["spam_plain", "hand_rule", "logreg_cv[all]"] if question_set == "multi" else preds):
        details(name, ok, preds[name], y, [s for s in show if s != "spam_plain"], dataset)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, default="email")
    parser.add_argument("--questions", choices=QUESTION_SETS, default="single")
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--per-class", type=int, default=100)
    size.add_argument("--all", action="store_true", help="classify every email in the dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-from", nargs="*", default=[], type=Path)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report", action="store_true", help="report on an existing results file, no API calls")
    args = parser.parse_args()
    per_class = None if args.all else args.per_class

    size = "all" if per_class is None else f"{per_class}x2_seed{args.seed}"
    prefix = "results" if args.dataset == "email" else f"results_{args.dataset}"
    out = RESULTS_DIR / f"{prefix}_{args.questions}_{size}.jsonl"
    if args.report:
        if not out.exists():
            raise SystemExit(f"{out} does not exist")
        report([json.loads(line) for line in out.open()], args.questions, args.dataset)
        return

    exclude = {json.loads(line)["file"] for path in args.exclude_from for line in path.open()}
    questions = QUESTION_SETS[args.questions]
    items = sample(per_class, args.seed, exclude, args.dataset)
    RESULTS_DIR.mkdir(exist_ok=True)

    kept = {}
    if args.resume and out.exists():
        kept = {r["file"]: r for r in map(json.loads, out.open()) if "nouls" in r}
    todo = [item for item in items if item["file"] not in kept]
    labels = Counter(item["label"] for item in items)
    print(f"classifying {len(todo)} of {len(items)} emails ({labels['ham']} ham + {labels['spam']} spam) "
          f"with {len(questions)} questions per request; {len(kept)} kept from {out.name}")

    sem = asyncio.Semaphore(args.concurrency)
    start = time.monotonic()
    results = list(kept.values())
    with out.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
        async with AsyncTypeSafeClient(api_key=load_api_key()) as client:
            pending = [classify(client, sem, item, questions) for item in todo]
            for count, next_result in enumerate(asyncio.as_completed(pending), 1):
                r = await next_result
                results.append(r)
                f.write(json.dumps(r) + "\n")
                if count % 1000 == 0:
                    f.flush()
                    errors = sum("error" in r for r in results)
                    print(f"  {count}/{len(todo)} in {time.monotonic() - start:.0f}s, {errors} errors", flush=True)
    elapsed = time.monotonic() - start
    print(f"done in {elapsed:.1f}s ({len(todo)} requests, concurrency {args.concurrency}); wrote {out.name}")
    report(results, args.questions, args.dataset)


if __name__ == "__main__":
    asyncio.run(main())
