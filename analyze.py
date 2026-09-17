"""Full-dataset analysis of the criteria run: duplicates, subsets, calibration, cutoff, review band.

Reads results/results_criteria_all.jsonl and re-reads email text from the dataset to find exact
duplicates. The "never inspected" subset excludes the two 500-email samples whose mistakes were
read while writing the structured criteria.

Usage: uv run analyze.py
"""

import json
from collections import defaultdict

from spam_noul import RESULTS_DIR, THRESHOLD, auc, read_email, require_dataset

QUESTIONS = ["spam_plain", "spam_generic_criteria", "spam_structured_criteria", "spam_structured_focus"]
INSPECTED = ["results_multi_250x2_seed42.jsonl", "results_multi_250x2_seed1.jsonl"]


def summarize(name: str, rows: list[dict]) -> None:
    y = [r["label"] == "spam" for r in rows]
    print(f"\n{name}: n={len(rows)} (ham {len(y) - sum(y)}, spam {sum(y)})")
    for q in QUESTIONS:
        p = [r["nouls"][q] for r in rows]
        pred = [v >= THRESHOLD for v in p]
        tp = sum(a and b for a, b in zip(y, pred))
        tn = sum(not a and not b for a, b in zip(y, pred))
        fp = sum(not a and b for a, b in zip(y, pred))
        fn = sum(a and not b for a, b in zip(y, pred))
        balanced = (tp / (tp + fn) + tn / (tn + fp)) / 2
        print(f"  {q:26s} acc {(tp + tn) / len(y):.4f}  balanced {balanced:.4f}  prec {tp / (tp + fp):.3f}  "
              f"rec {tp / (tp + fn):.3f}  AUC {auc(p, y):.4f}  FN {fn:4d}  FP {fp:4d}")


def main() -> None:
    require_dataset()
    rows = sorted((json.loads(line) for line in (RESULTS_DIR / "results_criteria_all.jsonl").open()),
                  key=lambda r: r["file"])
    ok = [r for r in rows if "nouls" in r]
    print(f"{len(rows)} rows, {len(ok)} ok, {len(rows) - len(ok)} errors, model {ok[0]['model']}")
    print(f"tokens in/out: {sum(r['input_tokens'] for r in ok)}/{sum(r['output_tokens'] for r in ok)}")

    groups = defaultdict(list)
    for r in ok:
        e = read_email(r["file"])[1]
        r["chars"] = len(e["subject"]) + len(e["body"])
        groups[(e["subject"], e.get("from", ""), e["body"])].append(r)
    conflicts = [g for g in groups.values() if len({r["label"] for r in g}) > 1]
    print(f"unique email texts: {len(groups)}; duplicate groups: {sum(len(g) > 1 for g in groups.values())}; "
          f"texts labeled both ham and spam: {len(conflicts)}")

    inspected = {json.loads(line)["file"] for f in INSPECTED for line in (RESULTS_DIR / f).open()}
    dedup = [g[0] for g in groups.values() if len({r["label"] for r in g}) == 1]
    summarize("all files", ok)
    summarize("exact duplicates removed", dedup)
    summarize("dedup, non-empty (subject + body >= 20 chars)", [r for r in dedup if r["chars"] >= 20])
    summarize("dedup, never inspected while writing criteria", [r for r in dedup if r["file"] not in inspected])

    y = [r["label"] == "spam" for r in dedup]
    p = [r["nouls"]["spam_structured_criteria"] for r in dedup]
    print("\nstructured criteria, dedup: accuracy by format")
    for bucket in sorted({r["bucket"] for r in dedup}):
        idx = [i for i, r in enumerate(dedup) if r["bucket"] == bucket]
        acc = sum((p[i] >= THRESHOLD) == y[i] for i in idx) / len(idx)
        print(f"  {bucket:13s} n={len(idx):5d} spam {sum(y[i] for i in idx):5d}  acc {acc:.4f}")

    print("calibration (score bucket -> observed spam rate):")
    for lo in [i / 10 for i in range(10)]:
        idx = [i for i, v in enumerate(p) if lo <= v < lo + 0.1 or (lo == 0.9 and v == 1.0)]
        if idx:
            print(f"  [{lo:.1f},{lo + 0.1:.1f})  n={len(idx):5d}  mean {sum(p[i] for i in idx) / len(idx):.2f}  "
                  f"spam rate {sum(y[i] for i in idx) / len(idx):.3f}")

    best = max((sum((v >= t) == label for v, label in zip(p, y)) / len(y), t) for t in [i / 100 for i in range(30, 80)])
    print(f"best single cutoff chosen with the labels: {best[1]:.2f} -> accuracy {best[0]:.4f}")
    for lo, hi in ((0.3, 0.7), (0.2, 0.8)):
        inside = [i for i, v in enumerate(p) if lo <= v <= hi]
        rest = [i for i, v in enumerate(p) if not lo <= v <= hi]
        wrong = sum((p[i] >= THRESHOLD) != y[i] for i in rest)
        print(f"review band [{lo}, {hi}]: {len(inside)} emails ({len(inside) / len(p):.1%}) reviewed; "
              f"accuracy on the rest {1 - wrong / len(rest):.4f} ({wrong} errors)")


if __name__ == "__main__":
    main()
