# Jev Choice with richer email evidence

Model: `jev-1.13.0`. Three variants on the same messages. Existing labels and category definitions preserved.

- Text: original input and original defined Choice, rerun contemporaneously.
- Enriched: same question plus Reply-To, link text/destinations/hostnames, and attachment names/types.
- Enriched + framing: enriched input plus evidence-focused instructions; same class definitions.
- No links were visited. No rule-based phishing scores, examples, fitting, or threshold changes.
- Body remains truncated at 6,000 characters. Links are extracted from full textual MIME parts; limits: 80 links, roughly 24,000 serialized link characters, 2,000 characters per destination, 500 per anchor text, 40 attachment records.
- Predictions are exploratory on previously inspected datasets, not an untouched benchmark.

## main: 5733/5733 paired messages

| Variant | Accuracy | Macro F1 | Phishing recall | Phishing precision | Ham → phish | Spam → phish |
|---|---:|---:|---:|---:|---:|---:|
| Text | 93.62% | 0.9356 | 85.71% | 98.97% | 1 | 16 |
| Enriched | 97.98% | 0.9798 | 98.43% | 99.16% | 1 | 15 |
| Enriched + framing | 98.64% | 0.9864 | 98.38% | 99.00% | 1 | 18 |

Confusion matrices below use actual rows and predicted columns in order **ham, spam, phish**.

Text:
```text
 1844    66     1
   10  1885    16
  204    69  1638
```

Enriched:
```text
 1848    62     1
    8  1888    15
   16    14  1881
```

Enriched + framing:
```text
 1891    19     1
    9  1884    18
   18    13  1880
```

- Enriched versus text: **256 errors corrected, 6 correct predictions lost**; exact paired McNemar p=1.17e-67 (message-level; campaign dependence limits interpretation).
- Enriched + framing versus text: **291 errors corrected, 3 correct predictions lost**; exact paired McNemar p=2.66e-82 (message-level; campaign dependence limits interpretation).
- Original saved Choice versus contemporaneous text rerun: 34 labels changed.

- eBay-subject phishing called ham by text: 102; recovered as phishing by enriched: 100; by enriched + framing: 100.

## fresh: 3300/3300 paired messages

| Variant | Accuracy | Macro F1 | Phishing recall | Phishing precision | Ham → phish | Spam → phish |
|---|---:|---:|---:|---:|---:|---:|
| Text | 92.67% | 0.9257 | 82.27% | 98.91% | 0 | 10 |
| Enriched | 95.39% | 0.9540 | 89.36% | 99.09% | 0 | 9 |
| Enriched + framing | 95.76% | 0.9575 | 89.55% | 98.80% | 0 | 12 |

Confusion matrices below use actual rows and predicted columns in order **ham, spam, phish**.

Text:
```text
 1066    34     0
    3  1087    10
   75   120   905
```

Enriched:
```text
 1075    25     0
    1  1090     9
   11   106   983
```

Enriched + framing:
```text
 1091     9     0
    4  1084    12
   11   104   985
```

- Enriched versus text: **93 errors corrected, 3 correct predictions lost**; exact paired McNemar p=3.72e-24 (message-level; campaign dependence limits interpretation).
- Enriched + framing versus text: **105 errors corrected, 3 correct predictions lost**; exact paired McNemar p=1.29e-27 (message-level; campaign dependence limits interpretation).
- Original saved Choice versus contemporaneous text rerun: 11 labels changed.

- eBay-subject phishing called ham by text: 54; recovered as phishing by enriched: 48; by enriched + framing: 47.

## recent: 853/853 paired messages

| Variant | Accuracy | Macro F1 | Phishing recall | Phishing precision | Ham → phish | Spam → phish |
|---|---:|---:|---:|---:|---:|---:|
| Text | — | — | 91.68% | — | — | — |
| Enriched | — | — | 95.31% | — | — | — |
| Enriched + framing | — | — | 94.49% | — | — | — |

Confusion matrices below use actual rows and predicted columns in order **ham, spam, phish**.

Text:
```text
    0     0     0
    0     0     0
   62     9   782
```

Enriched:
```text
    0     0     0
    0     0     0
   32     8   813
```

Enriched + framing:
```text
    0     0     0
    0     0     0
   39     8   806
```

- Enriched versus text: **31 errors corrected, 0 correct predictions lost**; exact paired McNemar p=9.31e-10 (message-level; campaign dependence limits interpretation).
- Enriched + framing versus text: **29 errors corrected, 5 correct predictions lost**; exact paired McNemar p=3.86e-05 (message-level; campaign dependence limits interpretation).
- Original saved Choice versus contemporaneous text rerun: 5 labels changed.

## Coverage and usage

- Successful unique requests: 19772 / 19772; recorded failed attempts: 0.
- Input tokens: 32,060,311; output tokens: 1,305,556.
- Historical-rate cost estimate: $1.347, using the previous experiment's $0.042/million input tokens. This is not a verified current bill; three-message pilot excluded.
- Enriched messages with links: 8007; with attachments: 565; with omitted links: 21.
- Recent set contains phishing only, so it measures recall, not false-positive rate or useful precision.
- Existing corpus labels include known spam/phishing ambiguity. Source and campaign artifacts remain.
- This experiment bundles several added metadata fields; a gain cannot be attributed to URLs alone.
