# Matched-evidence comparison: Jev versus logistic regression

Both methods receive the same original text or the same enriched evidence. All reconstructed input hashes match the preceding Jev experiment. No new API calls.

## Evaluation design

- Main: five-fold StratifiedGroupKFold, near-duplicate clusters kept together. Both regression variants use the exact same folds.
- Fresh: each regression variant trained on the main set only. Also report the common subset excluding near-copies of main messages.
- Recent: each regression variant trained on main + fresh (9,033 older messages); 853 later phishing messages are evaluation only.
- Word TF-IDF: sublinear TF, min_df=2, max_features=200,000. Logistic regression: C=10, max_iter=2,000. Unchanged hyperparameters; no test-set tuning.
- Vectorizer vocabulary and IDF fit only on each training partition. Unsupervised grouping/exclusion follows the original protocol and uses original text.
- Enriched regression appends a JSON serialization of exactly the additional Jev fields to the original subject/from/body text. Same caps and metadata; no hand-built phishing indicators.
- Jev predictions are the preceding contemporaneous rerun. Standard defined Choice is primary; evidence-focused wording is secondary.
- Jev classification uses the returned Choice label, preserving API tie-breaking. Rounded Jev probabilities are renormalized before log loss and ensemble averaging. Regression and ensemble ties use class order ham, spam, phish.
- Ensembles are fixed 50/50 probability averages, without fitted weights or thresholds.
- Matched evidence is not matched supervision: regression uses corpus labels; Jev uses a pretrained model and task definitions. Pretraining exposure is unknown.

## main (n=5,733)

| Model | Accuracy | Macro F1 | Phishing recall | Phishing precision | Ham → phish | Spam → phish |
|---|---:|---:|---:|---:|---:|---:|
| Jev text | 93.62% | 0.9356 | 85.71% | 98.97% | 1 | 16 |
| LR text | 98.74% | 0.9874 | 98.90% | 99.63% | 2 | 5 |
| Jev enriched | 97.98% | 0.9798 | 98.43% | 99.16% | 1 | 15 |
| LR enriched | 98.87% | 0.9887 | 99.22% | 99.53% | 2 | 7 |
| Jev enriched + framing | 98.64% | 0.9864 | 98.38% | 99.00% | 1 | 18 |
| 50/50 text | 99.25% | 0.9925 | 98.80% | 99.79% | 0 | 4 |
| 50/50 enriched | 99.30% | 0.9930 | 99.06% | 99.74% | 1 | 4 |

- LR text -> LR enriched: 19 corrected, 12 regressed; 53 shared errors.
- Jev text -> Jev enriched: 256 corrected, 6 regressed; 110 shared errors.
- LR enriched -> Jev enriched: 45 corrected, 96 regressed; 20 shared errors.

Matrices: actual rows / predicted columns = ham, spam, phish.

Jev text
```text
 1844    66     1
   10  1885    16
  204    69  1638
```

LR text
```text
 1898    11     2
   33  1873     5
    7    14  1890
```

Jev enriched
```text
 1848    62     1
    8  1888    15
   16    14  1881
```

LR enriched
```text
 1897    12     2
   29  1875     7
    4    11  1896
```

Jev enriched + framing
```text
 1891    19     1
    9  1884    18
   18    13  1880
```

50/50 text
```text
 1904     7     0
    9  1898     4
    5    18  1888
```

50/50 enriched
```text
 1902     8     1
    9  1898     4
    6    12  1893
```

## fresh (n=3,300)

| Model | Accuracy | Macro F1 | Phishing recall | Phishing precision | Ham → phish | Spam → phish |
|---|---:|---:|---:|---:|---:|---:|
| Jev text | 92.67% | 0.9257 | 82.27% | 98.91% | 0 | 10 |
| LR text | 96.39% | 0.9638 | 91.64% | 99.41% | 0 | 6 |
| Jev enriched | 95.39% | 0.9540 | 89.36% | 99.09% | 0 | 9 |
| LR enriched | 96.33% | 0.9632 | 91.55% | 99.21% | 1 | 7 |
| Jev enriched + framing | 95.76% | 0.9575 | 89.55% | 98.80% | 0 | 12 |
| 50/50 text | 96.45% | 0.9645 | 90.55% | 99.40% | 0 | 6 |
| 50/50 enriched | 96.21% | 0.9620 | 89.73% | 99.40% | 0 | 6 |

- LR text -> LR enriched: 10 corrected, 12 regressed; 109 shared errors.
- Jev text -> Jev enriched: 93 corrected, 3 regressed; 149 shared errors.
- LR enriched -> Jev enriched: 29 corrected, 60 regressed; 92 shared errors.

Matrices: actual rows / predicted columns = ham, spam, phish.

Jev text
```text
 1066    34     0
    3  1087    10
   75   120   905
```

LR text
```text
 1091     9     0
   12  1082     6
   29    63  1008
```

Jev enriched
```text
 1075    25     0
    1  1090     9
   11   106   983
```

LR enriched
```text
 1088    11     1
    9  1084     7
   23    70  1007
```

Jev enriched + framing
```text
 1091     9     0
    4  1084    12
   11   104   985
```

50/50 text
```text
 1096     4     0
    3  1091     6
    5    99   996
```

50/50 enriched
```text
 1096     4     0
    2  1092     6
    4   109   987
```

## fresh without near-copies (n=2,398)

| Model | Accuracy | Macro F1 | Phishing recall | Phishing precision | Ham → phish | Spam → phish |
|---|---:|---:|---:|---:|---:|---:|
| Jev text | 91.78% | 0.9072 | 75.52% | 98.33% | 0 | 8 |
| LR text | 95.16% | 0.9463 | 85.44% | 99.26% | 0 | 4 |
| Jev enriched | 93.83% | 0.9304 | 81.60% | 98.65% | 0 | 7 |
| LR enriched | 95.08% | 0.9450 | 85.28% | 98.89% | 1 | 5 |
| Jev enriched + framing | 94.33% | 0.9345 | 81.92% | 98.08% | 0 | 10 |
| 50/50 text | 95.25% | 0.9445 | 83.52% | 99.24% | 0 | 4 |
| 50/50 enriched | 94.91% | 0.9403 | 82.08% | 99.23% | 0 | 4 |

- LR text -> LR enriched: 10 corrected, 12 regressed; 106 shared errors.
- Jev text -> Jev enriched: 52 corrected, 3 regressed; 145 shared errors.
- LR enriched -> Jev enriched: 29 corrected, 59 regressed; 89 shared errors.

Matrices: actual rows / predicted columns = ham, spam, phish.

Jev text
```text
  980    33     0
    3   749     8
   34   119   472
```

LR text
```text
 1004     9     0
   12   744     4
   29    62   534
```

Jev enriched
```text
  988    25     0
    1   752     7
   10   105   510
```

LR enriched
```text
 1001    11     1
    9   746     5
   23    69   533
```

Jev enriched + framing
```text
 1004     9     0
    4   746    10
   10   103   512
```

50/50 text
```text
 1009     4     0
    3   753     4
    5    98   522
```

50/50 enriched
```text
 1009     4     0
    2   754     4
    4   108   513
```

## recent (n=853)

| Model | Accuracy | Macro F1 | Phishing recall | Phishing precision | Ham → phish | Spam → phish |
|---|---:|---:|---:|---:|---:|---:|
| Jev text | — | — | 91.68% | — | — | — |
| LR text | — | — | 70.34% | — | — | — |
| Jev enriched | — | — | 95.31% | — | — | — |
| LR enriched | — | — | 75.26% | — | — | — |
| Jev enriched + framing | — | — | 94.49% | — | — | — |
| 50/50 text | — | — | 91.44% | — | — | — |
| 50/50 enriched | — | — | 95.66% | — | — | — |

- LR text -> LR enriched: 78 corrected, 36 regressed; 175 shared errors.
- Jev text -> Jev enriched: 31 corrected, 0 regressed; 40 shared errors.
- LR enriched -> Jev enriched: 197 corrected, 26 regressed; 14 shared errors.

Matrices: actual rows / predicted columns = ham, spam, phish.

Jev text
```text
    0     0     0
    0     0     0
   62     9   782
```

LR text
```text
    0     0     0
    0     0     0
  138   115   600
```

Jev enriched
```text
    0     0     0
    0     0     0
   32     8   813
```

LR enriched
```text
    0     0     0
    0     0     0
  140    71   642
```

Jev enriched + framing
```text
    0     0     0
    0     0     0
   39     8   806
```

50/50 text
```text
    0     0     0
    0     0     0
   51    22   780
```

50/50 enriched
```text
    0     0     0
    0     0     0
   22    15   816
```

## Limits and verification

- All 9,886 message pairs retained; all state hashes verified. Every main message held out exactly once, with no duplicate-group overlap between training and test.
- 902 fresh messages excluded in the shared near-copy sensitivity analysis.
- Grouping is fixed from original text for comparability; extra URL/domain campaign relationships may remain across folds.
- Original labels contain known noise and classes come from different sources. Previously inspected sets are exploratory, not a locked benchmark.
- Recent contains only phishing: no modern false-positive rate or precision conclusion is supported.
- Word tokenization may not capture every URL distinction. This tests the existing regression family with equal evidence; it is not a tuned ceiling for supervised methods.
- Local fit and analysis time: 51.3s. All regression fits converged; probabilities checked. No API spend.
