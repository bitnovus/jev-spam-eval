# Out-of-distribution test

A trained classifier learns the mail it was trained on. This test asks what happens when both
approaches meet email unlike that mail: a TF-IDF classifier trained on this repository's datasets,
and [TypeSafe](https://typesafe.ai)'s Jev model, which is never trained and was not adjusted for
these emails.

> **This is an exploratory experiment, not a benchmark.** The sets are small (633 modern emails and
> 853 recent phishing emails), run once with `jev-1.13.0` in September 2026. See
> [Caveats](#caveats).

## Results

| Test | What changes | TypeSafe | TF-IDF |
|---|---|---|---|
| **Ling-Spam**, spam or ham | A linguistics mailing list from 2000; TF-IDF trained on email-dataset | **98.6%** accuracy (plain question) | **73.0%** accuracy |
| **Recent phishing**, 2024–25 | Nearly 20 years newer; TF-IDF trained on 2005–07 mail | **91.0–93.6%** called phishing | **70.3%** called phishing |
| **Modern mail**, 2026, legitimate or not | Current mailing-list posts and spam-trap mail; TF-IDF trained on 2005–07 mail | **97.3%** accuracy (with urgency and authority) | **72.5%** accuracy |

Trained on its own labels, TF-IDF matched or beat TypeSafe on every dataset in this repository. On
unfamiliar email it fell 25 or more points behind, while TypeSafe stayed close to its usual accuracy.

## 1. Ling-Spam

The 2,876 unique Ling-Spam messages (468 spam), described in the
[README](README.md#second-dataset-ling-spam). TypeSafe's answers come from the Ling-Spam run. The
TF-IDF logistic regression was trained on all 18,514 unique email-dataset emails.

| Approach | Accuracy | Balanced accuracy | Missed spam (of 468) | Ham flagged as spam (of 2,408) |
|---|---|---|---|---|
| **TypeSafe, plain question** | **0.9857** | **0.9898** | 2 | 39 |
| TypeSafe, detailed criteria | 0.9701 | 0.9753 | 8 | 78 |
| TF-IDF trained on email-dataset | 0.7298 | 0.8111 | 32 | 745 |

TF-IDF flagged 745 legitimate list posts, almost a third, as spam: academic announcements and
discussions don't look like the business mail it learned "legitimate" from. Trained on Ling-Spam's
own labels, the same method scored 0.9857.

## 2. Recent phishing

The 853 unique phishing emails from the 2024 and 2025 mailboxes of
[Jose Nazario's phishing corpus](https://monkey.org/~jose/phishing/), described in
[PHISHING.md](PHISHING.md). TypeSafe's three-way answers come from the phishing run. The three-way
TF-IDF classifier was trained on the phishing experiment's main and fresh tests: 9,033 emails of
2005–07 phishing and email-dataset ham and spam.

| Approach | Called phishing | Called spam | Called legitimate |
|---|---|---|---|
| TypeSafe, descriptions | 91.3% | 1.1% | 7.6% |
| TypeSafe, with urgency and authority | 91.0% | 1.3% | 7.7% |
| TypeSafe, names only | 93.6% | 2.2% | 4.2% |
| TF-IDF trained on 2005–07 mail | 70.3% | 13.5% | 16.2% |

Modern phishing (storage-full notices, DocuSign requests, payment errors) uses different words from
2006's bank and eBay lures, so TF-IDF called 16% of it legitimate.

## 3. Modern mail

- **Legitimate:** all 333 unique posts to two public Python mailing lists from January to August
  2026: 264 discussion posts on python-list and 69 release and event announcements on
  python-announce-list.
- **Spam trap:** a random sample of 300 (seed 7) of the unique emails in the August 2026 file of
  Bruce Guenter's [spam archive](http://untroubled.org/spam/), which collects mail sent to bait
  addresses. It mixes spam and phishing without saying which is which, so this test scores
  legitimate against not legitimate.

TypeSafe was asked the three phishing-experiment questions, unchanged. The TF-IDF classifier is the
same one as in test 2.

| Approach | Legitimate posts called legitimate | Spam trap called legitimate | Accuracy | Balanced accuracy |
|---|---|---|---|---|
| TypeSafe, descriptions | 96.1% | 2.0% | 0.9700 | 0.9705 |
| **TypeSafe, with urgency and authority** | 96.4% | 1.7% | **0.9731** | **0.9736** |
| TypeSafe, names only | 98.2% | 1.0% | 0.9858 | 0.9860 |
| TF-IDF trained on 2005–07 mail | 97.3% | 55.0% | 0.7251 | 0.7115 |

TypeSafe called the spam-trap mail about 59% spam and 39–41% phishing.

- **TF-IDF passed most modern spam and phishing as legitimate.** It called 165 of the 300 spam-trap
  emails legitimate. They include "Still pending: Your AARP sign-up", "Your CVS Pts. are set to expire
  today", Walmart and Sam's Club points notices, "Courtesy Road Kit - AAA Licensed Drivers Only",
  auto-policy "changes to review" and health cures. TypeSafe caught 163 of those 165.
- **TypeSafe's mistakes on legitimate posts were mostly self-promotion.** 11 of the 12 list posts it
  called spam were one person's repeated posts promoting their own Python editor, which a list
  moderator might also call spam.
- **TF-IDF's mistakes on legitimate posts were announcements.** It called 7 posts phishing, such as
  "ANN: Python Meeting Düsseldorf" and "PyCA cryptography 48.0.1 release", and 2 spam.
- **Names only did best here** (98.6%). Its lack of detail suited modern mail better than
  descriptions written from 2000s email, though it can't tell spam from phishing
  ([PHISHING.md](PHISHING.md)).

Classifying the 633 modern emails cost about $0.05 (1.2M input tokens at $0.042 per million).

## Caveats

- **Small sets.** 633 modern emails and 853 recent phishing emails; a few emails change the
  percentages by about a point.
- **Spam-trap labels are assumed, not checked.** Mail to bait addresses is almost all unsolicited,
  but individual messages weren't reviewed.
- **The legitimate mail isn't varied.** It all comes from two Python mailing lists, with no receipts,
  shipping notices, newsletters or personal mail, which a real inbox would have.
- **The TF-IDF models are simple.** A model trained on newer or broader mail would do better; the
  point is that any trained model depends on its training data matching what it sees later.
- **Jev may have seen some of this mail.** `jev-1.13.0` was released in early September 2026, and
  the public archives used here could be in its training data. There was no way to check.

## Reproduce

```sh
uv sync
./fetch_dataset.sh && ./fetch_lingspam.sh && ./fetch_nazario.sh
./fetch_ood.sh                          # modern list posts and the August 2026 spam archive
uv run ood_test.py --report             # all three tests from saved results, no API calls
uv run ood_test.py --concurrency 16     # classify the modern mail again (calls the API)
```

The modern-mail answers are in `results/ood_modern.jsonl`, which refers to messages by file name
and position, not text. The report retrains both TF-IDF models, which takes about a minute.

## Credit

The spam-trap mail comes from Bruce Guenter's spam archive, http://untroubled.org/spam/, which
grants permission to use it without restriction and asks for a reference in published work. The
legitimate posts come from the public archives of
[python-list](https://mail.python.org/archives/list/python-list@python.org/) and
[python-announce-list](https://mail.python.org/archives/list/python-announce-list@python.org/).
Ling-Spam and the phishing corpus are credited in the [README](README.md#acknowledgments) and
[PHISHING.md](PHISHING.md#credit). This repository doesn't include messages from any of them.
