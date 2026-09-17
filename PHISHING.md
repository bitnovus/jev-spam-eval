# Ham, spam or phishing

An experiment: can [TypeSafe](https://typesafe.ai)'s Jev model sort email into three categories,
legitimate (ham), spam and phishing, using one multiple-choice question and no training data? Its
answers are compared with a classifier that counts word frequencies (TF-IDF) and learns from labeled
examples. This follows the spam-or-ham experiment in the [README](README.md) and reads email the
same way.

> **This is an exploratory experiment, not a benchmark.** It was run once, with one model version
> (`jev-1.13.0`, September 2026). One of the three questions was reworded after reading the main
> test's mistakes, so a fresh set of emails was used to check it. See [Caveats](#caveats).

## Data

- **Ham and spam** come from
  [email-dataset](https://github.com/realprogrammersusevim/email-dataset). Only emails that still
  have their full headers are used, so the shape of a message doesn't reveal its category. After
  removing exact duplicates, 4,110 ham and 5,161 spam emails were available.
- **Phishing** comes from [Jose Nazario's phishing corpus](https://monkey.org/~jose/phishing/), a
  hand-sorted collection published under CC BY 4.0, one mailbox per period.

Three test sets were built from them:

| Set | Phishing | Ham and spam | Size |
|---|---|---|---|
| **Main test** | All 1,911 unique messages in `phishing3.mbox` (mostly 2005–07) | Random samples of 1,911 each (seed 7) | 5,733 |
| **Fresh test** | All 1,100 unique messages in `phishing2.mbox` (mostly 2006) that don't repeat a main-pool text | Random samples of 1,100 each from emails not in the main test | 3,300 |
| **Recent phishing** | All 853 unique messages in the 2024 and 2025 mailboxes | None: there is no ham or spam from those years | 853 |

The main and fresh phishing emails come from much the same period, mostly 2005–06, so "fresh" means
different messages, not later ones. 902 of the fresh emails are near-copies of main-test emails
(80% or more similar): 475 phishing, 340 spam and 87 ham.

## The questions

Each message is sent once with three [Choice](https://docs.typesafe.ai/primitives/choice)
questions, all asking "Is `email` legitimate, spam, or phishing?". A Choice question returns the
chosen category, a probability for each category, and a confidence score.

- **Descriptions:** each category has a written definition, fixed before any results. Phishing is
  email that pretends to come from a real bank, company, service or colleague to get passwords,
  account or payment details, or personal information, or to get the recipient to open a malicious
  link or attachment. Advance-fee, lottery and inheritance scams from strangers count as spam.
  Legitimate uses the same definition as the spam-or-ham experiment.
- **Descriptions with urgency and authority:** the same, plus two tactics phishing usually relies on.
  *Authority*: it pretends to be someone the reader is expected to trust or obey, such as a bank, a
  well-known company, their IT department, a manager, or a government agency, often with
  official-looking names, logos, reference numbers or security wording. *Urgency*: it pushes the
  reader to act at once or lose something, such as a suspended account, unauthorized activity, a
  failed payment, an unanswered buyer question, a full mailbox, a tax penalty, or a short deadline.
  This wording was added after reading the mistakes in the main test.
- **Names only:** the three categories with no descriptions.

The full text of each question is in `phish.py`.

## How TF-IDF was trained and tested

TypeSafe is never trained, so there is nothing to split. The TF-IDF classifier is trained, so it
needs a split:

- **Main test:** the 5,733 emails were divided into five groups. Five times over, the classifier
  trained on four groups and scored the fifth, so every email was scored by a model that had not
  seen it. Near-copies stayed in the same group, so the variants of one phishing campaign never sat
  on both sides.
- **Fresh test:** the classifier trained once on all 5,733 main-test emails, then scored the 3,300
  fresh ones.
- **Averages** below are a plain 50/50 average of the two sets of probabilities. Nothing was fitted
  to the test labels.

## Results

### Main test (5,733 emails)

| Approach | Accuracy | Phishing caught | "Phishing" answers that were phishing | Spam called phishing |
|---|---|---|---|---|
| TypeSafe, descriptions | 0.935 | 85.5% | 99.0% | 16 |
| TypeSafe, with urgency and authority | 0.942 | 87.8% | 98.4% | 26 |
| TypeSafe, names only | 0.731 | 88.6% | 60.7% | 1,095 |
| **TF-IDF (cross-validated)** | **0.987** | 98.9% | 99.6% | 5 |
| Average of either description question and TF-IDF | 0.993 | 98.8–98.9% | 99.7–99.8% | 4–5 |

### Fresh test (3,300 emails)

| Approach | Accuracy | Phishing caught | "Phishing" answers that were phishing | Without near-copies (2,398) |
|---|---|---|---|---|
| TypeSafe, descriptions | 0.926 | 82.2% | 98.9% | 0.917 |
| TypeSafe, with urgency and authority | 0.929 | 83.0% | 98.7% | 0.920 |
| TypeSafe, names only | 0.722 | 84.1% | 59.3% | 0.719 |
| **TF-IDF (trained on the main test)** | **0.964** | 91.6% | 99.4% | **0.952** |
| Average of either description question and TF-IDF | 0.964–0.965 | 90.5–90.6% | 99.4% | 0.952–0.953 |

TypeSafe with urgency and authority, fresh test (rows are the true category):

| | Called legitimate | Called spam | Called phishing |
|---|---|---|---|
| Legitimate (1,100) | 1,066 | 34 | 0 |
| Spam (1,100) | 3 | 1,085 | 12 |
| Phishing (1,100) | 71 | 116 | 913 |

### Recent phishing (853 emails, all phishing)

| Approach | Called phishing | Called spam | Called legitimate |
|---|---|---|---|
| TypeSafe, descriptions | 91.3% | 1.1% | 7.6% |
| TypeSafe, with urgency and authority | 91.0% | 1.3% | 7.7% |
| TypeSafe, names only | 93.6% | 2.2% | 4.2% |
| TF-IDF trained on the main and fresh tests | 70.3% | 13.5% | 16.2% |

The TF-IDF row comes from the [out-of-distribution test](OUT_OF_DISTRIBUTION.md).

### What the results show

- **Descriptions matter even more with three categories.** With names only, TypeSafe called 1,095
  of the 1,911 main-test spam emails phishing: without a definition, scams count as phishing. With
  descriptions, it did that for 16.
- **When TypeSafe says phishing, it's almost always right, but it misses some.** About 99% of its
  "phishing" answers were right, but it missed 12–18% of phishing, mostly by calling it legitimate.
- **Adding urgency and authority helped on the emails that prompted it, and barely on fresh ones.**
  On the main test it moved 45 phishing emails into phishing (25 of them had been called legitimate,
  20 spam), and wrongly moved 10 spam emails there too, taking accuracy from 0.935 to 0.942. On the
  fresh test it fixed 9 phishing emails and moved 2 spam emails, from 0.926 to 0.929. That is about
  as much as the answers shift when the same question is asked twice. The recent-phishing results
  didn't change at all.
- **Most missed phishing imitates eBay.** In the main test, 167 of the 185 phishing emails called
  legitimate (with urgency and authority) are fake eBay "question from a member" or listing notices;
  in the fresh test, 66 of 71. Their text copies real eBay mail, and about half their links really do
  go to eBay pages. 133 of the 167 also carry at least one link to an unrelated site, such as a raw
  IP address, but TypeSafe never sees that: the parsing keeps the words a link shows and drops the
  address behind it.
- **Part of TF-IDF's lead comes from how the data was assembled.** 244 main-test phishing emails
  have eBay in the subject, but only 1 legitimate email mentions eBay (a news item). So "sounds like
  eBay" means phishing in this data. With real eBay notifications in the mix, that shortcut would
  flag them too.
- **Many fresh "mistakes" are mislabeled spam.** TypeSafe called 116 fresh phishing emails spam. In
  a random sample of 20, 13 were clearly ordinary spam or advance-fee scams (pharmacy, casino, loan
  and credit offers, a job scam), 5 were garbled or unreadable, and 2 were bank phishing.
- **Confidence helps, but TF-IDF stays ahead however much is reviewed.** Have a person check the
  answers TypeSafe is least sure of — 5%, 10%, 20% or 34% of them — and the rest of the main test is
  96.0%, 97.0%, 98.7% and 99.6% correct. Checking the same share of TF-IDF's least certain answers
  leaves 99.7%, 99.8%, 99.9% and 99.9%.
- **Recent phishing is where TypeSafe does better.** On 2024–25 phishing, TypeSafe called 91–94%
  phishing, while TF-IDF trained on 2005–07 mail called 70%. See
  [OUT_OF_DISTRIBUTION.md](OUT_OF_DISTRIBUTION.md).

The run cost about $0.76 at $0.042 per million input tokens: 10.6M input tokens for the main test,
6.0M for the fresh test and 1.5M for recent phishing, with all three questions in each request. All
9,886 requests took about three minutes at 16 at a time.

## Caveats

- **Some labels are wrong in both directions.** The phishing corpus is hand-sorted from one inbox and
  includes ordinary spam and mailing-list posts (for example a botnet discussion list). email-dataset's
  spam includes some phishing, such as a "U.S. Bank fraud verification" email and fake IRS refund
  notices.
- **The categories come from different collections.** Phishing comes from one source, ham and spam
  from another, so a difference between categories might really be a difference between sources. I
  blanked out the words that give the phishing source away, its owner's name and its placeholder
  addresses, and retrained: TF-IDF still scored 98.8% on the main test. So it isn't leaning on those,
  but subtler differences can't be ruled out.
- **The model never saw where links point.** Passing each link's address and the sender's domain to
  TypeSafe might catch many of the fake eBay notices.
- **One model version, one run.** Asking the descriptions question a second time on the main test
  gave the same answer for 99.3% of the emails.
- **Jev may have seen these emails before.** The phishing corpus has been public since 2005, and
  email-dataset draws on older public collections. There was no way to check.

## Reproduce

```sh
uv sync
./fetch_dataset.sh
./fetch_nazario.sh          # downloads the phishing mailboxes and checks their checksums
uv run phish.py --report                      # rebuild the report from saved results, no API calls
uv run phish.py --all --concurrency 16        # run again (calls the API); add --resume to retry failures
```

`--report` retrains the TF-IDF models, which takes about a minute. Results are in
`results/phish_main_all.jsonl`, `results/phish_fresh.jsonl` and `results/phish_recent.jsonl`. They
hold each message's answers, not its text: phishing messages are referred to by mailbox name and
position, such as `phishing3.mbox#12`.

## Credit

The phishing messages come from Jose Nazario's phishing corpus, https://monkey.org/~jose/phishing/,
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Thanks to Jose Nazario for
collecting and publishing it. This repository doesn't include messages from it.
