# Sorting email with TypeSafe questions

Experiments on whether [TypeSafe](https://typesafe.ai)'s Jev model can sort email as well as a
trained classifier, using questions in plain English and no training data. Jev is asked whether an
email is spam, or which of legitimate, spam and phishing it is, and its answers are compared with
word-frequency (TF-IDF) classifiers trained on each dataset's own labels. Four public email
collections are used, spanning 2000 to 2026.

**The main finding: a trained classifier wins on the mail it was trained on, and loses badly on
anything else.** On each dataset's own labels, TF-IDF matched or beat TypeSafe. On email from a
different source or a different era, it fell 25 points or more behind, while TypeSafe scored about
what it always does. Details in the [out-of-distribution test](OUT_OF_DISTRIBUTION.md).

| Email unlike the classifier's training mail | TypeSafe (no training) | TF-IDF |
|---|---|---|
| Ling-Spam, a 2000 linguistics list: spam or legitimate | **98.6%** | 73.0% |
| Phishing from 2024–25: share called phishing | **91.0–93.6%** | 70.3% |
| 2026 list posts and spam-trap mail: legitimate or not | **97.3%** | 72.5% |

The rest of this page covers the first experiment, spam or ham on
[email-dataset](https://github.com/realprogrammersusevim/email-dataset) and
[Ling-Spam](#second-dataset-ling-spam). Sorting mail into three categories, including phishing, is
in [PHISHING.md](PHISHING.md).

Cost is part of the motivation. Jev costs $0.042 per million input tokens ($42 per billion, as
listed on [typesafe.ai](https://typesafe.ai)), and output tokens are free
([The Rundown](https://www.therundown.ai/news/typesafe-jev-ai-decisions-software)). At that price,
running a model on every email is cheap: checking all 19,528 emails, with four questions per email,
cost about $1.12.

> **These are exploratory experiments, not benchmarks.** Each was run once, with one model version
> (`jev-1.13.0`, September 2026). The questions changed as the work went on, and the
> best-performing wording was written after reading the first dataset's mistakes; on the second
> dataset it did worse than the plain question. The out-of-distribution sets are small: 2,876
> Ling-Spam messages, 853 phishing emails and 633 modern ones. Treat the numbers as a record of what
> happened here, not a general measure of spam-filter accuracy. See [Caveats](#caveats) and the
> caveats in each write-up.

## Spam or ham

Each email is sent to the TypeSafe API with a yes/no question, "Is `email` spam?", and the API
returns the probability that the answer is yes. TypeSafe calls this kind of question a
[Noul](https://docs.typesafe.ai/primitives/noul). The biggest improvement came from defining spam
precisely: listing which kinds of mail count as spam and which don't raised accuracy from 96.0% to
98.3%. That matches a logistic regression classifier trained on about 14,800 labeled emails.

## Results

These results cover the 18,514 unique emails (exact duplicates removed). An email counts as spam
when its score is 0.5 or higher. Each TF-IDF score comes from a model that never saw that email, or
any near-copy of it, during training: 5-fold cross-validation, with near-copies always kept in the
same fold.

| Approach | Labeled emails used | Accuracy | AUC | Missed spam | Ham flagged as spam |
|---|---|---|---|---|---|
| TypeSafe, plain question | 0 | 0.9596 | 0.9975 | 54 | 694 |
| **TypeSafe, detailed criteria** | **0** | **0.9833** | 0.9984 | 106 | 203 |
| **TF-IDF logistic regression** | **~14.8K** | **0.9839** | 0.9986 | 149 | 150 |
| TF-IDF naive Bayes | ~14.8K | 0.9707 | 0.9971 | 367 | 175 |
| Average of detailed criteria and logistic regression | ~14.8K | 0.9922 | 0.9995 | 69 | 75 |
| Logistic regression, with near-copies of test emails allowed in training | ~14.8K | 0.9890 | 0.9992 | 103 | 100 |

AUC measures how well the scores rank spam above ham, where 1.0 is perfect.

- **TypeSafe with detailed criteria matches the trained classifier.** The two disagree on 466
  emails: TypeSafe is right on 228 of them and logistic regression on 238. That difference isn't
  statistically significant (McNemar test, p = 0.68).
- **They make different mistakes.** Only 71 emails fool both, so averaging their two scores cuts
  errors roughly in half. TypeSafe catches spam that uses words the trained model never learned,
  such as disguised drug names, random filler words and non-English spam. Many of logistic
  regression's wins are news-feed items and vendor announcements; it learned that this dataset
  labels those sources as ham.
- **Near-copies make trained models look better than they are.** When near-copies of the test
  emails were allowed into training, logistic regression gained half a point.
- **Very high and very low scores are reliable.** Of the 8,333 emails scored below 0.1, 0.1% are
  spam. Of the 7,016 scored 0.9 or higher, 99.9% are. Scores in the middle overstate spam a
  little: emails scored 0.5–0.6 are only 38% spam. Moving the cutoff to 0.56 (chosen using the
  labels) gives 0.9863.
- **Sending uncertain emails to a person helps.** Reviewing the emails scored 0.3–0.7 (855 emails,
  4.6%) leaves 99.50% accuracy on the rest; reviewing 0.2–0.8 (8.1%) leaves 99.76%. Reviewing the
  same 4.6% share gives 99.58% for logistic regression and 99.87% for the average of both.
- **It's cheap to run.** The full run used about 1,370 input tokens per email with four questions.
  At $0.042 per million input tokens, that is $0.058 per 1,000 emails, or about $58 per million.
  All the runs in `results/` together used 30.4M input tokens, about $1.28.

## The question that worked best

```python
Noul(instructions="Is `email` spam?", criteria={
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
})
```

In the code this is the `spam_structured_criteria` question. Each email is sent as
`{"email": {"subject", "from", "body"}}`. Only the text parts of each email are kept; they are
decoded, HTML tags are removed, and the body is cut off at 6,000 characters. Only emails that
include full headers have a `from` field.

## How the experiment went

**1. One question, two wordings.** On 500 emails (250 ham, 250 spam, seed 42), the plain question
scored 0.980 and a version with a short, general definition of spam scored 0.978. Most mistakes
were newsletters and mailing-list messages.

**2. Nine narrower questions.** The spam question was broken into separate yes/no questions, all
sent in one request: scam or phishing, sales pitch, adult content, drugs, mass mailing, tricks to
get past filters, personal or work email, subscribed newsletter, plus the plain question. Code then
combined the answers.

| How the answers were combined (seed 42 / seed 1) | Accuracy |
|---|---|
| Plain question alone | 0.978 / 0.962 |
| A hand-written rule | 0.974 / 0.952 |
| A small trained model using only the plain question | 0.980 / 0.972 |
| A small trained model using the eight narrower questions | 0.978 / 0.968 |
| A small trained model using all nine | 0.982 / 0.972 |

The small trained models were scored with cross-validation. The extra questions helped explain the
mistakes, but they didn't improve accuracy more than moving the cutoff did. Nine questions took as
long as two, with 33% more input tokens and 4.3 times as many output tokens.

**3. Detailed criteria.** These were written after reading the mistakes in the seed 42 and seed 1
samples, then tested on a fresh sample (seed 3) that shares no emails with those two.

| Question | Seed 42 | Seed 1 | Seed 3 (fresh sample) | Seed 3 ham flagged as spam |
|---|---|---|---|---|
| Plain question | 0.980 | 0.964 | 0.960 | 18 |
| Short general definition | 0.978 | 0.962 | 0.964 | 14 |
| Detailed criteria | 0.986 | 0.972 | **0.986** | **2** |
| Detailed criteria with an extra instruction | 0.986 | 0.972 | 0.980 | 4 |

The extra instruction told the model to judge whether the recipient asked for or expects this kind
of email, not whether it is commercial or sent to many people. It didn't help.

**4. Full dataset.** All 19,528 emails, with all four questions sent in one request per email.

| Question | All files | Unique emails not read while writing the criteria (17,559) |
|---|---|---|
| Plain question | 0.9596 | 0.9590 |
| Short general definition | 0.9663 | 0.9663 |
| Detailed criteria | 0.9829 | 0.9836 |
| Detailed criteria with an extra instruction | 0.9795 | 0.9801 |

## Second dataset: Ling-Spam

[Ling-Spam](https://www.aueb.gr/users/ion/data/lingspam_public.tar.gz) was created by Ion
Androutsopoulos and colleagues (2000) and is used here with thanks; see
[Acknowledgments](#acknowledgments). It has 2,893 messages: 2,412 legitimate posts from the Linguist mailing list and 481 spam
messages. This test uses its "bare" version, which keeps every word. The text is already lowercased,
and each message has only a subject line and a body. None of its messages appear
in email-dataset, and only 2 are 90% or more similar to a message there. As with the first dataset,
bodies are cut off at 6,000 characters, which affects 419 messages.

The four questions from the full run were used unchanged, including the detailed criteria written
for email-dataset. All 2,893 messages took 31 seconds and 4.5M input tokens, about $0.19.

| Question (all 2,893 messages) | Accuracy | Missed spam (of 481) | Ham flagged as spam (of 2,412) |
|---|---|---|---|
| **Plain question** | **0.986** | 2 | **39** |
| Short general definition | 0.921 | 2 | 227 |
| Detailed criteria | 0.970 | 8 | 78 |
| Detailed criteria with an extra instruction | 0.962 | 8 | 101 |

The comparison with trained classifiers uses the 2,876 unique messages and the same cross-validation
as the first dataset, with near-copies kept in the same fold. Balanced accuracy averages the share of
spam caught and the share of ham passed, so it isn't dominated by the much larger ham class.

| Approach | Labeled messages used | Accuracy | Balanced accuracy | Missed spam (of 468) | Ham flagged as spam (of 2,408) |
|---|---|---|---|---|---|
| **TypeSafe, plain question** | **0** | 0.9857 | **0.9898** | 2 | 39 |
| TypeSafe, detailed criteria | 0 | 0.9701 | 0.9753 | 8 | 78 |
| TF-IDF logistic regression | ~2.3K | 0.9857 | 0.9571 | 40 | 1 |
| TF-IDF naive Bayes | ~2.3K | **0.9941** | 0.9844 | 14 | 3 |
| Average of TypeSafe (either question) and logistic regression | ~2.3K | 0.9983 | 0.9955 | 4 | 1 |

- **The detailed criteria didn't carry over.** On Ling-Spam the plain question did best, and every
  added definition increased the number of legitimate messages flagged as spam. About three-quarters
  of the 78 flagged by the detailed criteria are publishers' notices of new books posted to the list;
  most of the rest are course, job and association announcements. A likely reason is that the
  criteria count "advertising or offers" as spam, while Ling-Spam messages have no headers or list
  footers to show they came through a list the reader joined.
- **TypeSafe and the trained classifiers err in opposite directions.** TypeSafe rarely misses spam
  but flags some list announcements; the trained classifiers almost never flag legitimate messages
  but miss more spam. With only about 2,300 training messages, logistic regression missed 40 spam
  messages, including diploma offers, adult sites and diet pills that TypeSafe caught.
- **By plain accuracy naive Bayes wins; by balanced accuracy the plain question does.** Ling-Spam is
  known to be an easy corpus for word-based filters, because all its legitimate mail comes from one
  mailing list about linguistics.
- **Combining still helps most.** Averaging TypeSafe's score with logistic regression's gives 0.9983
  accuracy, with 4 missed spam messages and 1 false flag, whichever TypeSafe question is used.
- **The corpus's own 10 parts flatter the trained models.** 106 groups of near-copies are spread
  across more than one part. Using those parts as the folds, as the original paper did, raises
  logistic regression from 0.9857 to 0.9910 and naive Bayes from 0.9941 to 0.9962.

## Further experiments

**[Ham, spam or phishing](PHISHING.md).** One multiple-choice question sorts email into three
categories, with the phishing coming from Jose Nazario's collection. On 5,733 emails, equally split
between the three, TypeSafe scored 0.94 and a TF-IDF classifier trained on 4,600 labeled examples
scored 0.99. Writing out what each category means mattered most: given only the category names,
TypeSafe called more than half the spam phishing. Describing how phishing appeals to urgency and
authority helped on the emails that prompted the change, and barely on a fresh set.

**[Out-of-distribution test](OUT_OF_DISTRIBUTION.md).** The main finding above, in full: both
approaches shown email unlike anything the classifiers had learned from, including 2026 mailing-list
posts and spam-trap mail collected weeks before the run. It also records what each got wrong. The
classifier passed 165 of 300 spam-trap emails as legitimate, among them AARP sign-ups and CVS points
notices; TypeSafe caught 163 of those. TypeSafe's own false flags were mostly one person
repeatedly promoting their software on a mailing list.

## Caveats

- **Some labels are wrong.** The dataset treats newsletters and promotions inconsistently, and some
  of TypeSafe's most confident "mistakes" are labeling errors. For example, an Irish-lottery scam is
  labeled ham (`1/02595.eml`), and two ordinary personal emails are labeled spam (`2/5823.eml`,
  `2/0402.eml`).
- **The criteria fit the first dataset.** They were written from its mistakes, and they cause some
  errors of their own. For example, they treat bounce messages as delivery notices, but the dataset
  labels some bounce messages as spam. On Ling-Spam they made results worse. For other mail, write
  your own definition and test it.
- **Jev may have seen these emails before.** The first dataset appears to combine the public Enron,
  SpamAssassin and CSDMC2010 collections, and Ling-Spam has been public since 2000. Any of them may
  be in Jev's training data. There was no way to check.
- **One model version, one run.** Results are from `jev-1.13.0` (the version behind `jev-latest`)
  in September 2026. Scores close to 0.5 can change slightly between runs: the same 500-email
  sample scored 0.980 and then 0.978 with the plain question. Costs use the price listed in
  September 2026.
- **Duplicates and empty files.** The dataset contains 815 sets of exact duplicates (none labeled
  both ways) and about 100 nearly empty files. The main tables leave out exact duplicates.

## Reproduce

You need [uv](https://docs.astral.sh/uv/). New classification runs also need a TypeSafe API key.

```sh
uv sync
./fetch_dataset.sh          # downloads email-dataset at the version used here
cp .env.example .env        # then add your TYPESAFE_API_KEY
```

Rebuild the reports from the saved results, without calling the API:

```sh
uv run spam_noul.py --questions criteria --all --report
uv run analyze.py           # duplicates, subsets, score reliability, review rates
uv run tfidf_baseline.py    # trains the TF-IDF models (about a minute), writes results/tfidf_oof.jsonl
```

For Ling-Spam:

```sh
./fetch_lingspam.sh         # downloads Ling-Spam and checks it matches the copy used here
uv run spam_noul.py --dataset lingspam --questions criteria --all --report
uv run tfidf_baseline.py --dataset lingspam    # writes results/lingspam_tfidf_oof.jsonl
```

`report/confusion_matrices.ipynb` draws the confusion matrices for both datasets from
`results/tfidf_oof.jsonl` and `results/lingspam_tfidf_oof.jsonl`, and lets you try a different
cutoff. GitHub shows it with its charts; to run it again:

```sh
uv run jupyter nbconvert --to notebook --execute --inplace report/confusion_matrices.ipynb
```

Run the classification again. This calls the API and overwrites the matching file in `results/`:

```sh
uv run spam_noul.py --questions single --per-class 250 --seed 42
uv run spam_noul.py --questions multi --per-class 250 --seed 42
uv run spam_noul.py --questions multi --per-class 250 --seed 1
uv run spam_noul.py --questions criteria --per-class 250 --seed 42
uv run spam_noul.py --questions criteria --per-class 250 --seed 1
uv run spam_noul.py --questions criteria --per-class 250 --seed 3 \
  --exclude-from results/results_multi_250x2_seed42.jsonl results/results_multi_250x2_seed1.jsonl
uv run spam_noul.py --questions criteria --all --concurrency 16    # add --resume to retry failures
uv run spam_noul.py --dataset lingspam --questions criteria --all --concurrency 16
```

A 500-email sample takes about 10 seconds and costs about $0.03. The full run took 206 seconds with
16 requests at a time and used 26.7M input and 1.6M output tokens with four questions per email.
That cost about $1.12, since only input tokens are billed. Asking a single question per email would
use fewer tokens.

## Repository layout

| Path | Contents |
|---|---|
| `spam_noul.py` | Reads the messages from either dataset, picks samples, defines the three question sets, calls the API and prints reports |
| `tfidf_baseline.py` | Trains the TF-IDF classifiers, keeping near-copies together, and compares them with TypeSafe |
| `analyze.py` | Analyzes the email-dataset full run: duplicates, subsets, score reliability, cutoffs and review rates |
| `fetch_dataset.sh` | Downloads email-dataset at commit `84209612` |
| `fetch_lingspam.sh` | Downloads Ling-Spam and checks its SHA-256 checksum |
| `phish.py`, `PHISHING.md` | The three-way ham, spam or phishing experiment |
| `ood_test.py`, `OUT_OF_DISTRIBUTION.md` | The out-of-distribution test |
| `fetch_nazario.sh`, `fetch_ood.sh` | Download the phishing corpus and the 2026 mail |
| `results/` | Scores and token counts for each email, without the email text |
| `report/` | Notebooks of charts: `confusion_matrices.ipynb` (spam or ham) and `phishing_matrices.ipynb` (three categories), plus `confusion_matrices.html` |

## Data

[email-dataset](https://github.com/realprogrammersusevim/email-dataset) is MIT-licensed; its
LICENSE file has the copyright notices.

The further experiments use three more sources, credited in their own write-ups: Jose Nazario's
[phishing corpus](https://monkey.org/~jose/phishing/) (CC BY 4.0), Bruce Guenter's
[spam archive](http://untroubled.org/spam/), and the public archives of the python-list and
python-announce-list mailing lists.

The Ling-Spam corpus is described in: I. Androutsopoulos, J. Koutsias, K.V. Chandrinos, G. Paliouras
and C.D. Spyropoulos, "An Evaluation of Naive Bayesian Anti-Spam Filtering", *Proceedings of the
Workshop on Machine Learning in the New Information Age, 11th European Conference on Machine
Learning*, Barcelona, 2000, pp. 9–17. Its readme asks that published work using it credit the corpus
and notify its author.

This repository doesn't include messages from either dataset: `results/` refers to them only by file
path.

## Acknowledgments

Thanks to I. Androutsopoulos, J. Koutsias, K.V. Chandrinos, G. Paliouras and C.D. Spyropoulos for
creating the Ling-Spam corpus and making it freely available. It has been a standard test set for
spam filtering since 2000, and this experiment's second test would not exist without it. Their paper
is cited in [Data](#data).

Thanks also to the creators of [email-dataset](https://github.com/realprogrammersusevim/email-dataset)
for publishing it under the MIT license, to Jose Nazario for the phishing corpus, and to Bruce
Guenter for maintaining a public spam archive since 1998.

## License

MIT. See [LICENSE](LICENSE).
