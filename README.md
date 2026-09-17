# Classification from a specification

**How far can a written decision rule take you before you need a labeled training set?**

This repository explores that question with email classification. It uses
[TypeSafe](https://typesafe.ai)'s pretrained Jev model to turn natural-language questions into
probability scores, then compares those scores with TF-IDF classifiers trained on labeled email.
The experiments cover spam detection, three-way phishing classification, learning curves, and
distribution shift across public mail collections spanning 2000–2026.

On 18,514 unique emails, a written definition of spam reached **98.3% accuracy**, alongside
**98.4%** for logistic regression trained on roughly 14,800 labels per fold. Averaging their scores
reached **99.2%**, cutting errors from 299 to 144. On a separate set of 633 emails from 2026, an
unchanged TypeSafe question scored **97.3%**, versus **72.5%** for TF-IDF trained on older mail.

For AI engineers, these results raise three practical possibilities:

- **Start with a specification before collecting a training set.** The plain question already
  scored 96.0% on the first dataset. Defining the boundary between unsolicited mail and expected
  newsletters raised it to 98.3%, without fitting a task-specific model.
- **Test transfer as well as held-out accuracy.** TF-IDF matched or beat TypeSafe when trained on
  each dataset's own labels. Changing the source or era of the mail reversed that result.
- **Use the score alongside an existing classifier.** The two approaches made different mistakes.
  A simple 50/50 average improved accuracy on both binary datasets, without learning ensemble weights.

The largest API run, 19,528 emails with four questions per email, cost about **$1.12** at the
September 2026 price recorded for these experiments. That makes this an interesting design space
for classification, routing, and triage, although only email classification is evaluated here.

> **Scope:** exploratory experiments with `jev-1.13.0`, not a production benchmark. Jev was not
> fine-tuned, and requests contained no labeled demonstrations. However, the detailed criteria were
> written after inspecting errors in labeled samples totaling 1,000 emails. “No task-specific
> training” does not mean “no supervision.” Public datasets may also have appeared in Jev's
> pretraining. See [Caveats](#caveats).

## The interface: a question becomes a score

TypeSafe calls a yes/no question a [Noul](https://docs.typesafe.ai/primitives/noul). The application
supplies an email and a question; the API returns a score between 0 and 1 representing the
probability of “yes.” This is the plain-question path used in the experiments:

```python
from typesafe_sdk import AsyncTypeSafeClient, Noul

async def spam_score(email: dict, api_key: str) -> float:
    async with AsyncTypeSafeClient(api_key=api_key) as client:
        response = await client.system_one(
            {"email": email},
            {"spam": Noul(instructions="Is this email spam?")},
        )
    return response.nouls["spam"].noul
```

The score can feed a threshold, a review queue, or an ensemble. A Noul can also take structured
criteria defining what counts as true or false; the [tested specification](#the-question-that-worked-best)
is below. For multiclass decisions, a [Choice](https://docs.typesafe.ai/primitives/choice) returns
a category and per-category probabilities.

The engineering question is how well those scores behave: whether the specification transfers,
whether confidence is useful, and how much labeled data a conventional model needs to compete.

## Results

### Spam detection: competitive alone, stronger in combination

The main comparison uses 18,514 unique messages from
[email-dataset](https://github.com/realprogrammersusevim/email-dataset). Exact duplicates are
removed. TF-IDF predictions use five-fold cross-validation with near-duplicate clusters kept
together, so near-copies cannot appear in both training and test folds. Binary decisions use a
0.5 threshold.

| Approach | Accuracy | AUC | Missed spam | Ham flagged as spam |
|---|---|---|---|---|
| TypeSafe, plain question | 95.96% | 0.9975 | 54 | 694 |
| TypeSafe, detailed criteria | **98.33%** | 0.9984 | 106 | 203 |
| TF-IDF logistic regression, ~14.8K training labels per fold | **98.39%** | 0.9986 | 149 | 150 |
| TF-IDF naive Bayes, ~14.8K training labels per fold | 97.07% | 0.9971 | 367 | 175 |
| 50/50 average of detailed criteria + logistic regression | **99.22%** | **0.9995** | **69** | **75** |

Ham means legitimate email. AUC measures ranking quality; 1.0 is perfect.

**Complementary errors are the most useful result here.** TypeSafe and logistic regression disagree
on 466 messages, with 228 wins for TypeSafe and 238 for logistic regression (McNemar p = 0.68).
Only 71 messages fool both. TypeSafe catches unfamiliar or obfuscated spam; logistic regression
often wins on news-feed items and vendor announcements whose sources the dataset labels as ham.
Their average reduces errors by **52% relative to logistic regression**.

Evaluation details matter: allowing near-copies into different folds raises logistic regression
from 98.39% to 98.90%. That apparent gain is larger than the gap between the two standalone approaches.

### Distribution shift: where the comparison changes

In these tests, TF-IDF is trained on a different collection or older mail. TypeSafe uses questions
from the earlier experiments without rewriting them for the new messages.

| Test set | TypeSafe | TF-IDF | Metric |
|---|---|---|---|
| Ling-Spam, 2,876 unique messages; TF-IDF trained on email-dataset | **98.6%** | 73.0% | Binary accuracy |
| 2024–25 phishing, 853 messages; TF-IDF trained on older mail | **91.0–93.6%** | 70.3% | Phishing recall only |
| 2026 list posts and spam-trap mail, 633 messages; TF-IDF trained on older mail | **97.3%** | 72.5% | Legitimate vs. not-legitimate accuracy |

On the modern set, TF-IDF passed 165 of 300 spam-trap messages as legitimate. TypeSafe caught 163
of those 165. The 97.3% result uses the three-way question with urgency and authority criteria;
the names-only question scored 98.6% on this binary task, but performed poorly at separating spam
from phishing in the main three-way test.

This suggests value when representative labels are scarce or the input distribution changes. It
does not establish robustness to arbitrary drift: the sets are small, modern legitimate mail comes
from only two Python mailing lists, and Jev's pretraining exposure is unknown. The phishing-only
test cannot measure false positives. [Full results and failure cases →](OUT_OF_DISTRIBUTION.md)

## How many labeled emails TF-IDF needs

The answer varies sharply by dataset. These learning curves keep the test folds fixed and train
TF-IDF on progressively smaller, stratified samples of the training folds, with five draws per
sample size.

| Task | TypeSafe accuracy | TF-IDF labels to reach roughly that accuracy | TF-IDF with all training labels |
|---|---|---|---|
| email-dataset, binary | 98.3%, detailed criteria | ~10,000 | 98.4% (~14,800 labels) |
| Ling-Spam, binary | 98.6%, plain question | ~200 | **99.4%** (~2,300 labels) |
| Legitimate / spam / phishing | 94.2%, urgency and authority criteria | ~100 | **98.7%** (~4,600 labels) |

These are approximate comparisons at the sampled training sizes, not exact sample-complexity
estimates. On email-dataset, TF-IDF reached the **plain question's 96.0% with about 1,000 labels**;
the larger gap depends on criteria developed with labeled feedback. The Ling-Spam comparison is
by accuracy: even with all labels, TF-IDF's best balanced accuracy was 98.4%, versus TypeSafe's 99.0%.

The two tasks requiring fewer labels also offer strong dataset-specific shortcuts. Ling-Spam's ham
comes from one linguistics list. In the three-way task, phishing comes from a different collection
than ham and spam. Representative labels can be extremely effective; their value depends on what
they represent.

<details>
<summary>Full learning curves</summary>

Accuracy averaged across five draws at each reduced training size. “All” uses one run with the
full training folds. TypeSafe uses the question named in the summary table above.

| Dataset / model | TypeSafe | 25 | 50 | 100 | 200 | 500 | 1,000 | 2,000 | 5,000 | 10,000 | All |
|---|---|---|---|---|---|---|---|---|---|---|---|
| email-dataset, logistic regression | 0.983 | 0.846 | 0.892 | 0.911 | 0.928 | 0.948 | 0.962 | 0.972 | 0.979 | 0.983 | 0.984 |
| email-dataset, naive Bayes | 0.983 | 0.855 | 0.896 | 0.914 | 0.929 | 0.945 | 0.953 | 0.958 | 0.964 | 0.968 | 0.971 |
| Ling-Spam, logistic regression | 0.986 | 0.840 | 0.861 | 0.909 | 0.957 | 0.976 | 0.982 | 0.986 | | | 0.986 |
| Ling-Spam, naive Bayes | 0.986 | 0.902 | 0.943 | 0.975 | 0.987 | 0.991 | 0.993 | 0.994 | | | 0.994 |
| Three categories, logistic regression | 0.942 | 0.888 | 0.923 | 0.942 | 0.958 | 0.974 | 0.980 | 0.984 | | | 0.987 |

Accuracy varied between draws by up to 3.4 percentage points at 100 labels or fewer, and by less
than half a point at 1,000 or more. The underlying runs are in `results/learning_curve_*.jsonl`.

</details>

## Specifications are part of the model

Writing a more detailed definition changed the decision boundary substantially. On email-dataset,
it reduced false positives from 694 to 203, while increasing missed spam from 54 to 106. This was
a tradeoff between which mistakes to make, not simply a better version of the same classifier.

Breaking the decision into nine questions did not produce a similar improvement. Questions about
commercial intent, scams, filter evasion, and expected correspondence made errors easier to inspect,
but combining their answers did no better than adjusting the plain question's threshold in these
samples. More decomposition did not automatically yield a better classifier.

### Second dataset: Ling-Spam

The four questions were carried over unchanged to
[Ling-Spam](https://www.aueb.gr/users/ion/data/lingspam_public.tar.gz), a corpus of linguistics-list
posts and spam from 2000. On its 2,876 unique messages, **the plain question beat the detailed
specification**.

| Approach | Accuracy | Balanced accuracy | Missed spam (of 468) | Ham flagged as spam (of 2,408) |
|---|---|---|---|---|
| TypeSafe, plain question | 98.57% | **98.98%** | 2 | 39 |
| TypeSafe, detailed criteria | 97.01% | 97.53% | 8 | 78 |
| TF-IDF logistic regression, ~2.3K training labels per fold | 98.57% | 95.71% | 40 | 1 |
| TF-IDF naive Bayes, ~2.3K training labels per fold | **99.41%** | 98.44% | 14 | 3 |
| 50/50 average of TypeSafe + logistic regression, either question | **99.83%** | **99.55%** | 4 | 1 |

Balanced accuracy averages recall for ham and spam, rather than letting the larger ham class
dominate. TypeSafe catches more spam; the trained models make fewer false accusations.

About three-quarters of the 78 false positives from the detailed criteria are publishers' book
announcements. A likely explanation: the specification treats unsolicited advertising as spam,
but this corpus lacks the headers and list footers that would establish a subscription context.
The specification depends on information the input may not contain.

That is a useful failure mode for anyone building with language-defined decisions: **a more
explicit policy still needs evaluation on the inputs where it will run.**

### Three categories make the definition even more consequential

In the [phishing experiment](PHISHING.md), asking for “legitimate,” “spam,” or “phishing” with
category names alone scored 73.1%. Adding definitions raised accuracy to 93.5%; adding urgency
and authority criteria raised it to 94.2%, but helped little on a fresh set.

The largest change was semantic: names alone classified 1,095 of 1,911 spam messages as phishing;
definitions reduced that to 16. TF-IDF still won the main test at 98.7%, and a probability average
reached 99.3%. [Definitions, confusion matrices, and limitations →](PHISHING.md)

### Confidence can support selective automation

On email-dataset, routing TypeSafe scores from 0.3 to 0.7 to review covers **4.6% of messages**
and leaves **99.50% accuracy on the remainder**. At the same review share, logistic regression
leaves 99.58% and the ensemble 99.87%. These are retained-set accuracies, not measurements of a
deployed human-review workflow.

The scores also need calibration checks. For the detailed criteria, only 0.1% of messages scored
below 0.1 are spam, and 99.9% of those scored at least 0.9 are spam. But the 0.5–0.6 bucket is only
38% spam. A probability-shaped output is useful; it does not guarantee calibrated uncertainty.

## The question that worked best

This is the specification behind the 98.3% email-dataset result. It was developed from that
dataset's errors and did worse than the plain question on Ling-Spam.

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

<details>
<summary>Experiment history: wording changes, decomposition, and fresh samples</summary>

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

</details>

## Caveats

- **No task-specific fitting is not the same as no supervision.** The detailed spam criteria were
  written after inspecting mistakes in two labeled 500-email samples. The phishing urgency and
  authority criteria were also revised after inspecting errors. Fresh-sample checks are reported,
  but these are exploratory comparisons, not a locked benchmark. Different tables use different
  question variants; there is no single preselected policy behind every headline result.
- **Pretraining exposure is unknown.** Jev may have seen these public corpora, including recent
  archives. These runs cannot establish performance on data unseen during pretraining.
- **The baselines answer a bounded question.** TF-IDF logistic regression and naive Bayes are
  useful supervised comparisons. There are no embedding classifiers, fine-tuned transformers,
  other language models, or production spam filters in this experiment. TF-IDF trained on newer
  or more varied mail may transfer better.
- **Labels and source artifacts limit the conclusions.** Some newsletters, scams, and personal
  emails are mislabeled. Three-way categories come from different collections. Modern spam-trap
  labels are assumed, and its legitimate mail comes from only two lists. See the caveats in
  [PHISHING.md](PHISHING.md#caveats) and
  [OUT_OF_DISTRIBUTION.md](OUT_OF_DISTRIBUTION.md#caveats).
- **The input omits useful context.** Bodies are truncated to 6,000 characters; HTML is reduced
  to text, dropping link destinations. Many messages have no sender headers. Recipient history,
  attachments, and other signals available to a real mail system are not evaluated.
- **One model version, mostly single runs.** Results use `jev-1.13.0` in September 2026. Repeat
  calls can change borderline predictions. Learning curves repeat training-sample draws, not
  independent Jev evaluations. API costs use the recorded September 2026 rate of $0.042 per
  million input tokens with no output-token charge; this is not an end-to-end operating-cost comparison.

## Reproduce

You need [uv](https://docs.astral.sh/uv/). Saved predictions are included in `results/`, so you can
rebuild reports and train the baselines without an API key or new TypeSafe calls.

```sh
uv sync
./fetch_dataset.sh          # pinned email-dataset version
uv run spam_noul.py --questions criteria --all --report
uv run analyze.py           # duplicates, subsets, calibration, review rates
uv run tfidf_baseline.py    # trains baselines; writes results/tfidf_oof.jsonl
uv run learning_curve.py    # training sizes from 25 to 10,000 labels
```

Additional datasets and reports:

```sh
./fetch_lingspam.sh
uv run spam_noul.py --dataset lingspam --questions criteria --all --report
uv run tfidf_baseline.py --dataset lingspam
uv run learning_curve.py --dataset lingspam

./fetch_nazario.sh
uv run phish.py --report
uv run learning_curve.py --dataset phish

./fetch_ood.sh
uv run ood_test.py --report
```

The notebooks in `report/` visualize saved predictions and let you explore thresholds. GitHub
renders their saved charts. To regenerate the binary-classification notebook:

```sh
uv run jupyter nbconvert --to notebook --execute --inplace report/confusion_matrices.ipynb
```

### Make new API calls

These commands require a TypeSafe API key and overwrite the matching result files. Add `--resume`
to retain successful predictions and retry missing or failed messages.

```sh
cp .env.example .env        # add TYPESAFE_API_KEY
uv run spam_noul.py --questions single --per-class 250 --seed 42
uv run spam_noul.py --questions criteria --all --concurrency 16
uv run spam_noul.py --dataset lingspam --questions criteria --all --concurrency 16
uv run phish.py --all --concurrency 16
uv run ood_test.py --concurrency 16
```

A 500-email sample took about 10 seconds and cost about $0.03. The full email-dataset run took
206 seconds at concurrency 16 and used 26.7M input tokens for four questions per email, about
$1.12 at the recorded rate. These are observed batch timings, not per-request latency guarantees.

<details>
<summary>Recreate the intermediate prompt experiments</summary>

```sh
uv run spam_noul.py --questions multi --per-class 250 --seed 42
uv run spam_noul.py --questions multi --per-class 250 --seed 1
uv run spam_noul.py --questions criteria --per-class 250 --seed 42
uv run spam_noul.py --questions criteria --per-class 250 --seed 1
uv run spam_noul.py --questions criteria --per-class 250 --seed 3 \
  --exclude-from results/results_multi_250x2_seed42.jsonl results/results_multi_250x2_seed1.jsonl
```

</details>

## Repository layout

| Path | Contents |
|---|---|
| `spam_noul.py` | Reads the messages from either dataset, picks samples, defines the three question sets, calls the API and prints reports |
| `tfidf_baseline.py` | Trains the TF-IDF classifiers, keeping near-copies together, and compares them with TypeSafe |
| `learning_curve.py` | Trains the TF-IDF classifiers on smaller and smaller samples of their labels, to find how many they need to match TypeSafe |
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

This repository doesn't include the source email messages: `results/` refers to them only by file
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
