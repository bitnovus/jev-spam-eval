"""Recompute paired metrics from saved predictions; no API calls."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from evaluate import OUT, MODEL, rows, base, phish

raw = [json.loads(line) for line in (OUT/'predictions.jsonl').read_text().splitlines()]
ok = {(r['set'], r['file'], r['arm']): r for r in raw if 'choices' in r}
source = rows()
lines = ['# Jev Choice with richer email evidence', '',
         f'Model: `{MODEL}`. Three variants on the same messages. Existing labels and category definitions preserved.', '',
         '- Text: original input and original defined Choice, rerun contemporaneously.',
         '- Enriched: same question plus Reply-To, link text/destinations/hostnames, and attachment names/types.',
         '- Enriched + framing: enriched input plus evidence-focused instructions; same class definitions.',
         '- No links were visited. No rule-based phishing scores, examples, fitting, or threshold changes.',
         '- Body remains truncated at 6,000 characters. Links are extracted from full textual MIME parts; limits: 80 links, roughly 24,000 serialized link characters, 2,000 characters per destination, 500 per anchor text, 40 attachment records.',
         '- Predictions are exploratory on previously inspected datasets, not an untouched benchmark.', '']
variants = [('Text', 'text', 'category'), ('Enriched', 'enriched', 'category'), ('Enriched + framing', 'enriched', 'evidence_framing')]
summaries = {}
for group in ('main', 'fresh', 'recent'):
    saved = [r for r in source if r['set'] == group]
    paired = [r for r in saved if all((group, r['file'], arm) in ok for arm in ('text', 'enriched'))]
    if not paired:
        continue
    y = np.array([r['label'] for r in paired])
    lines += [f'## {group}: {len(paired)}/{len(saved)} paired messages', '',
              '| Variant | Accuracy | Macro F1 | Phishing recall | Phishing precision | Ham → phish | Spam → phish |',
              '|---|---:|---:|---:|---:|---:|---:|']
    results = {}
    for title, arm, question in variants:
        records = [ok[(group, r['file'], arm)] for r in paired]
        pred = np.array([phish.OPTION_TO_CLASS[r['choices'][question]['choice']] for r in records])
        mat = confusion_matrix(y, pred, labels=phish.CLASSES)
        result = {'n': len(y), 'accuracy': accuracy_score(y, pred),
                  'macro_f1': f1_score(y, pred, labels=phish.CLASSES, average='macro', zero_division=0),
                  'phishing_recall': recall_score(y == 'phish', pred == 'phish', zero_division=0),
                  'phishing_precision': precision_score(y == 'phish', pred == 'phish', zero_division=0),
                  'confusion': mat.tolist()}
        results[title] = pred
        if group == 'recent':
            lines.append(f'| {title} | — | — | {result["phishing_recall"]:.2%} | — | — | — |')
        else:
            lines.append(f'| {title} | {result["accuracy"]:.2%} | {result["macro_f1"]:.4f} | {result["phishing_recall"]:.2%} | {result["phishing_precision"]:.2%} | {mat[0,2]} | {mat[1,2]} |')
            result['phishing_auc'] = roc_auc_score(y == 'phish', [r['choices'][question]['probabilities']['phishing'] for r in records])
        summaries.setdefault(group, {})[title] = result
    lines += ['', 'Confusion matrices below use actual rows and predicted columns in order **ham, spam, phish**.', '']
    for title, result in summaries[group].items():
        lines += [f'{title}:', '```text', *[' '.join(f'{v:5d}' for v in row) for row in result['confusion']], '```', '']
    for title in ('Enriched', 'Enriched + framing'):
        old, new = results['Text'], results[title]
        fixed = int(((old != y) & (new == y)).sum())
        broken = int(((old == y) & (new != y)).sum())
        p = binomtest(fixed, fixed+broken, .5).pvalue if fixed+broken else 1.
        summaries[group][title]['vs_text'] = {'fixed': fixed, 'regressed': broken, 'mcnemar_exact_p': p}
        lines.append(f'- {title} versus text: **{fixed} errors corrected, {broken} correct predictions lost**; exact paired McNemar p={p:.3g} (message-level; campaign dependence limits interpretation).')
    historic = np.array([phish.predicted(r, 'category') for r in paired])
    lines += [f'- Original saved Choice versus contemporaneous text rerun: {int((historic != results["Text"]).sum())} labels changed.', '']
    if group in ('main', 'fresh'):
        ebay = np.array(['ebay' in base.read_email(r['file'], phish.dataset_of(r['file']))[1]['subject'].lower() for r in paired])
        missed = (y == 'phish') & (results['Text'] == 'ham') & ebay
        lines.append(f'- eBay-subject phishing called ham by text: {int(missed.sum())}; recovered as phishing by enriched: {int((missed & (results["Enriched"] == "phish")).sum())}; by enriched + framing: {int((missed & (results["Enriched + framing"] == "phish")).sum())}.')
        lines.append('')

inputs = sum(r.get('input_tokens', 0) for r in raw)
outputs = sum(r.get('output_tokens', 0) for r in raw)
coverage = [r for r in ok.values() if r['arm'] == 'enriched']
lines += ['## Coverage and usage', '',
          f'- Successful unique requests: {len(ok)} / {2*len(source)}; recorded failed attempts: {sum("error" in r for r in raw)}.',
          f'- Input tokens: {inputs:,}; output tokens: {outputs:,}.',
          f'- Historical-rate cost estimate: ${inputs * .042/1e6:.3f}, using the previous experiment\'s $0.042/million input tokens. This is not a verified current bill; three-message pilot excluded.',
          f'- Enriched messages with links: {sum(r["link_count"] > 0 for r in coverage)}; with attachments: {sum(r["attachment_count"] > 0 for r in coverage)}; with omitted links: {sum(r["links_omitted"] > 0 for r in coverage)}.',
          '- Recent set contains phishing only, so it measures recall, not false-positive rate or useful precision.',
          '- Existing corpus labels include known spam/phishing ambiguity. Source and campaign artifacts remain.',
          '- This experiment bundles several added metadata fields; a gain cannot be attributed to URLs alone.', '']
(OUT/'REPORT.md').write_text('\n'.join(lines))
(OUT/'metrics.json').write_text(json.dumps(summaries, indent=2))
print('\n'.join(lines))
