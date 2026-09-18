"""Matched-evidence TF-IDF logistic regression and Jev comparison; no API calls."""
import hashlib
import json
import time
from collections import Counter

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from evaluate import OUT, inputs, rows, phish
from tfidf_baseline import near_duplicate_groups


def digest(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def train_predict(texts, y, train, test):
    vectorizer, model = phish.new_model()
    model.fit(vectorizer.fit_transform([texts[i] for i in train]), y[train])
    assert list(model.classes_) == [0, 1, 2]
    assert max(model.n_iter_) < model.max_iter
    return model.predict_proba(vectorizer.transform([texts[i] for i in test]))


def measure(y, p, pred):
    m = confusion_matrix(y, pred, labels=[0, 1, 2])
    result = {'n': len(y), 'accuracy': float(accuracy_score(y, pred)),
              'macro_f1': float(f1_score(y, pred, labels=[0, 1, 2], average='macro', zero_division=0)),
              'phishing_recall': float(m[2, 2]/m[2].sum()),
              'phishing_precision': float(m[2, 2]/m[:, 2].sum()) if m[:, 2].sum() else 0.,
              'confusion': m.tolist()}
    if len(set(y)) > 1:
        result['phishing_auc'] = float(roc_auc_score(y == 2, p[:, 2]))
        result['log_loss'] = float(log_loss(y, p, labels=[0, 1, 2]))
    return result


def main():
    started = time.monotonic()
    source = rows()
    jev = {(r['set'], r['file'], r['arm']): r for r in
           map(json.loads, (OUT/'predictions.jsonl').read_text().splitlines()) if 'choices' in r}
    texts = {'text': [], 'enriched': []}
    print('Reconstructing and verifying the exact Jev inputs...', flush=True)
    for r in source:
        original, rich = inputs(r)
        for arm, state in [('text', original), ('enriched', rich)]:
            assert digest(state) == jev[(r['set'], r['file'], arm)]['state_sha256']
        plain = phish.state_text(original)
        texts['text'].append(plain)
        metadata = {k: v for k, v in rich.items() if k not in original}
        texts['enriched'].append(plain + '\n' + json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    y = np.array([phish.CLASSES.index(r['label']) for r in source])
    indices = {s: np.array([i for i, r in enumerate(source) if r['set'] == s]) for s in ('main', 'fresh', 'recent')}
    main_ix = indices['main']
    main_text = [texts['text'][i] for i in main_ix]
    groups = near_duplicate_groups(main_text, .8)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(main_text, y[main_ix], groups))
    fold_ids = np.full(len(source), -1)
    coverage = np.zeros(len(main_ix), dtype=int)
    for f, (train, test) in enumerate(folds):
        assert not set(groups[train]) & set(groups[test])
        assert not set(train) & set(test)
        coverage[test] += 1
        fold_ids[main_ix[test]] = f
    assert np.all(coverage == 1)
    with (OUT/'regression_splits.jsonl').open('w') as out:
        for local, i in enumerate(main_ix):
            out.write(json.dumps({'file': source[i]['file'], 'fold': int(fold_ids[i]), 'group': int(groups[local])})+'\n')
    print(f'Frozen identical folds: {len(main_ix)} messages, {len(set(groups))} near-duplicate groups.', flush=True)
    predictions = {}
    for arm in ('text', 'enriched'):
        p = np.full((len(source), 3), np.nan)
        for f, (train, test) in enumerate(folds):
            p[main_ix[test]] = train_predict(texts[arm], y, main_ix[train], main_ix[test])
            print(f'{arm}: main fold {f+1}/5 complete ({time.monotonic()-started:.1f}s)', flush=True)
        p[indices['fresh']] = train_predict(texts[arm], y, main_ix, indices['fresh'])
        older = np.concatenate([main_ix, indices['fresh']])
        p[indices['recent']] = train_predict(texts[arm], y, older, indices['recent'])
        assert np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1)
        predictions['LR '+arm] = p
        print(f'{arm}: fresh and recent complete ({time.monotonic()-started:.1f}s)', flush=True)
    predicted_labels = {name:p.argmax(axis=1) for name,p in predictions.items()}
    rounded_sums = {}
    for title, arm, question in [('Jev text', 'text', 'category'), ('Jev enriched', 'enriched', 'category'),
                                 ('Jev enriched + framing', 'enriched', 'evidence_framing')]:
        predictions[title] = np.array([[jev[(r['set'],r['file'],arm)]['choices'][question]['probabilities'][option]
                                       for option in ('legitimate','spam','phishing')] for r in source])
        assert np.isfinite(predictions[title]).all()
        assert np.allclose(predictions[title].sum(axis=1), 1, atol=.02)
        rounded_sums[title] = int((np.abs(predictions[title].sum(axis=1)-1)>1e-8).sum())
        predictions[title] /= predictions[title].sum(axis=1, keepdims=True)
        # Returned choice can break ties obscured by rounded probability values.
        predicted_labels[title] = np.array([phish.CLASSES.index(phish.OPTION_TO_CLASS[
            jev[(r['set'],r['file'],arm)]['choices'][question]['choice']]) for r in source])
    for arm in ('text', 'enriched'):
        predictions['50/50 '+arm] = (predictions['LR '+arm]+predictions['Jev '+arm])/2
        predicted_labels['50/50 '+arm] = predictions['50/50 '+arm].argmax(axis=1)
    # Same original-text near-copy exclusion for every model; no variant-specific filtering.
    fresh_rows = [source[i] for i in indices['fresh']]
    main_rows = [source[i] for i in main_ix]
    copies = phish.near_copies(fresh_rows, main_rows)
    indices['fresh without near-copies'] = indices['fresh'][~copies]
    with (OUT/'regression_predictions.jsonl').open('w') as out:
        for i,r in enumerate(source):
            item = {k:r[k] for k in ('set','file','label')}
            item['fold'] = int(fold_ids[i]) if r['set']=='main' else None
            item['probabilities'] = {name:dict(zip(phish.CLASSES, map(float,p[i]))) for name,p in predictions.items()}
            item['predicted'] = {name:phish.CLASSES[pred[i]] for name,pred in predicted_labels.items()}
            out.write(json.dumps(item)+'\n')
    order = ['Jev text','LR text','Jev enriched','LR enriched','Jev enriched + framing','50/50 text','50/50 enriched']
    results = {group:{name:measure(y[ix], predictions[name][ix], predicted_labels[name][ix]) for name in order} for group,ix in indices.items()}
    deltas = {}
    for group,ix in indices.items():
        deltas[group] = {}
        for a,b in [('LR text','LR enriched'),('Jev text','Jev enriched'),('LR enriched','Jev enriched')]:
            ca = predicted_labels[a][ix]==y[ix]
            cb = predicted_labels[b][ix]==y[ix]
            deltas[group][a+' -> '+b] = {'corrected':int((~ca & cb).sum()), 'regressed':int((ca & ~cb).sum()),
                                        'both_wrong':int((~ca & ~cb).sum())}
    manifest = {'sklearn':sklearn.__version__, 'numpy':np.__version__, 'jev_probability_sums_normalized':rounded_sums,
                'source_code_sha256':hashlib.sha256((OUT/'regression.py').read_bytes()).hexdigest(),
                'input_verification':'All 19,772 reconstructed Jev state SHA-256 values matched.',
                'tfidf':{'sublinear_tf':True,'min_df':2,'max_features':200000,'analyzer':'word','ngram_range':[1,1]},
                'logistic_regression':{'C':10,'max_iter':2000,'solver':'lbfgs'},
                'folds':5,'split_seed':0,'group_cosine_threshold':.8,'groups':len(set(groups)),
                'fresh_near_copies':int(copies.sum()),'training_main':'out-of-fold, approximately 80% of main',
                'training_fresh':'all 5733 main messages','training_recent':'all 9033 main + fresh messages',
                'elapsed_seconds':time.monotonic()-started}
    (OUT/'regression_manifest.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'regression_metrics.json').write_text(json.dumps({'metrics':results,'paired_changes':deltas},indent=2))
    lines = ['# Matched-evidence comparison: Jev versus logistic regression','',
             'Both methods receive the same original text or the same enriched evidence. All reconstructed input hashes match the preceding Jev experiment. No new API calls.', '',
             '## Evaluation design','',
             '- Main: five-fold StratifiedGroupKFold, near-duplicate clusters kept together. Both regression variants use the exact same folds.',
             '- Fresh: each regression variant trained on the main set only. Also report the common subset excluding near-copies of main messages.',
             '- Recent: each regression variant trained on main + fresh (9,033 older messages); 853 later phishing messages are evaluation only.',
             '- Word TF-IDF: sublinear TF, min_df=2, max_features=200,000. Logistic regression: C=10, max_iter=2,000. Unchanged hyperparameters; no test-set tuning.',
             '- Vectorizer vocabulary and IDF fit only on each training partition. Unsupervised grouping/exclusion follows the original protocol and uses original text.',
             '- Enriched regression appends a JSON serialization of exactly the additional Jev fields to the original subject/from/body text. Same caps and metadata; no hand-built phishing indicators.',
             '- Jev predictions are the preceding contemporaneous rerun. Standard defined Choice is primary; evidence-focused wording is secondary.',
             '- Jev classification uses the returned Choice label, preserving API tie-breaking. Rounded Jev probabilities are renormalized before log loss and ensemble averaging. Regression and ensemble ties use class order ham, spam, phish.',
             '- Ensembles are fixed 50/50 probability averages, without fitted weights or thresholds.',
             '- Matched evidence is not matched supervision: regression uses corpus labels; Jev uses a pretrained model and task definitions. Pretraining exposure is unknown.', '']
    for group in ('main','fresh','fresh without near-copies','recent'):
        lines += [f'## {group} (n={len(indices[group]):,})','',
                  '| Model | Accuracy | Macro F1 | Phishing recall | Phishing precision | Ham → phish | Spam → phish |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for name in order:
            m=results[group][name]
            if group=='recent':
                lines.append(f'| {name} | — | — | {m["phishing_recall"]:.2%} | — | — | — |')
            else:
                lines.append(f'| {name} | {m["accuracy"]:.2%} | {m["macro_f1"]:.4f} | {m["phishing_recall"]:.2%} | {m["phishing_precision"]:.2%} | {m["confusion"][0][2]} | {m["confusion"][1][2]} |')
        lines.append('')
        for comparison,counts in deltas[group].items():
            lines.append(f'- {comparison}: {counts["corrected"]} corrected, {counts["regressed"]} regressed; {counts["both_wrong"]} shared errors.')
        lines += ['','Matrices: actual rows / predicted columns = ham, spam, phish.','']
        for name in order:
            lines += [name, '```text', *[' '.join(f'{v:5}' for v in row) for row in results[group][name]['confusion']], '```','']
    lines += ['## Limits and verification','',
              f'- All {len(source):,} message pairs retained; all state hashes verified. Every main message held out exactly once, with no duplicate-group overlap between training and test.',
              f'- {int(copies.sum())} fresh messages excluded in the shared near-copy sensitivity analysis.',
              '- Grouping is fixed from original text for comparability; extra URL/domain campaign relationships may remain across folds.',
              '- Original labels contain known noise and classes come from different sources. Previously inspected sets are exploratory, not a locked benchmark.',
              '- Recent contains only phishing: no modern false-positive rate or precision conclusion is supported.',
              '- Word tokenization may not capture every URL distinction. This tests the existing regression family with equal evidence; it is not a tuned ceiling for supervised methods.',
              f'- Local fit and analysis time: {manifest["elapsed_seconds"]:.1f}s. All regression fits converged; probabilities checked. No API spend.', '']
    (OUT/'REGRESSION_REPORT.md').write_text('\n'.join(lines))
    for group in ('main','fresh','fresh without near-copies','recent'):
        print(group, {name:{k:round(v,5) for k,v in results[group][name].items() if k in ('accuracy','phishing_recall')} for name in order}, flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
