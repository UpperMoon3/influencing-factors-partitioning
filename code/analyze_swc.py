"""Does the silhouette coefficient predict, before training, whether a regime is
worth separating?

The question is answered at the level of an individual regime, which is the level
at which the merge decision of Algorithm 1 is actually taken: for every regime,
the silhouette gain over the pooled sample is compared with the accuracy that
partitioning actually gains on the control objects of that same regime.
"""
import json
import numpy as np
from scipy.stats import spearmanr, pearsonr

RES = json.load(open('../results/main.json'))
CLFS = ['LDA', 'QDA', 'NB', 'KNN', 'RF', 'HGB', 'MLP']
PARAM = {'LDA', 'QDA', 'NB'}

rows = []
for ds, v in RES.items():
    for lv, o in v['levels'].items():
        sw = o['swc']
        pooled = sw['pooled']
        for c in CLFS:
            per = {}
            for r in o['rows']:
                if r['clf'] != c:
                    continue
                for k in r:
                    if k.startswith('d_r'):
                        per.setdefault(int(k[3:]), []).append(r[k])
            for reg, vals in per.items():
                key = f'r{reg}'
                if key not in sw or sw[key] is None or np.isnan(sw[key]):
                    continue
                rows.append({'ds': ds, 'lv': lv, 'clf': c, 'regime': reg,
                             'dswc': sw[key] - pooled, 'swc': sw[key],
                             'dacc': float(np.mean(vals)), 'n': sw[f'n{reg}']})

print(f'{len(rows)} (stream, level, classifier, regime) cells')
d = np.array([r['dswc'] for r in rows]); a = np.array([r['dacc'] for r in rows])
print('Spearman  rho = %+.3f  p = %.3g' % spearmanr(d, a))
print('Pearson   r   = %+.3f  p = %.3g' % pearsonr(d, a))

# the silhouette gain as a decision rule
pred = d > 0
truth = a > 0
tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum())
fn = int((~pred & truth).sum()); tn = int((~pred & ~truth).sum())
print(f'rule "separate if dSWC > 0":  TP {tp}  FP {fp}  FN {fn}  TN {tn}')
if tp + fp: print('  precision %.3f' % (tp / (tp + fp)))
if tp + fn: print('  recall    %.3f' % (tp / (tp + fn)))
print('  accuracy  %.3f' % ((tp + tn) / len(rows)))
print('  base rate of regimes that actually gain: %.3f' % truth.mean())

# the same, split by model class, because the theory predicts a difference
for name, sel in (('parametric', [r['clf'] in PARAM for r in rows]),
                  ('flexible', [r['clf'] not in PARAM for r in rows])):
    sel = np.array(sel)
    print(f'{name:10s}: mean dacc {a[sel].mean():+.2f}  rho {spearmanr(d[sel], a[sel])[0]:+.3f}')

# subsample size against gain
n = np.array([r['n'] for r in rows], float)
print('Spearman(size, gain) = %+.3f' % spearmanr(n, a)[0])
json.dump(rows, open('../results/swc_cells.json', 'w'), indent=1)
