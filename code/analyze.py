"""Aggregate the raw runs into the tables and statistics reported in the paper."""
import json, itertools, sys
import numpy as np
from scipy.stats import wilcoxon, spearmanr, friedmanchisquare

RES = json.load(open('../results/main.json'))
CLFS = ['LDA', 'QDA', 'NB', 'KNN', 'RF', 'HGB', 'MLP']
PARAMETRIC = {'LDA', 'QDA', 'NB'}


def holm(pvals):
    idx = np.argsort(pvals)
    out = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(idx):
        adj = (len(pvals) - rank) * pvals[i]
        running = max(running, adj)
        out[i] = min(1.0, running)
    return out


def collect(rows, clf, key):
    return np.array([r[key] for r in rows if r['clf'] == clf and key in r])


def table_paired(dataset, level, rows):
    """Mean accuracy of every method and the paired differences."""
    recs = []
    praw = []
    for c in CLFS:
        g = collect(rows, c, 'global_acc'); f = collect(rows, c, 'feature_acc'); q = collect(rows, c, 'regime_acc')
        if len(g) == 0:
            continue
        d1, d2 = q - g, q - f
        p1 = wilcoxon(q, g).pvalue if np.any(d1 != 0) else 1.0
        p2 = wilcoxon(q, f).pvalue if np.any(d2 != 0) else 1.0
        praw += [p1, p2]
        recs.append({'clf': c, 'pool': g.mean(), 'feat': f.mean(), 'part': q.mean(),
                     'd_pool': d1.mean(), 'sd_pool': d1.std(ddof=1),
                     'd_feat': d2.mean(), 'sd_feat': d2.std(ddof=1),
                     'auc_pool': collect(rows, c, 'global_auc').mean(),
                     'auc_part': collect(rows, c, 'regime_auc').mean()})
    adj = holm(np.array(praw))
    for i, r in enumerate(recs):
        r['p_pool'], r['p_feat'] = adj[2 * i], adj[2 * i + 1]
    return recs


def stars(p):
    return '$^{*}$' if p < 0.05 else ''


def fmt(x, n=2):
    return f'{x:+.{n}f}' if x < 0 or x > 0 else f'{x:.{n}f}'


def main():
    out = {}
    for ds, v in RES.items():
        out[ds] = {}
        for lv, o in v['levels'].items():
            recs = table_paired(ds, lv, o['rows'])
            sel = np.array([r['sel_acc'] for r in o['rows'] if 'sel_acc' in r])
            out[ds][lv] = {'recs': recs, 'n_regimes': o['n_regimes'],
                           'sel': float(sel.mean()) if len(sel) else None,
                           'swc': o['swc']}
            # chronological
            cr = o.get('chrono', [])
            crecs = []
            for c in CLFS:
                g = np.array([r['global_acc'] for r in cr if r['clf'] == c])
                f = np.array([r['feature_acc'] for r in cr if r['clf'] == c])
                q = np.array([r['regime_acc'] for r in cr if r['clf'] == c])
                if len(g):
                    crecs.append({'clf': c, 'pool': g.mean(), 'feat': f.mean(), 'part': q.mean(),
                                  'd_pool': (q - g).mean(), 'd_feat': (q - f).mean()})
            csel = [r['sel_acc'] for r in cr if r['clf'] == 'SELECTION']
            out[ds][lv]['chrono'] = crecs
            out[ds][lv]['chrono_sel'] = float(np.mean(csel)) if csel else None
    json.dump(out, open('../results/agg.json', 'w'), indent=1, default=float)

    # ------------------------------------------------------------- printout
    for ds in out:
        print('=' * 78); print(ds)
        for lv in out[ds]:
            o = out[ds][lv]
            print(f'-- {lv}  (M={o["n_regimes"]})   selection={o["sel"]}')
            print(f'{"clf":5s} {"pool":>7s} {"+feat":>7s} {"part":>7s} {"d(p-pool)":>11s} {"d(p-feat)":>11s}')
            for r in o['recs']:
                print(f'{r["clf"]:5s} {r["pool"]:7.2f} {r["feat"]:7.2f} {r["part"]:7.2f} '
                      f'{r["d_pool"]:+8.2f}{stars(r["p_pool"]):3s} {r["d_feat"]:+8.2f}{stars(r["p_feat"]):3s}')
            if o['chrono']:
                print('   chronological:', ', '.join(f'{r["clf"]}: {r["d_pool"]:+.2f}/{r["d_feat"]:+.2f}'
                                                     for r in o['chrono']), '| sel', o['chrono_sel'])

    # parametric vs flexible, pooled over everything
    dp_par, dp_flex, df_par, df_flex = [], [], [], []
    for ds in out:
        for lv in out[ds]:
            for r in out[ds][lv]['recs']:
                (dp_par if r['clf'] in PARAMETRIC else dp_flex).append(r['d_pool'])
                (df_par if r['clf'] in PARAMETRIC else df_flex).append(r['d_feat'])
    print('\nGain over pooled  : parametric %.2f  flexible %.2f' % (np.mean(dp_par), np.mean(dp_flex)))
    print('Gain over +feature: parametric %.2f  flexible %.2f' % (np.mean(df_par), np.mean(df_flex)))


if __name__ == '__main__':
    main()
