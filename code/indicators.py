"""Which cheap indicator predicts, before the candidate classifiers are trained,
whether a regime is worth separating?

Four indicators are compared against the accuracy that partitioning actually
gains on the control objects of the same regime:

  dSWC   silhouette gain of the regime over the pooled sample (the indicator
         proposed in the original formulation of the method);
  dF1    gain in the maximum Fisher discriminant ratio over the features;
  dN2    gain in the ratio of intra-class to inter-class nearest-neighbour
         distance, reversed in sign so that larger is better;
  gapLDA a direct pilot estimate of the heterogeneity gap of Section 3: the
         difference between the cross-validated error of one linear model fitted
         to the pooled sample and to the regime alone.  It is not training-free,
         but it costs two fits of the cheapest classifier in the candidate set.
"""
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr

import datasets as D, partitions as P

RNG = np.random.default_rng(11)
MAXN = 4000


def _sub(X, y, n=MAXN):
    if len(X) <= n:
        return X, y
    i = RNG.choice(len(X), n, replace=False)
    return X[i], y[i]


def f1_measure(X, y):
    """Maximum Fisher discriminant ratio over single features (larger = easier)."""
    cls = np.unique(y)
    if len(cls) < 2:
        return np.nan
    a, b = X[y == cls[0]], X[y == cls[1]]
    num = (a.mean(0) - b.mean(0)) ** 2
    den = a.var(0) + b.var(0) + 1e-12
    return float(np.max(num / den))


def n2_measure(X, y):
    """Ratio of intra-class to inter-class nearest-neighbour distance (smaller = easier)."""
    Z = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=min(len(Z) - 1, 20)).fit(Z)
    dist, idx = nn.kneighbors(Z)
    intra, inter = [], []
    for i in range(len(Z)):
        same = y[idx[i, 1:]] == y[i]
        d = dist[i, 1:]
        if same.any():
            intra.append(d[same][0])
        if (~same).any():
            inter.append(d[~same][0])
    if not intra or not inter:
        return np.nan
    return float(np.sum(intra) / (np.sum(inter) + 1e-12))


def swc_measure(X, y):
    if len(np.unique(y)) < 2:
        return np.nan
    return float(silhouette_score(StandardScaler().fit_transform(X), y))


def lda_gap(Xp, yp, Xr, yr):
    """Pilot estimate of the heterogeneity gap for a linear class, in p.p."""
    if len(np.unique(yr)) < 2 or len(yr) < 30:
        return np.nan
    m = LDA().fit(Xp, yp)
    err_pool = 100.0 * (1.0 - m.score(Xr, yr))
    err_reg = 100.0 * (1.0 - cross_val_score(LDA(), Xr, yr, cv=3).mean())
    return float(err_pool - err_reg)


def main():
    RES = json.load(open('../results/main.json'))
    data = {}
    X, hour, dates = D.load_powersupply()
    data['Powersupply'] = (X, D.powersupply_task(hour), P.powersupply_levels(dates)[0])
    Xe, _, de, ye = D.load_elec2()
    data['Elec2'] = (Xe, ye, P.elec2_levels(de)[0])

    cells = []
    for ds, v in RES.items():
        X, y, levels = data[ds]
        Xs, ys = _sub(X, y)
        base = {'swc': swc_measure(Xs, ys), 'f1': f1_measure(Xs, ys), 'n2': n2_measure(Xs, ys)}
        for lv, o in v['levels'].items():
            regime = levels[lv]
            for r in range(o['n_regimes']):
                m = regime == r
                if m.sum() < 50:
                    continue
                Xr, yr = _sub(X[m], y[m])
                ind = {'dswc': swc_measure(Xr, yr) - base['swc'],
                       'df1': f1_measure(Xr, yr) - base['f1'],
                       'dn2': base['n2'] - n2_measure(Xr, yr),
                       'gaplda': lda_gap(Xs, ys, Xr, yr)}
                for c in ['LDA', 'QDA', 'NB', 'KNN', 'RF', 'HGB', 'MLP']:
                    vals = [row[f'd_r{r}'] for row in o['rows']
                            if row['clf'] == c and f'd_r{r}' in row]
                    if not vals:
                        continue
                    cells.append({'ds': ds, 'lv': lv, 'clf': c, 'regime': r,
                                  'n': int(m.sum()), 'dacc': float(np.mean(vals)), **ind})
    json.dump(cells, open('../results/indicators.json', 'w'), indent=1)
    print(f'{len(cells)} cells')
    a = np.array([c['dacc'] for c in cells])
    print(f'{"indicator":10s} {"rho":>7s} {"p":>10s} {"precision":>10s} {"recall":>8s} {"rule acc":>9s}')
    base_rate = (a > 0).mean()
    for name in ('dswc', 'df1', 'dn2', 'gaplda'):
        d = np.array([c[name] for c in cells], float)
        ok = ~np.isnan(d)
        rho, pv = spearmanr(d[ok], a[ok])
        pred, truth = d[ok] > 0, a[ok] > 0
        tp = (pred & truth).sum(); fp = (pred & ~truth).sum(); fn = (~pred & truth).sum()
        tn = (~pred & ~truth).sum()
        prec = tp / (tp + fp) if tp + fp else float('nan')
        rec = tp / (tp + fn) if tp + fn else float('nan')
        print(f'{name:10s} {rho:+7.3f} {pv:10.3g} {prec:10.3f} {rec:8.3f} {(tp + tn) / ok.sum():9.3f}')
    print(f'base rate (always separate): precision {base_rate:.3f}, recall 1.000, accuracy {base_rate:.3f}')


if __name__ == '__main__':
    main()
