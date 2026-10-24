"""Sensitivity to the minimum subsample size.

Two experiments on the nested partitions L1 < L2 < L3 of both streams.

pmin   The size rule of Algorithm 1: a regime of the deepest level whose
       training part has fewer than p_min records is merged into its parent
       (L3 -> L2 -> L1 -> pooled).  Accuracy of the resulting model is reported
       for a grid of p_min values, under the randomised and the chronological
       protocol.  p_min = 5 reproduces the partitioned model of run_main.py and
       p_min = infinity the pooled model.

curve  A learning curve: the training part is reduced to a fraction of its size
       (stratified by regime and class) and the pooled model, the pooled model
       with the regime as a feature, the partitioned model, and the partitioned
       model with the size rule (p_min = 100) are trained on the same reduced
       set and scored on the same, full control set.

Usage: python run_size.py [n_splits_pmin] [n_splits_curve]
"""
import json
import sys
import time
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

import datasets as D
import partitions as P
import protocol as PR

N_PMIN = int(sys.argv[1]) if len(sys.argv) > 1 else 10
N_CURVE = int(sys.argv[2]) if len(sys.argv) > 2 else 5
PMIN_GRID = [5, 250, 500, 1000, 2000, 4000, 8000, 16000, 10 ** 9]
FRACTIONS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 1.0]
CURVE_PMIN = 100
MIN_FIT = 15          # smallest training set on which every candidate can be fitted (KNN, k = 15)
SEED = PR.RANDOM_STATE


def nested(levels):
    """Level arrays from coarse to fine and, for each level, the parent of every regime."""
    names = [k for k in levels if not k.startswith('L0')]
    arrs = [levels[k] for k in names]
    parents = []
    for lo, hi in zip([np.zeros_like(arrs[0])] + arrs[:-1], arrs):
        par = {}
        for r in np.unique(hi):
            ps = np.unique(lo[hi == r])
            assert len(ps) == 1, 'levels are not nested'
            par[int(r)] = int(ps[0])
        parents.append(par)
    return names, arrs, parents


class Fitted:
    def __init__(self, make, X, y):
        cls, cnt = np.unique(y, return_counts=True)
        degenerate = len(cls) < 2 or len(y) < MIN_FIT or cnt.min() < 2
        self.const = (cls[np.argmax(cnt)] if len(cls) else 0) if degenerate else None
        if self.const is None:
            self.m = make().fit(X, y)

    def predict(self, X):
        return np.full(len(X), self.const) if self.const is not None else self.m.predict(X)


def predict_with_rule(make, Xtr, ytr, Xte, arrs_tr, arrs_te, parents, pmins):
    """Accuracy-ready predictions for every p_min in pmins, sharing the fits."""
    L = len(arrs_tr)
    counts = [np.bincount(a, minlength=int(max(a.max(), b.max())) + 1) for a, b in zip(arrs_tr, arrs_te)]
    cache = {}

    def model(level, r):
        key = (level, r)
        if key not in cache:
            if level < 0:
                cache[key] = Fitted(make, Xtr, ytr)
            else:
                m = arrs_tr[level] == r
                cache[key] = Fitted(make, Xtr[m], ytr[m])
        return cache[key]

    preds = {}
    deep = arrs_te[-1]
    for pmin in pmins:
        out = np.empty(len(Xte), dtype=ytr.dtype)
        for r in np.unique(deep):
            sel = deep == r
            level, reg = L - 1, int(r)
            while level >= 0 and counts[level][reg] < pmin:
                reg = parents[level][reg]
                level -= 1
            out[sel] = model(level, reg if level >= 0 else 0).predict(Xte[sel])
        preds[pmin] = out
    return preds


def pmin_randomised(X, y, arrs, parents):
    strat = y.astype(str) + '_' + arrs[-1].astype(str)
    sss = StratifiedShuffleSplit(n_splits=N_PMIN, test_size=0.3, random_state=SEED)
    acc = {c: {p: [] for p in PMIN_GRID} for c in PR.CLASSIFIERS}
    for tr, te in sss.split(X, strat):
        for c, make in PR.CLASSIFIERS.items():
            pr = predict_with_rule(make, X[tr], y[tr], X[te], [a[tr] for a in arrs],
                                   [a[te] for a in arrs], parents, PMIN_GRID)
            for p, v in pr.items():
                acc[c][p].append(float((v == y[te]).mean() * 100))
    return acc


def pmin_chrono(X, y, arrs, parents, dates, n_folds=6):
    order = np.argsort([d.toordinal() for d in dates], kind='stable')
    X, y, arrs = X[order], y[order], [a[order] for a in arrs]
    bounds = np.linspace(0, len(X), n_folds + 1).astype(int)
    acc = {c: {p: [] for p in PMIN_GRID} for c in PR.CLASSIFIERS}
    for k in range(1, n_folds):
        tr, te = np.arange(0, bounds[k]), np.arange(bounds[k], bounds[k + 1])
        for c, make in PR.CLASSIFIERS.items():
            pr = predict_with_rule(make, X[tr], y[tr], X[te], [a[tr] for a in arrs],
                                   [a[te] for a in arrs], parents, PMIN_GRID)
            for p, v in pr.items():
                acc[c][p].append(float((v == y[te]).mean() * 100))
    return acc


def curve(X, y, arrs, parents):
    deep = arrs[-1]
    n_reg = int(deep.max()) + 1
    strat = y.astype(str) + '_' + deep.astype(str)
    sss = StratifiedShuffleSplit(n_splits=N_CURVE, test_size=0.3, random_state=SEED)
    rng = np.random.default_rng(SEED)
    out = {c: {f: {'pool': [], 'feat': [], 'part': [], 'part_pmin': [], 'n_min': []}
               for f in FRACTIONS} for c in PR.CLASSIFIERS}
    for tr, te in sss.split(X, strat):
        for f in FRACTIONS:
            if f < 1.0:
                sub = []
                for s in np.unique(strat[tr]):
                    idx = tr[strat[tr] == s]
                    k = max(1, int(round(f * len(idx))))
                    sub.append(rng.choice(idx, k, replace=False))
                trf = np.concatenate(sub)
            else:
                trf = tr
            nmin = int(np.bincount(deep[trf], minlength=n_reg).min())
            Xf_tr = np.hstack([X[trf], PR.onehot(deep[trf], n_reg)])
            Xf_te = np.hstack([X[te], PR.onehot(deep[te], n_reg)])
            for c, make in PR.CLASSIFIERS.items():
                g = Fitted(make, X[trf], y[trf]).predict(X[te])
                fe = Fitted(make, Xf_tr, y[trf]).predict(Xf_te)
                pr = predict_with_rule(make, X[trf], y[trf], X[te], [a[trf] for a in arrs],
                                       [a[te] for a in arrs], parents, [MIN_FIT, CURVE_PMIN])
                o = out[c][f]
                o['pool'].append(float((g == y[te]).mean() * 100))
                o['feat'].append(float((fe == y[te]).mean() * 100))
                o['part'].append(float((pr[MIN_FIT] == y[te]).mean() * 100))
                o['part_pmin'].append(float((pr[CURVE_PMIN] == y[te]).mean() * 100))
                o['n_min'].append(nmin)
    return out


def main():
    X, hour, dates = D.load_powersupply()
    lv, _ = P.powersupply_levels(dates)
    Xe, _, de, ye = D.load_elec2()
    lv2, _ = P.elec2_levels(de)
    res = {}
    for name, Xd, yd, dd, levels in (('Powersupply', X, D.powersupply_task(hour), dates, lv),
                                     ('Elec2', Xe, ye, de, lv2)):
        names, arrs, parents = nested(levels)
        t0 = time.time()
        r_rand = pmin_randomised(Xd, yd, arrs, parents)
        r_chr = pmin_chrono(Xd, yd, arrs, parents, dd)
        print(f'== {name}: p_min sweep done in {time.time() - t0:.0f}s', flush=True)
        res[name] = {'levels': names, 'pmin_randomised': {c: {str(p): v for p, v in d.items()} for c, d in r_rand.items()},
                     'pmin_chrono': {c: {str(p): v for p, v in d.items()} for c, d in r_chr.items()}}
        json.dump(res, open('../results/size.json', 'w'), indent=1)
        for c in PR.CLASSIFIERS:
            print('  ', c, ' '.join(f'{p if p < 10**9 else "inf"}:{np.mean(r_rand[c][p]):.2f}/{np.mean(r_chr[c][p]):.2f}'
                                   for p in PMIN_GRID), flush=True)
        t0 = time.time()
        r_curve = curve(Xd, yd, arrs, parents)
        print(f'== {name}: learning curve done in {time.time() - t0:.0f}s', flush=True)
        res[name]['sizes'] = {n: np.bincount(a).tolist() for n, a in zip(names, arrs)}
        res[name]['curve'] = {c: {str(f): v for f, v in d.items()} for c, d in r_curve.items()}
        json.dump(res, open('../results/size.json', 'w'), indent=1)
    print('saved ../results/size.json')


if __name__ == '__main__':
    main()
