"""How does the choice of influencing factors affect the result?

Every candidate factor is used alone and in combination, together with a factor
that is deliberately too fine (month x non-working day) and with random
partitions of the same shape as the deepest real partition, drawn by whole
days.  For each factor set the pooled model, the pooled model with the regime as
a feature, and the partitioned model are compared on identical control objects
under the randomised and the chronological protocol of protocol.py.

Usage: python run_factors.py [n_splits]
"""
import json
import sys
import time
import numpy as np

import datasets as D
import partitions as P
import protocol as PR
from run_control import random_partition

N_SPLITS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
ONLY = sys.argv[2] if len(sys.argv) > 2 else None      # run a single stream
OUT = f'../results/factors_{ONLY}.json' if ONLY else '../results/factors.json'
N_RANDOM = 3
rng = np.random.default_rng(2026)


def chrono_light(X, y, regime, dates, n_folds=6):
    """protocol.chronological_run without the per-regime selection rule."""
    n_regimes = int(regime.max()) + 1
    order = np.argsort([d.toordinal() for d in dates], kind='stable')
    X, y, regime = X[order], y[order], regime[order]
    bounds = np.linspace(0, len(X), n_folds + 1).astype(int)
    rows = []
    for k in range(1, n_folds):
        tr, te = np.arange(0, bounds[k]), np.arange(bounds[k], bounds[k + 1])
        Xtr_f = np.hstack([X[tr], PR.onehot(regime[tr], n_regimes)])
        Xte_f = np.hstack([X[te], PR.onehot(regime[te], n_regimes)])
        for cname, make in PR.CLASSIFIERS.items():
            g, _ = PR._fit_predict(make, X[tr], y[tr], X[te])
            f, _ = PR._fit_predict(make, Xtr_f, y[tr], Xte_f)
            q = np.empty(len(te), dtype=y.dtype)
            for r in range(n_regimes):
                mtr, mte = regime[tr] == r, regime[te] == r
                if mte.sum() == 0:
                    continue
                if mtr.sum() < 5:
                    q[mte], _ = PR._fit_predict(make, X[tr], y[tr], X[te][mte])
                else:
                    q[mte], _ = PR._fit_predict(make, X[tr][mtr], y[tr][mtr], X[te][mte])
            rows.append({'rep': k, 'clf': cname, 'global_acc': (g == y[te]).mean() * 100,
                         'feature_acc': (f == y[te]).mean() * 100,
                         'regime_acc': (q == y[te]).mean() * 100})
    return rows


def codes(*arrays):
    key = np.zeros(len(arrays[0]), dtype=np.int64)
    for a in arrays:
        a = np.asarray(a)
        _, inv = np.unique(a, return_inverse=True)
        key = key * (inv.max() + 1) + inv
    return np.unique(key, return_inverse=True)[1].astype(int)


def factor_sets_powersupply(dates):
    f = D.factor_table(dates, 'IT')
    month = np.array([d.month for d in dates])
    season = np.select([np.isin(month, [12, 1, 2]), np.isin(month, [3, 4, 5]),
                        np.isin(month, [6, 7, 8])], [0, 1, 2], default=3)
    nw, aug = f['nonworking'], f['august']
    return {
        'season': codes(season),
        'non-working day': codes(nw),
        'August shutdown': codes(aug),
        'non-working x August': codes(nw, aug),
        'season x non-working (L2)': codes(season, nw),
        'season x non-working x August (L3)': codes(season, nw, aug),
        'month x non-working (too fine)': codes(month, nw),
    }


def factor_sets_elec2(dates):
    f = D.factor_table(dates, 'AU')
    month = np.array([d.month for d in dates])
    season = np.select([np.isin(month, [12, 1, 2]), np.isin(month, [6, 7, 8])], [0, 1], default=2)
    nw, hol = f['nonworking'], f['holiday']
    return {
        'non-working day (L1)': codes(nw),
        'season': codes(season),
        'public holiday': codes(hol),
        'non-working x season (L2)': codes(nw, season),
        'non-working x season x holiday (L3)': codes(nw, season, hol),
        'month x non-working (too fine)': codes(month, nw),
    }


def summarise(rows):
    out = {}
    for c in PR.CLASSIFIERS:
        g = np.array([r['global_acc'] for r in rows if r['clf'] == c])
        f = np.array([r['feature_acc'] for r in rows if r['clf'] == c])
        q = np.array([r['regime_acc'] for r in rows if r['clf'] == c])
        out[c] = {'pool': float(g.mean()), 'feat': float(f.mean()), 'part': float(q.mean()),
                  'd_pool': float((q - g).mean()), 'd_feat': float((q - f).mean())}
    return out


def main():
    X, hour, dates = D.load_powersupply()
    Xe, _, de, ye = D.load_elec2()
    jobs = (('Powersupply', X, D.powersupply_task(hour), dates, factor_sets_powersupply(dates),
             'season x non-working x August (L3)'),
            ('Elec2', Xe, ye, de, factor_sets_elec2(de), 'non-working x season x holiday (L3)'))
    res = {}
    for name, Xd, yd, dd, sets, deepest in jobs:
        if ONLY and name != ONLY:
            continue
        sizes = np.bincount(sets[deepest])
        for k in range(N_RANDOM):
            sets[f'random, shape of L3 (draw {k + 1})'] = random_partition(dd, len(sizes), sizes, rng)
        res[name] = {}
        for fname, reg in sets.items():
            t0 = time.time()
            rows = PR.paired_run(Xd, yd, reg, name, n_splits=N_SPLITS,
                                 methods=('global', 'feature', 'regime'), verbose=False)
            crow = chrono_light(Xd, yd, reg, dd)
            res[name][fname] = {'M': int(reg.max() + 1), 'min_size': int(np.bincount(reg).min()),
                                'random': summarise(rows), 'chrono': summarise(crow)}
            s, c = res[name][fname]['random'], res[name][fname]['chrono']
            print(f'{name:12s} {fname:38s} M={reg.max() + 1:3d}  '
                  f'rand d_pool {np.mean([v["d_pool"] for v in s.values()]):+.2f} '
                  f'd_feat {np.mean([v["d_feat"] for v in s.values()]):+.2f} | '
                  f'chrono d_pool {np.mean([v["d_pool"] for v in c.values()]):+.2f} '
                  f'd_feat {np.mean([v["d_feat"] for v in c.values()]):+.2f}  '
                  f'{time.time() - t0:.0f}s', flush=True)
            json.dump(res, open(OUT, 'w'), indent=1)
    print('saved', OUT)


if __name__ == '__main__':
    main()
