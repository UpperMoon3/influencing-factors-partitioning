"""Additional simulations with the generator of the simulation study (Section 4).

part A  Over-partitioning at a fixed training budget.  The data have two true
        regimes (shift delta = 0.5); the partition used for training splits each
        true regime into K arbitrary equal parts, so M = 2K parts of which only
        the first split carries information.  The size rule of Algorithm 1
        (merge parts with fewer than p_min training objects into their parent,
        here the true regime, and then into the pooled sample) is applied for
        several p_min.

part B  Validity of the a priori indicators over many configurations.  Random
        configurations vary the shift along the class axis, a shift that moves
        only one class (a change of concept), the class balance of the second
        regime, and the training size.  For every configuration the silhouette
        gain and a pilot estimate of the heterogeneity gap are compared with the
        accuracy gain that partitioning actually delivers on independent test
        data.

Usage: python sim_extra.py
"""
import json
import numpy as np
from scipy.stats import spearmanr, wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score

SIG = 0.35 * np.array([[1, .9], [.9, 1]])
CH = np.linalg.cholesky(SIG)
N_TEST = 2000


def make(name, n):
    if name == 'LDA':
        return LDA()
    if name == 'QDA':
        return QDA(reg_param=1e-3)
    if name == 'NB':
        return GaussianNB()
    return KNeighborsClassifier(n_neighbors=max(1, min(15, n)))


CLFS = ('LDA', 'QDA', 'NB', 'KNN')


def fit_predict(name, Xtr, ytr, Xte):
    cls, cnt = np.unique(ytr, return_counts=True)
    if len(cls) < 2 or len(ytr) < 4 or cnt.min() < 2:     # degenerate part: majority class
        return np.full(len(Xte), cls[np.argmax(cnt)] if len(cls) else 0)
    return make(name, len(ytr)).fit(Xtr, ytr).predict(Xte)


def draw(rng, n, mu0, mu1, p1):
    y = (rng.random(n) < p1).astype(int)
    if y.min() == y.max():                      # keep both classes present
        y[: max(1, n // 10)] = 1 - y[0]
    z = rng.standard_normal((n, 2)) @ CH.T
    mu = np.where(y[:, None] == 1, mu1, mu0)
    return z + mu, y


# ---------------------------------------------------------------- part A
def part_a(reps=50):
    delta = 0.5
    out = []
    for p in (200, 1000, 4000):
        for K in (1, 2, 4, 8, 16):
            M = 2 * K
            res = {c: {'pool': [], 'true2': [], 'part': [], 'pmin50': [], 'pmin100': []} for c in CLFS}
            for rep in range(reps):
                rng = np.random.default_rng(10_000 * p + 100 * K + rep)
                Xtr, ytr, rtr, Xte, yte, rte = [], [], [], [], [], []
                for r in (0, 1):
                    mu0 = np.full(2, 1.6 * delta * r)
                    mu1 = mu0 + 1.6
                    a, b = draw(rng, p // 2, mu0, mu1, 0.5)
                    c, d = draw(rng, N_TEST, mu0, mu1, 0.5)
                    Xtr.append(a); ytr.append(b); rtr.append(np.full(len(b), r))
                    Xte.append(c); yte.append(d); rte.append(np.full(len(d), r))
                Xtr, ytr, rtr = np.vstack(Xtr), np.concatenate(ytr), np.concatenate(rtr)
                Xte, yte, rte = np.vstack(Xte), np.concatenate(yte), np.concatenate(rte)
                # arbitrary sub-parts inside every true regime
                sub_tr = rng.integers(0, K, len(ytr))
                sub_te = rng.integers(0, K, len(yte))
                part_tr, part_te = rtr * K + sub_tr, rte * K + sub_te
                for c in CLFS:
                    g = fit_predict(c, Xtr, ytr, Xte)
                    t2 = np.empty_like(yte)
                    for r in (0, 1):
                        t2[rte == r] = fit_predict(c, Xtr[rtr == r], ytr[rtr == r], Xte[rte == r])
                    preds = {}
                    for key, pmin in (('part', 0), ('pmin50', 50), ('pmin100', 100)):
                        q = np.empty_like(yte)
                        for j in range(M):
                            mte = part_te == j
                            mtr = part_tr == j
                            if mtr.sum() >= pmin:
                                q[mte] = fit_predict(c, Xtr[mtr], ytr[mtr], Xte[mte])
                            elif (rtr == j // K).sum() >= pmin:
                                q[mte] = t2[mte]
                            else:
                                q[mte] = g[mte]
                        preds[key] = q
                    res[c]['pool'].append((g == yte).mean() * 100)
                    res[c]['true2'].append((t2 == yte).mean() * 100)
                    for key, q in preds.items():
                        res[c][key].append((q == yte).mean() * 100)
            row = {'p': p, 'M': M, 'n_per_part': p // M}
            for key in ('pool', 'true2', 'part', 'pmin50', 'pmin100'):
                row[key] = float(np.mean([np.mean(res[c][key]) for c in CLFS]))
            d = np.array([np.mean(res[c]['part']) - np.mean(res[c]['pool']) for c in CLFS])
            row['d_part_pool_by_clf'] = d.tolist()
            sig = 0
            for c in CLFS:
                dd = np.array(res[c]['part']) - np.array(res[c]['true2'])
                if np.any(dd != 0) and wilcoxon(dd).pvalue < 0.05 and dd.mean() < 0:
                    sig += 1
            row['n_clf_sig_loss_vs_true'] = sig
            out.append(row)
            print('A p=%4d M=%2d n/part=%4d pool %.2f true %.2f part %.2f pmin50 %.2f pmin100 %.2f'
                  % (p, M, p // M, row['pool'], row['true2'], row['part'], row['pmin50'], row['pmin100']),
                  flush=True)
    return out


# ---------------------------------------------------------------- part B
def silhouette(X, y):
    if len(np.unique(y)) < 2:
        return np.nan
    return float(silhouette_score(X, y))


def pilot_gap(Xp, yp, Xr, yr):
    if len(np.unique(yr)) < 2 or len(yr) < 30:
        return np.nan
    err_pool = 1.0 - LDA().fit(Xp, yp).score(Xr, yr)
    cv = min(3, int(np.bincount(yr).min()))
    if cv < 2:
        return np.nan
    err_reg = 1.0 - cross_val_score(LDA(), Xr, yr, cv=cv).mean()
    return float((err_pool - err_reg) * 100)


def part_b(n_configs=300, reps=5):
    rng0 = np.random.default_rng(424242)
    cells = []
    for k in range(n_configs):
        cfg = {'delta': float(rng0.uniform(0, 1.2)),          # shift of both classes
               'concept': float(rng0.uniform(0, 1.0)),        # shift of class 1 only, across the class axis
               'p1_regime2': float(rng0.choice([0.5, 0.5, 0.3, 0.7, 0.2, 0.8])),
               'n': int(rng0.choice([20, 50, 200, 1000]))}
        dswc, gap, gain = [], [], []
        for rep in range(reps):
            rng = np.random.default_rng(1_000_000 + 10 * k + rep)
            mus = []
            for r in (0, 1):
                mu0 = np.full(2, 1.6 * cfg['delta'] * r)
                mu1 = mu0 + 1.6 + (np.array([1.0, -1.0]) * 0.8 * cfg['concept'] * r)
                mus.append((mu0, mu1, 0.5 if r == 0 else cfg['p1_regime2']))
            tr = [draw(rng, cfg['n'], *m) for m in mus]
            te = [draw(rng, N_TEST, *m) for m in mus]
            Xp = np.vstack([t[0] for t in tr]); yp = np.concatenate([t[1] for t in tr])
            Xt = np.vstack([t[0] for t in te]); yt = np.concatenate([t[1] for t in te])
            s_pool = silhouette(Xp, yp)
            # indicator of separating regime 2 from the pooled sample
            dswc.append(silhouette(*tr[1]) - s_pool)
            gap.append(pilot_gap(Xp, yp, *tr[1]))
            m2 = np.arange(len(yt)) >= N_TEST          # test objects of regime 2
            g_all = []
            for c in CLFS:
                g = fit_predict(c, Xp, yp, Xt[m2])
                q = fit_predict(c, *tr[1], Xt[m2])
                g_all.append(((q == yt[m2]).mean() - (g == yt[m2]).mean()) * 100)
            gain.append(np.mean(g_all))
        cells.append({**cfg, 'dswc': float(np.nanmean(dswc)), 'gap': float(np.nanmean(gap)),
                      'gain': float(np.mean(gain))})
        if (k + 1) % 50 == 0:
            print(f'B {k + 1}/{n_configs} configurations', flush=True)
    a = np.array([c['gain'] for c in cells])
    summary = {}
    for name in ('dswc', 'gap'):
        d = np.array([c[name] for c in cells])
        ok = ~np.isnan(d)
        rho, pv = spearmanr(d[ok], a[ok])
        pred, truth = d[ok] > 0, a[ok] > 0
        tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum()); fn = int((~pred & truth).sum())
        tn = int((~pred & ~truth).sum())
        summary[name] = {'rho': float(rho), 'p': float(pv), 'precision': tp / max(tp + fp, 1),
                         'recall': tp / max(tp + fn, 1), 'accuracy': (tp + tn) / int(ok.sum()), 'n': int(ok.sum())}
    summary['always'] = {'precision': float((a > 0).mean()), 'recall': 1.0,
                         'accuracy': float((a > 0).mean())}
    for k, v in summary.items():
        print('B', k, {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()}, flush=True)
    return cells, summary


if __name__ == '__main__':
    A = part_a()
    cells, summary = part_b()
    json.dump({'part_a': A, 'part_b_cells': cells, 'part_b_summary': summary},
              open('../results/sim_extra.json', 'w'), indent=1)
    print('saved ../results/sim_extra.json')
