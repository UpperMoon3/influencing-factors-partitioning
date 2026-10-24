"""Two ways of estimating the per-regime risk that drives the selection rule.

'cv'     k-fold cross-validation over the whole training part of the regime,
         which is what Equation (8) prescribes and what the main experiment uses;
'recent' a hold-out consisting of the most recent quarter of the training part of
         the regime, trained on the rest.

Under a chronological protocol the two disagree, because a regime accumulates its
own drift and the recent hold-out is the part of the past that resembles the
future.
"""
import json, time
import numpy as np
from sklearn.metrics import accuracy_score
import datasets as D, partitions as P, protocol as PR

N_FOLDS = 6
out = {}


def select_and_predict(Xtr, ytr, rtr, Xte, yte, rte, M, mode):
    pred = np.empty(len(yte), dtype=ytr.dtype)
    chosen = {}
    for r in range(M):
        mtr, mte = rtr == r, rte == r
        if mte.sum() == 0:
            continue
        if mtr.sum() < 60:
            pred[mte], _ = PR._fit_predict(PR.CLASSIFIERS['LDA'], Xtr, ytr, Xte[mte])
            chosen[r] = 'pooled'
            continue
        Xr, yr = Xtr[mtr], ytr[mtr]
        sc = {}
        if mode == 'cv':
            from sklearn.model_selection import cross_val_score
            for c, mk in PR.CLASSIFIERS.items():
                try:
                    sc[c] = cross_val_score(mk(), Xr, yr, cv=3, n_jobs=1).mean()
                except Exception:
                    sc[c] = -1
        else:
            cut = int(len(yr) * 0.75)
            Xa, ya, Xb, yb = Xr[:cut], yr[:cut], Xr[cut:], yr[cut:]
            if len(np.unique(ya)) < 2 or len(np.unique(yb)) < 2:
                sc = {c: 0 for c in PR.CLASSIFIERS}
            else:
                for c, mk in PR.CLASSIFIERS.items():
                    try:
                        p, _ = PR._fit_predict(mk, Xa, ya, Xb)
                        sc[c] = accuracy_score(yb, p)
                    except Exception:
                        sc[c] = -1
        best = max(sc, key=sc.get)
        chosen[r] = best
        pred[mte], _ = PR._fit_predict(PR.CLASSIFIERS[best], Xr, yr, Xte[mte])
    return accuracy_score(yte, pred) * 100, chosen


def run(name, X, y, regime, dates):
    M = int(regime.max()) + 1
    order = np.argsort([d.toordinal() for d in dates], kind='stable')
    X, y, regime = X[order], y[order], regime[order]
    bounds = np.linspace(0, len(X), N_FOLDS + 1).astype(int)
    rows = []
    for k in range(1, N_FOLDS):
        tr = np.arange(0, bounds[k]); te = np.arange(bounds[k], bounds[k + 1])
        rec = {'block': k}
        for mode in ('cv', 'recent'):
            a, ch = select_and_predict(X[tr], y[tr], regime[tr], X[te], y[te], regime[te], M, mode)
            rec[mode] = a
            rec[mode + '_choice'] = ch
        # oracle: the single best classifier type on this block, for reference
        best = 0.0
        for c, mk in PR.CLASSIFIERS.items():
            pred = np.empty(len(te), dtype=y.dtype)
            for r in range(M):
                mtr, mte = regime[tr] == r, regime[te] == r
                if mte.sum() == 0:
                    continue
                if mtr.sum() < 5:
                    pp, _ = PR._fit_predict(mk, X[tr], y[tr], X[te][mte])
                else:
                    pp, _ = PR._fit_predict(mk, X[tr][mtr], y[tr][mtr], X[te][mte])
                pred[mte] = pp
            best = max(best, accuracy_score(y[te], pred) * 100)
        rec['oracle'] = best
        rows.append(rec)
        print(f'  {name} block {k}: cv {rec["cv"]:.2f}  recent {rec["recent"]:.2f}  '
              f'best-single {best:.2f}', flush=True)
    return rows


X, hour, dates = D.load_powersupply()
lv, _ = P.powersupply_levels(dates)
out['Powersupply'] = run('Powersupply', X, D.powersupply_task(hour),
                         list(lv.values())[-1], dates)
Xe, _, de, ye = D.load_elec2()
lv2, _ = P.elec2_levels(de)
out['Elec2'] = run('Elec2', Xe, ye, list(lv2.values())[-1], de)
json.dump(out, open('../results/selfix.json', 'w'), default=str, indent=1)
for ds, rows in out.items():
    print(f'{ds}: cv {np.mean([r["cv"] for r in rows]):.2f}  '
          f'recent {np.mean([r["recent"] for r in rows]):.2f}  '
          f'best-single {np.mean([r["oracle"] for r in rows]):.2f}')
