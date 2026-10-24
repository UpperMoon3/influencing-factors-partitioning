"""Prequential comparison with drift-adaptation methods.

Every method processes the stream in chronological order, one day at a time:
the records of a day are predicted with the current model(s), then their labels
are revealed and the method may update itself.  All methods therefore see the
same records, in the same order, with the same label delay of one day, and are
scored on the same records.  The first WARMUP_DAYS days are used only for
training.

Batch learners (LDA, QDA, NB, KNN) are combined with seven update policies:
  pooled        all past records, retrained every RETRAIN_EVERY days
  feature       as pooled, with the regime appended as a one-hot input
  window        the last WINDOW_DAYS days, retrained every RETRAIN_EVERY days
  ddm           records since the last drift signalled by DDM on the error
                stream; retrained at every drift and every RETRAIN_EVERY days
  adwin         records inside the current ADWIN window on the error stream;
                retrained at every change and every RETRAIN_EVERY days
  partitioned   one model per calendar regime on all past records of that
                regime (the proposed method), retrained every RETRAIN_EVERY days
  part+adwin    as partitioned, with a separate ADWIN window per regime
Incremental learners from river are run with the same day-batched protocol:
  ARF, ARF + regime feature, HAT.

Requires river (tested with 0.22.0).  Usage: python run_stream.py
"""
import json
import time
import numpy as np
from scipy.stats import wilcoxon
from river import drift, forest, tree

import datasets as D
import partitions as P
import protocol as PR

WARMUP_DAYS = 56
RETRAIN_EVERY = 7
WINDOW_DAYS = 56
MIN_REGIME = 25          # fewer training records than this -> pooled model
SEED = PR.RANDOM_STATE
BATCH = ('LDA', 'QDA', 'NB', 'KNN')
POLICIES = ('pooled', 'feature', 'window', 'ddm', 'adwin', 'partitioned', 'part+adwin')


class Model:
    """A fitted classifier with a fallback for degenerate training sets."""

    def __init__(self, make, X, y):
        self.const = None
        classes = np.unique(y)
        if len(classes) < 2 or len(y) < 5:
            self.const = classes[0] if len(classes) else 0
        else:
            self.m = make().fit(X, y)

    def predict(self, X):
        if self.const is not None:
            return np.full(len(X), self.const)
        return self.m.predict(X)


def day_index(dates):
    _, inv = np.unique(np.array([d.toordinal() for d in dates]), return_inverse=True)
    return inv


def run_batch(X, y, regime, day, make, policy):
    n_reg = int(regime.max()) + 1
    oh = PR.onehot(regime, n_reg)
    Xf = np.hstack([X, oh])
    days = np.unique(day)
    first = {int(d): int(np.flatnonzero(day == d)[0]) for d in days}   # days are consecutive
    pred = np.full(len(y), -1)
    n_fit = 0
    n_det = 0
    det = drift.binary.DDM() if policy == 'ddm' else drift.ADWIN() if policy == 'adwin' else None
    det_r = {r: drift.ADWIN() for r in range(n_reg)} if policy == 'part+adwin' else None
    start = 0                       # beginning of the current concept (ddm, adwin)
    warn_at = None
    start_r = {r: 0 for r in range(n_reg)}
    models = {}
    pooled = None

    def fit(now):
        nonlocal n_fit, pooled, models
        past = np.arange(now)
        if policy in ('pooled', 'partitioned', 'part+adwin'):
            pooled = Model(make, X[past], y[past]); n_fit += 1
        if policy == 'pooled':
            models = {'all': pooled}
        elif policy == 'feature':
            models = {'all': Model(make, Xf[past], y[past])}; n_fit += 1
        elif policy == 'window':
            lo = first[max(int(days[0]), int(day[now - 1]) - WINDOW_DAYS + 1)]
            models = {'all': Model(make, X[lo:now], y[lo:now])}; n_fit += 1
        elif policy in ('ddm', 'adwin'):
            lo = min(start, first[max(int(days[0]), int(day[now - 1]) - 1)])   # at least two days
            models = {'all': Model(make, X[lo:now], y[lo:now])}; n_fit += 1
        else:
            models = {}
            for r in range(n_reg):
                lo = start_r[r] if policy == 'part+adwin' else 0
                idx = past[(regime[past] == r) & (past >= lo)]
                if len(idx) < MIN_REGIME and policy == 'part+adwin':
                    idx = past[regime[past] == r]          # window too short: whole history
                if len(idx) >= MIN_REGIME:
                    models[r] = Model(make, X[idx], y[idx]); n_fit += 1

    fit(first[days[WARMUP_DAYS]])
    since = 0
    for d in days[WARMUP_DAYS:]:
        te = np.flatnonzero(day == d)
        if policy in ('partitioned', 'part+adwin'):
            for r in np.unique(regime[te]):
                m = te[regime[te] == r]
                pred[m] = (models.get(r) or pooled).predict(X[m])
        elif policy == 'feature':
            pred[te] = models['all'].predict(Xf[te])
        else:
            pred[te] = models['all'].predict(X[te])
        refit = False
        err = pred[te] != y[te]
        if det is not None:
            for i, e in zip(te, err):
                det.update(bool(e))
                if policy == 'ddm' and det.warning_detected and warn_at is None:
                    warn_at = i
                if det.drift_detected:
                    n_det += 1
                    if policy == 'ddm':
                        start = warn_at if warn_at is not None else i
                        warn_at = None
                    else:
                        start = i + 1 - int(det.width)
                    refit = True
        if det_r is not None:
            for i, e in zip(te, err):
                a = det_r[regime[i]]
                a.update(bool(e))
                if a.drift_detected:
                    n_det += 1
                    start_r[regime[i]] = i + 1 - int(a.width)
                    refit = True
        since += 1
        if since >= RETRAIN_EVERY or refit:
            fit(te[-1] + 1)
            since = 0
    return pred, n_fit, n_det


def run_river(X, y, regime, day, kind):
    n_reg = int(regime.max()) + 1
    if kind == 'HAT':
        model = tree.HoeffdingAdaptiveTreeClassifier(seed=SEED)
    else:
        model = forest.ARFClassifier(n_models=10, seed=SEED)
    use_reg = kind == 'ARF + feature'

    def row(i):
        x = {f'x{j}': float(v) for j, v in enumerate(X[i])}
        if use_reg:
            for r in range(n_reg):
                x[f'r{r}'] = float(regime[i] == r)
        return x

    days = np.unique(day)
    pred = np.full(len(y), -1)
    warm_end = int(np.flatnonzero(day == days[WARMUP_DAYS])[0])
    for i in range(warm_end):
        model.learn_one(row(i), int(y[i]))
    for d in days[WARMUP_DAYS:]:
        te = np.flatnonzero(day == d)
        rows = [row(i) for i in te]
        for i, x in zip(te, rows):
            p = model.predict_one(x)
            pred[i] = int(p) if p is not None else 0
        for i, x in zip(te, rows):
            model.learn_one(x, int(y[i]))
    return pred


def monthly(pred, y, dates, mask):
    key = np.array([d.year * 12 + d.month for d in dates])
    out = []
    for k in np.unique(key[mask]):
        m = mask & (key == k)
        out.append(float((pred[m] == y[m]).mean() * 100))
    return out


def streams():
    X, hour, dates = D.load_powersupply()
    lv, _ = P.powersupply_levels(dates)
    yield 'Powersupply', X, D.powersupply_task(hour), lv['L3 + August shutdown'], dates
    Xe, _, de, ye = D.load_elec2()
    lv2, _ = P.elec2_levels(de)
    yield 'Elec2', Xe, ye, lv2['L3 + holiday'], de


if __name__ == '__main__':
    out = {}
    for name, X, y, regime, dates in streams():
        day = day_index(dates)
        mask = day >= np.unique(day)[WARMUP_DAYS]
        res = {}
        for clf in BATCH:
            for pol in POLICIES:
                t0 = time.time()
                pred, n_fit, n_det = run_batch(X, y, regime, day, PR.CLASSIFIERS[clf], pol)
                acc = float((pred[mask] == y[mask]).mean() * 100)
                res[f'{clf}|{pol}'] = {'acc': acc, 'monthly': monthly(pred, y, dates, mask),
                                       'fits': n_fit, 'detections': n_det,
                                       'seconds': round(time.time() - t0, 1)}
                print(f'{name:12s} {clf:4s} {pol:12s} acc {acc:6.2f}  fits {n_fit:5d}  '
                      f'detections {n_det:4d}  {time.time() - t0:6.1f}s', flush=True)
        for kind in ('ARF', 'ARF + feature', 'HAT'):
            t0 = time.time()
            pred = run_river(X, y, regime, day, kind)
            acc = float((pred[mask] == y[mask]).mean() * 100)
            res[f'{kind}|incremental'] = {'acc': acc, 'monthly': monthly(pred, y, dates, mask),
                                          'seconds': round(time.time() - t0, 1)}
            print(f'{name:12s} {kind:13s} incremental  acc {acc:6.2f}  {time.time() - t0:6.1f}s', flush=True)
        # paired tests over months: partitioned against every other policy, per learner
        tests = {}
        for clf in BATCH:
            a = np.array(res[f'{clf}|partitioned']['monthly'])
            for pol in POLICIES:
                if pol == 'partitioned':
                    continue
                b = np.array(res[f'{clf}|{pol}']['monthly'])
                d = a - b
                p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
                tests[f'{clf}|{pol}'] = {'mean_diff': float(d.mean()), 'p': p}
        out[name] = {'n_eval': int(mask.sum()), 'months': len(res[f'{BATCH[0]}|pooled']['monthly']),
                     'n_regimes': int(regime.max() + 1), 'results': res, 'tests': tests}
        json.dump(out, open('../results/stream.json', 'w'), indent=1)
    print('saved ../results/stream.json')
