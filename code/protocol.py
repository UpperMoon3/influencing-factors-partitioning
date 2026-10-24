"""Paired evaluation protocol.

Every method is trained on the same training split and evaluated on exactly the
same control objects, so the differences are paired by construction.  This is
the property that a comparison of quality indicators computed on different
subsamples does not have.
"""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, silhouette_score
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 20260920

CLASSIFIERS = {
    'LDA': lambda: LinearDiscriminantAnalysis(),
    'QDA': lambda: QuadraticDiscriminantAnalysis(reg_param=1e-3),
    'NB':  lambda: GaussianNB(),
    'KNN': lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
    'RF':  lambda: RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                          random_state=RANDOM_STATE),
    'HGB': lambda: HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    'MLP': lambda: make_pipeline(StandardScaler(),
                                 MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                                               early_stopping=True,
                                               random_state=RANDOM_STATE)),
}


def _fit_predict(make, Xtr, ytr, Xte):
    """Fit and predict, falling back to the majority class for degenerate splits."""
    classes = np.unique(ytr)
    if len(classes) < 2 or len(ytr) < 5:
        p = np.full(len(Xte), classes[0] if len(classes) else 0)
        return p, np.full(len(Xte), 0.5)
    m = make().fit(Xtr, ytr)
    pred = m.predict(Xte)
    try:
        proba = m.predict_proba(Xte)[:, 1]
    except Exception:
        proba = pred.astype(float)
    return pred, proba


def _scores(y, pred, proba):
    out = {'acc': accuracy_score(y, pred) * 100,
           'f1': f1_score(y, pred, average='macro')}
    if len(np.unique(y)) == 2:
        try:
            out['auc'] = roc_auc_score(y, proba)
        except ValueError:
            out['auc'] = np.nan
    return out


def onehot(regime, n_regimes):
    M = np.zeros((len(regime), n_regimes))
    M[np.arange(len(regime)), regime] = 1.0
    return M


def paired_run(X, y, regime, name, n_splits=20, test_size=0.3, seed=RANDOM_STATE,
               methods=('global', 'feature', 'regime', 'regime_sel'), verbose=True):
    """Repeated stratified splits; all methods scored on identical control sets."""
    n_regimes = int(regime.max()) + 1
    strat = y.astype(str) + '_' + regime.astype(str)
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    rows = []
    for rep, (tr, te) in enumerate(sss.split(X, strat)):
        Xtr, ytr, rtr = X[tr], y[tr], regime[tr]
        Xte, yte, rte = X[te], y[te], regime[te]
        Xtr_f = np.hstack([Xtr, onehot(rtr, n_regimes)])
        Xte_f = np.hstack([Xte, onehot(rte, n_regimes)])
        for cname, make in CLASSIFIERS.items():
            rec = {'dataset': name, 'rep': rep, 'clf': cname}
            if 'global' in methods:
                p, q = _fit_predict(make, Xtr, ytr, Xte)
                for k, v in _scores(yte, p, q).items():
                    rec['global_' + k] = v
                rec['_gpred'] = p
            if 'feature' in methods:
                p, q = _fit_predict(make, Xtr_f, ytr, Xte_f)
                for k, v in _scores(yte, p, q).items():
                    rec['feature_' + k] = v
            if 'regime' in methods:
                pred = np.empty(len(yte), dtype=y.dtype)
                proba = np.zeros(len(yte))
                for r in range(n_regimes):
                    mtr, mte = rtr == r, rte == r
                    if mte.sum() == 0:
                        continue
                    if mtr.sum() < 5:           # fallback to the pooled model
                        pp, qq = _fit_predict(make, Xtr, ytr, Xte[mte])
                    else:
                        pp, qq = _fit_predict(make, Xtr[mtr], ytr[mtr], Xte[mte])
                    pred[mte], proba[mte] = pp, qq
                for k, v in _scores(yte, pred, proba).items():
                    rec['regime_' + k] = v
                # per-regime paired differences
                if 'global' in methods:
                    for r in range(n_regimes):
                        mte = rte == r
                        if mte.sum() == 0:
                            continue
                        rec[f'd_r{r}'] = (accuracy_score(yte[mte], pred[mte])
                                          - accuracy_score(yte[mte], rec['_gpred'][mte])) * 100
                        rec[f'n_r{r}'] = int(mte.sum())
            if 'regime_sel' in methods and cname == list(CLASSIFIERS)[0]:
                # the selection rule is a property of the partition, not of one
                # classifier, so it is evaluated once per repetition
                pred = np.empty(len(yte), dtype=y.dtype)
                proba = np.zeros(len(yte))
                chosen = {}
                for r in range(n_regimes):
                    mtr, mte = rtr == r, rte == r
                    if mte.sum() == 0:
                        continue
                    if mtr.sum() < 25:
                        pp, qq = _fit_predict(make, Xtr, ytr, Xte[mte])
                        chosen[r] = 'pooled'
                    else:
                        sc = {}
                        for c2, mk2 in CLASSIFIERS.items():
                            try:
                                sc[c2] = cross_val_score(mk2(), Xtr[mtr], ytr[mtr], cv=3,
                                                         n_jobs=1).mean()
                            except Exception:
                                sc[c2] = -1
                        best = max(sc, key=sc.get)
                        chosen[r] = best
                        pp, qq = _fit_predict(CLASSIFIERS[best], Xtr[mtr], ytr[mtr], Xte[mte])
                    pred[mte], proba[mte] = pp, qq
                sel = _scores(yte, pred, proba)
                rec['sel_acc'] = sel['acc']
                rec['sel_choice'] = chosen
            rec.pop('_gpred', None)
            rows.append(rec)
        if verbose:
            print(f'  {name} rep {rep + 1}/{n_splits}', flush=True)
    return rows


def swc(X, y, max_n=4000, seed=RANDOM_STATE):
    """Mean silhouette width over class labels, subsampled for large sets."""
    if len(np.unique(y)) < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    if len(X) > max_n:
        idx = rng.choice(len(X), max_n, replace=False)
        X, y = X[idx], y[idx]
        if len(np.unique(y)) < 2:
            return np.nan
    Xs = StandardScaler().fit_transform(X)
    return float(silhouette_score(Xs, y))


def chronological_run(X, y, regime, dates, name, n_folds=6, methods=('global', 'feature', 'regime', 'regime_sel')):
    """Blocked evaluation in chronological order.

    The stream is cut into consecutive blocks; block k is the control set and
    everything before it is the training set.  This is the deployment situation
    the method is meant for: the regime of a future record is known from the
    calendar, but its data have not been seen.
    """
    n_regimes = int(regime.max()) + 1
    order = np.argsort([d.toordinal() for d in dates], kind='stable')
    X, y, regime = X[order], y[order], regime[order]
    bounds = np.linspace(0, len(X), n_folds + 1).astype(int)
    rows = []
    for k in range(1, n_folds):
        tr = np.arange(0, bounds[k])
        te = np.arange(bounds[k], bounds[k + 1])
        Xtr, ytr, rtr = X[tr], y[tr], regime[tr]
        Xte, yte, rte = X[te], y[te], regime[te]
        Xtr_f = np.hstack([Xtr, onehot(rtr, n_regimes)])
        Xte_f = np.hstack([Xte, onehot(rte, n_regimes)])
        for cname, make in CLASSIFIERS.items():
            rec = {'dataset': name, 'rep': k, 'clf': cname, 'n_train': len(tr), 'n_test': len(te)}
            p, q = _fit_predict(make, Xtr, ytr, Xte)
            for kk, v in _scores(yte, p, q).items():
                rec['global_' + kk] = v
            p, q = _fit_predict(make, Xtr_f, ytr, Xte_f)
            for kk, v in _scores(yte, p, q).items():
                rec['feature_' + kk] = v
            pred = np.empty(len(yte), dtype=y.dtype); proba = np.zeros(len(yte))
            for r in range(n_regimes):
                mtr, mte = rtr == r, rte == r
                if mte.sum() == 0:
                    continue
                if mtr.sum() < 5:
                    pp, qq = _fit_predict(make, Xtr, ytr, Xte[mte])
                else:
                    pp, qq = _fit_predict(make, Xtr[mtr], ytr[mtr], Xte[mte])
                pred[mte], proba[mte] = pp, qq
            for kk, v in _scores(yte, pred, proba).items():
                rec['regime_' + kk] = v
            rows.append(rec)
        # per-regime algorithm selection, estimated on the training part only
        pred = np.empty(len(yte), dtype=y.dtype); proba = np.zeros(len(yte)); chosen = {}
        for r in range(n_regimes):
            mtr, mte = rtr == r, rte == r
            if mte.sum() == 0:
                continue
            if mtr.sum() < 25:
                pp, qq = _fit_predict(CLASSIFIERS['LDA'], Xtr, ytr, Xte[mte]); chosen[r] = 'pooled'
            else:
                sc = {}
                for c2, mk2 in CLASSIFIERS.items():
                    try:
                        sc[c2] = cross_val_score(mk2(), Xtr[mtr], ytr[mtr], cv=3, n_jobs=1).mean()
                    except Exception:
                        sc[c2] = -1
                best = max(sc, key=sc.get); chosen[r] = best
                pp, qq = _fit_predict(CLASSIFIERS[best], Xtr[mtr], ytr[mtr], Xte[mte])
            pred[mte], proba[mte] = pp, qq
        rows.append({'dataset': name, 'rep': k, 'clf': 'SELECTION',
                     'sel_acc': _scores(yte, pred, proba)['acc'], 'sel_choice': chosen})
    return rows
