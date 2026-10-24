"""Training, prediction and indicator cost of the compared methods."""
import json, time
import numpy as np
import datasets as D, partitions as P, protocol as PR
from sklearn.model_selection import train_test_split

out = {}
for name, (X, y, lv) in {
        'Powersupply': (lambda: (lambda X, h, d: (X, D.powersupply_task(h), P.powersupply_levels(d)[0]))(*D.load_powersupply()))(),
        'Elec2': (lambda: (lambda X, dd, d, y: (X, y, P.elec2_levels(d)[0]))(*D.load_elec2()))()}.items():
    reg = list(lv.values())[-1]
    Xtr, Xte, ytr, yte, rtr, rte = train_test_split(X, y, reg, test_size=0.3, random_state=0, stratify=y)
    M = int(reg.max()) + 1
    res = {}
    for c, mk in PR.CLASSIFIERS.items():
        t0 = time.perf_counter(); m = mk().fit(Xtr, ytr); t_pool = time.perf_counter() - t0
        t0 = time.perf_counter(); m.predict(Xte); q_pool = time.perf_counter() - t0
        t0 = time.perf_counter()
        for r in range(M):
            mm = rtr == r
            if mm.sum() >= 5:
                mk().fit(Xtr[mm], ytr[mm])
        t_part = time.perf_counter() - t0
        res[c] = {'fit_pooled': t_pool, 'fit_partitioned': t_part, 'predict_pooled': q_pool,
                  'ratio': t_part / t_pool if t_pool else None}
    t0 = time.perf_counter(); PR.swc(X, y); t_swc = time.perf_counter() - t0
    out[name] = {'M': M, 'n_train': int(len(Xtr)), 'clf': res, 'swc_seconds': t_swc}
    print(name, 'M', M, 'SWC %.2fs' % t_swc,
          {c: round(v['ratio'], 2) for c, v in res.items()})
json.dump(out, open('../results/timing.json', 'w'), indent=1)
