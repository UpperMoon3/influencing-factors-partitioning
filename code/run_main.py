"""Main experiment: paired comparison of the partitioned model with baselines.

Two protocols are reported.  The randomised protocol repeats stratified
train/control splits and is directly comparable with the way sample-formation
methods are usually evaluated.  The chronological protocol trains on the past
and tests on the future, which is the situation the method is designed for.
"""
import json, sys, time
import numpy as np
import datasets as D, partitions as P, protocol as PR

N_SPLITS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
OUT = sys.argv[2] if len(sys.argv) > 2 else '../results/main.json'


def prep():
    X, hour, dates = D.load_powersupply()
    lv, _ = P.powersupply_levels(dates)
    yield 'Powersupply', X, D.powersupply_task(hour), lv, dates
    Xe, _, de, ye = D.load_elec2()
    lv2, _ = P.elec2_levels(de)
    yield 'Elec2', Xe, ye, lv2, de


res = {}
for name, X, y, levels, dates in prep():
    res[name] = {'n': int(len(X)), 'd': int(X.shape[1]),
                 'class_balance': float(np.mean(y)), 'levels': {}}
    for lname, regime in levels.items():
        if lname.startswith('L0'):
            continue
        t0 = time.time()
        print(f'== {name} / {lname}: {regime.max() + 1} regimes', flush=True)
        rows = PR.paired_run(X, y, regime, name, n_splits=N_SPLITS, verbose=False)
        print(f'   random splits done in {time.time() - t0:.0f}s', flush=True)
        t1 = time.time()
        crows = PR.chronological_run(X, y, regime, dates, name)
        print(f'   chronological done in {time.time() - t1:.0f}s', flush=True)
        sw = {'pooled': PR.swc(X, y)}
        for r in range(int(regime.max()) + 1):
            m = regime == r
            sw[f'r{r}'] = PR.swc(X[m], y[m])
            sw[f'n{r}'] = int(m.sum())
        res[name]['levels'][lname] = {'rows': rows, 'chrono': crows, 'swc': sw,
                                      'n_regimes': int(regime.max() + 1),
                                      'seconds': round(time.time() - t0, 1)}
        json.dump(res, open(OUT, 'w'), default=str)
print('saved', OUT)
