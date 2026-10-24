"""Spot check: re-run part of run_main.py and compare with the stored results.

The first two randomised splits and the whole chronological protocol at the
deepest level of each stream are recomputed and compared record by record
with ../results/main.json.
"""
import json
import numpy as np
import datasets as D, partitions as P, protocol as PR

ORIG = json.load(open('../results/main.json'))
KEYS = ('global_acc', 'feature_acc', 'regime_acc')


def compare(name, new_rows, old_rows):
    old = {(r['rep'], r['clf']): r for r in old_rows}
    worst = 0.0
    n = 0
    for r in new_rows:
        o = old.get((r['rep'], r['clf']))
        if o is None:
            continue
        for k in KEYS + ('sel_acc',):
            if k in r and k in o:
                worst = max(worst, abs(float(r[k]) - float(o[k])))
                n += 1
    print(f'{name}: compared {n} values, max abs difference {worst:.6f}', flush=True)


X, hour, dates = D.load_powersupply()
lv, _ = P.powersupply_levels(dates)
Xe, _, de, ye = D.load_elec2()
lv2, _ = P.elec2_levels(de)
for name, Xd, y, levels, dd in (('Powersupply', X, D.powersupply_task(hour), lv, dates),
                                ('Elec2', Xe, ye, lv2, de)):
    lname = [k for k in levels if k.startswith('L3')][0]
    reg = levels[lname]
    stored = ORIG[name]['levels'][lname]
    rows = PR.paired_run(Xd, y, reg, name, n_splits=2, verbose=False)
    compare(f'{name} {lname} randomised (2 splits)', rows, stored['rows'])
    crows = PR.chronological_run(Xd, y, reg, dd, name)
    compare(f'{name} {lname} chronological', crows, stored['chrono'])
