"""The negative control under the chronological protocol."""
import json
import numpy as np
import datasets as D, partitions as P, protocol as PR


def random_partition(dates, n_regimes, sizes_like, rng):
    """Assign whole days at random so that regime sizes match the real ones."""
    days = np.array([d.toordinal() for d in dates])
    uday = np.unique(days)
    target = np.array(sizes_like, float); target /= target.sum()
    labels = rng.choice(n_regimes, size=len(uday), p=target)
    lut = dict(zip(uday, labels))
    return np.array([lut[d] for d in days])


rng = np.random.default_rng(7)
out = {}
X, hour, dates = D.load_powersupply()
sets = [('Powersupply', X, D.powersupply_task(hour), dates,
         list(P.powersupply_levels(dates)[0].values())[-1])]
Xe, _, de, ye = D.load_elec2()
sets.append(('Elec2', Xe, ye, de, list(P.elec2_levels(de)[0].values())[-1]))

for name, Xd, yd, dd, real in sets:
    fake = random_partition(dd, int(real.max()) + 1, np.bincount(real), rng)
    rows = PR.chronological_run(Xd, yd, fake, dd, name, methods=('global', 'feature', 'regime'))
    per = {}
    for c in PR.CLASSIFIERS:
        d = [r['regime_acc'] - r['global_acc'] for r in rows if r['clf'] == c]
        per[c] = float(np.mean(d))
    out[name] = {'per_clf': per, 'mean': float(np.mean(list(per.values())))}
    print(name, {k: round(v, 2) for k, v in per.items()}, 'mean %.2f' % out[name]['mean'], flush=True)
json.dump(out, open('../results/control_chrono.json', 'w'), indent=1)
print('saved')
