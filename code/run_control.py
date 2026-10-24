"""Negative control: the same partitioning machinery driven by a random factor.

The regimes are replaced by a random partition with the same number of parts and
the same block structure --- whole days are assigned at random --- so the only
thing that changes is whether the partition corresponds to anything real.  The
analysis of Section 3 predicts a heterogeneity gap of zero and therefore a small
loss, which is what separates a genuine factor from partitioning as such.
"""
import json, sys, time
import numpy as np
import datasets as D, partitions as P, protocol as PR

N_SPLITS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
rng = np.random.default_rng(7)


def random_partition(dates, n_regimes, sizes_like, rng):
    """Assign whole days at random so that regime sizes match the real ones."""
    days = np.array([d.toordinal() for d in dates])
    uday = np.unique(days)
    target = np.array(sizes_like, float); target /= target.sum()
    labels = rng.choice(n_regimes, size=len(uday), p=target)
    lut = dict(zip(uday, labels))
    return np.array([lut[d] for d in days])


def main():
    res = {}
    X, hour, dates = D.load_powersupply()
    y = D.powersupply_task(hour)
    lv, _ = P.powersupply_levels(dates)
    Xe, _, de, ye = D.load_elec2()
    lv2, _ = P.elec2_levels(de)
    
    for name, Xd, yd, dd, levels in (('Powersupply', X, y, dates, lv), ('Elec2', Xe, ye, de, lv2)):
        real = list(levels.values())[-1]
        sizes = np.bincount(real)
        fake = random_partition(dd, len(sizes), sizes, rng)
        print(f'== {name}: real M={len(sizes)}, random M={fake.max() + 1}, '
              f'sizes real {sorted(sizes)[:4]}... random {sorted(np.bincount(fake))[:4]}...', flush=True)
        t0 = time.time()
        rows = PR.paired_run(Xd, yd, fake, name, n_splits=N_SPLITS,
                             methods=('global', 'feature', 'regime'), verbose=False)
        sw = {'pooled': PR.swc(Xd, yd)}
        for r in range(int(fake.max()) + 1):
            m = fake == r
            sw[f'r{r}'] = PR.swc(Xd[m], yd[m]); sw[f'n{r}'] = int(m.sum())
        res[name] = {'rows': rows, 'swc': sw, 'n_regimes': int(fake.max() + 1),
                     'seconds': round(time.time() - t0, 1)}
        json.dump(res, open('../results/control.json', 'w'), default=str)
        for c in PR.CLASSIFIERS:
            g = np.array([r['global_acc'] for r in rows if r['clf'] == c])
            q = np.array([r['regime_acc'] for r in rows if r['clf'] == c])
            print(f'   {c:4s} pooled {g.mean():6.2f}  random-partition {q.mean():6.2f}  '
                  f'delta {(q - g).mean():+.2f}', flush=True)
    print('saved ../results/control.json')


if __name__ == '__main__':
    main()
