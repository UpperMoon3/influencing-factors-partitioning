"""Factor-induced partitions at increasing levels of refinement."""
import numpy as np
import datasets as D


def _codes(*bools):
    """Combine boolean factor conditions into consecutive regime indices."""
    key = np.zeros(len(bools[0]), dtype=np.int64)
    for b in bools:
        key = key * 2 + b.astype(np.int64)
    _, inv = np.unique(key, return_inverse=True)
    return inv.astype(int)


def powersupply_levels(dates):
    f = D.factor_table(dates, 'IT')
    month = np.array([d.month for d in dates])
    season = np.select([np.isin(month, [12, 1, 2]), np.isin(month, [3, 4, 5]),
                        np.isin(month, [6, 7, 8])], [0, 1, 2], default=3)
    lv = {}
    lv['L0 pooled'] = np.zeros(len(dates), dtype=int)
    lv['L1 season'] = season.astype(int)
    lv['L2 season x non-working'] = _codes(season == 0, season == 1, season == 2,
                                           f['nonworking'])
    lv['L3 + August shutdown'] = _codes(season == 0, season == 1, season == 2,
                                        f['nonworking'], f['august'])
    return lv, f


def elec2_levels(dates):
    f = D.factor_table(dates, 'AU')
    month = np.array([d.month for d in dates])
    season = np.select([np.isin(month, [12, 1, 2]), np.isin(month, [6, 7, 8])],
                       [0, 1], default=2)
    lv = {}
    lv['L0 pooled'] = np.zeros(len(dates), dtype=int)
    lv['L1 non-working'] = f['nonworking'].astype(int)
    lv['L2 + season'] = _codes(f['nonworking'], season == 0, season == 1)
    lv['L3 + holiday'] = _codes(f['nonworking'], season == 0, season == 1, f['holiday'])
    return lv, f
