"""Recovery and verification of the timestamps of the two streams.

Elec2 is used as a positive control: it carries a day-of-week attribute, so the
recovered calendar can be checked against ground truth.  The same procedure is
then applied to Powersupply, where no ground truth exists and the alignment is
verified against the physical signature of the Italian working calendar.
"""
import json
import datetime as dt
import numpy as np
import datasets as D

out = {}

# ---------------------------------------------------------------- Elec2 check
Xe, day_attr, dates_e, ye = D.load_elec2()
rec_dow = np.array([d.weekday() for d in dates_e]) + 1
out['elec2'] = {
    'records': int(len(Xe)),
    'days': int(len({d for d in dates_e})),
    'first': str(dates_e[0]), 'last': str(dates_e[-1]),
    'dow_match': float((rec_dow == day_attr).mean()),
}
# a wrong start date must destroy the agreement
shifted = []
for off in range(1, 7):
    shifted.append(float((((rec_dow - 1 + off) % 7 + 1) == day_attr).mean()))
out['elec2']['dow_match_wrong_offsets'] = shifted

# --------------------------------------------------------- Powersupply check
X, hour, dates = D.load_powersupply()
load = X.sum(axis=1)
daily = load.reshape(-1, 24).mean(axis=1)
udates = dates[::24]
dow = np.array([d.weekday() for d in udates])
hol = D.it_holiday(udates)
month = np.array([d.month for d in udates])

z = (daily - daily.mean()) / daily.std(ddof=1)
ordinary = (dow < 5) & (~hol)
out['powersupply'] = {
    'records': int(len(X)), 'days': int(len(daily)),
    'first': str(dates[0]), 'last': str(dates[-1]),
    'weekday_mean': float(daily[dow < 5].mean()),
    'weekend_mean': float(daily[dow >= 5].mean()),
    'holiday_z': float(z[hol].mean()),
    'ordinary_weekday_z': float(z[ordinary].mean()),
    'august_mean': float(daily[month == 8].mean()),
    'january_mean': float(daily[month == 1].mean()),
    'ferragosto_rank': [int((daily < daily[(month == 8) & (np.array([d.day for d in udates]) == 15)
                                           & (np.array([d.year for d in udates]) == yr)][0]).sum()) + 1
                        for yr in (1995, 1996, 1997)],
}

# The alignment is identified rather than assumed: every one of the seven
# possible day-of-week phases is scored by the weekday/weekend contrast.
phase_scores = []
for ph in range(7):
    d2 = (np.arange(len(daily)) + ph) % 7
    phase_scores.append(float(daily[d2 < 5].mean() - daily[d2 >= 5].mean()))
out['powersupply']['phase_contrast'] = phase_scores
out['powersupply']['phase_selected'] = int(np.argmax(phase_scores))

# Holiday effect tested formally.
from scipy.stats import mannwhitneyu
u = mannwhitneyu(daily[hol], daily[ordinary], alternative='less')
out['powersupply']['holiday_test'] = {'U': float(u.statistic), 'p': float(u.pvalue),
                                      'n_holiday': int(hol.sum())}

# Data defects found in the published file.
out['defects'] = {
    'powersupply_trailing_comma_line': 22211,
    'powersupply_corrupt_hour_index': 24968,
    'elec2_date_attribute_collisions': int(45312 // 48 - len(np.unique(np.loadtxt(
        D.os.path.join(D.DATA, 'elec2_data.dat'), usecols=0)))),
}

json.dump(out, open('../results/calendar.json', 'w'), indent=1)
for k, v in out.items():
    print(k, json.dumps(v, indent=1)[:900])
