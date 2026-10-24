"""Datasets and a priori factor definitions.

Every factor condition c_j(t) used here is derived exclusively from a calendar,
never from the feature values or the class labels.
"""
import datetime as dt
import numpy as np
import os

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

# Italian national public holidays with fixed dates (the Powersupply stream is
# recorded by an Italian electricity provider).  Easter Monday is movable and is
# added separately.
IT_FIXED = {(1, 1), (1, 6), (4, 25), (5, 1), (6, 2), (8, 15),
            (11, 1), (12, 8), (12, 25), (12, 26)}

# New South Wales public holidays with fixed dates (Elec2 covers the NSW market).
NSW_FIXED = {(1, 1), (1, 26), (4, 25), (6, 8), (12, 25), (12, 26)}


def easter(year):
    """Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month, day = divmod(h + ell - 7 * m + 114, 31)
    return dt.date(year, month, day + 1)


def load_powersupply():
    """Powersupply stream with timestamps recovered from the hour-of-day cycle.

    The published file carries two power readings and the hour of the day but no
    dates.  The 29,928 records form 1,247 complete 24-hour cycles, so the stream
    is contiguous and a single start date fixes every timestamp.  The start date
    1 January 1995 is verified in verify_calendar.py.
    """
    rows = []
    with open(os.path.join(DATA, 'powersupply.arff')) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(('@', '%')):
                continue
            # One data line carries a trailing comma (line 22211 of the file).
            parts = [t for t in line.split(',') if t != '']
            a, b, c = parts
            rows.append((float(a), float(b), int(c)))
    arr = np.array(rows)
    X = arr[:, :2]
    hour = arr[:, 2].astype(int)
    # One record carries a corrupted hour label: at index 24968 the cycle reads
    # 8, 6, 10 where 8, 9, 10 is expected.  The position in the cycle is
    # unambiguous, so the hour is restored from the record index.
    hour = np.arange(len(X)) % 24
    start = dt.date(1995, 1, 1)
    dates = np.array([start + dt.timedelta(days=int(i)) for i in np.arange(len(X)) // 24])
    return X, hour, dates


def load_elec2():
    """Elec2 with timestamps recovered from the half-hour period cycle.

    The normalised file rounds the date attribute, which collapses 11 of the 944
    days, so days are recovered from the resets of the period counter instead.
    The recovered day of week reproduces the day attribute of the file exactly.
    """
    raw = np.loadtxt(os.path.join(DATA, 'elec2_data.dat'))
    y = np.array([0 if line.strip() in ('0', '0.0') else 1
                  for line in open(os.path.join(DATA, 'elec2_label.dat'))])
    period = np.round(raw[:, 2] * 47).astype(int)
    day_index = np.cumsum(np.concatenate([[0], (np.diff(period) < 0).astype(int)]))
    start = dt.date(1996, 5, 7)
    dates = np.array([start + dt.timedelta(days=int(i)) for i in day_index])
    # date and nswprice are dropped following Gama et al.; the day attribute is
    # dropped from the features because it defines the factor and would
    # otherwise be available to the pooled model as an ordinary input.
    X = raw[:, [2, 4, 5, 6, 7]]
    return X, raw[:, 1].astype(int), dates, y


def powersupply_task(hour, work_start=8, work_end=18):
    """Binary task: does the record fall in working hours?"""
    return ((hour >= work_start) & (hour < work_end)).astype(int)


def it_holiday(dates):
    out = np.zeros(len(dates), dtype=bool)
    years = {d.year for d in dates}
    movable = set()
    for yr in years:
        e = easter(yr)
        movable.add(e + dt.timedelta(days=1))   # Easter Monday
    for i, d in enumerate(dates):
        out[i] = ((d.month, d.day) in IT_FIXED) or (d in movable)
    return out


def nsw_holiday(dates):
    out = np.zeros(len(dates), dtype=bool)
    years = {d.year for d in dates}
    movable = set()
    for yr in years:
        e = easter(yr)
        movable.update({e - dt.timedelta(days=2), e + dt.timedelta(days=1)})
    for i, d in enumerate(dates):
        out[i] = ((d.month, d.day) in NSW_FIXED) or (d in movable)
    return out


def factor_table(dates, country):
    """Boolean a priori factor conditions c_j(t) for every record."""
    month = np.array([d.month for d in dates])
    dow = np.array([d.weekday() for d in dates])
    hol = it_holiday(dates) if country == 'IT' else nsw_holiday(dates)
    f = {'weekend': dow >= 5, 'holiday': hol}
    if country == 'IT':
        f['winter'] = np.isin(month, [12, 1, 2])
        # The Italian industrial shutdown around Ferragosto.
        f['august'] = np.array([(d.month == 8 and 5 <= d.day <= 25) for d in dates])
    else:
        # NSW: summer is December to February.
        f['summer'] = np.isin(month, [12, 1, 2])
        f['winter'] = np.isin(month, [6, 7, 8])
    f['nonworking'] = f['weekend'] | f['holiday']
    return f
