"""Figures produced directly from the data: calendar evidence, feature space,
class-conditional densities and silhouette profiles."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
import datasets as D, partitions as P

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['TeX Gyre Pagella', 'Palatino', 'DejaVu Serif'],
                     'font.size': 8.5, 'mathtext.fontset': 'stix', 'axes.linewidth': 0.6,
                     'axes.edgecolor': '#555', 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'legend.frameon': False})
C0, C1, CA = '#7A9CC6', '#E3A587', '#8FBF9F'

X, hour, dates = D.load_powersupply()
y = D.powersupply_task(hour)
lv, fac = P.powersupply_levels(dates)
daily = X.sum(1).reshape(-1, 24).mean(1)
udates = dates[::24]
dow = np.array([d.weekday() for d in udates])
month = np.array([d.month for d in udates])

# ---------------------------------------------------------------- calendar
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.2))
ax = axs[0]
mm = [daily[dow == k].mean() for k in range(7)]
ax.bar(range(7), mm, color=[C0] * 5 + [C1] * 2, zorder=3)
ax.set_xticks(range(7)); ax.set_xticklabels(['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su'])
ax.set_ylabel('mean daily load'); ax.set_title('(a) recovered day of week')
ax.set_ylim(min(mm) * 0.9, max(mm) * 1.03)

ax = axs[1]
mo = [daily[month == k].mean() for k in range(1, 13)]
ax.bar(range(1, 13), mo, color=[C1 if k == 8 else C0 for k in range(1, 13)], zorder=3)
ax.set_xticks(range(1, 13)); ax.set_xticklabels(list('JFMAMJJASOND'))
ax.set_title('(b) recovered month'); ax.set_ylim(min(mo) * 0.9, max(mo) * 1.03)

ax = axs[2]
ph = [daily[(np.arange(len(daily)) + k) % 7 < 5].mean() - daily[(np.arange(len(daily)) + k) % 7 >= 5].mean()
      for k in range(7)]
ax.bar(range(7), ph, color=[CA if k == int(np.argmax(ph)) else '#c9c9c9' for k in range(7)], zorder=3)
ax.axhline(0, color='#555', lw=0.6)
ax.set_xlabel('weekly phase'); ax.set_title('(c) working-day contrast')
for a in axs:
    a.grid(axis='y', lw=0.4, color='#ddd', zorder=0)
    for sp in ('top', 'right'): a.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig('../figures/calendar.pdf'); plt.close(fig)

# ------------------------------------------------- feature space and density
reg = lv['L3 + August shutdown']
# the most specific regime that is large enough to plot
sizes = np.bincount(reg)
target = int(np.argmax([s if s > 400 else 0 for s in sizes * (np.arange(len(sizes)) > 0)]))
mask = reg == target

fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.35))
ax = axs[0]
ax.scatter(X[::7, 0], X[::7, 1], s=2, c='#d9d9d9', lw=0, label='pooled sample $X^p$')
ax.scatter(X[mask][::3, 0], X[mask][::3, 1], s=2, c=CA, lw=0, label='one regime')
ax.set_xlabel('power from the main grid'); ax.set_ylabel('power from other grids')
ax.set_title('(a) feature space'); ax.legend(markerscale=3, fontsize=7, loc='upper left')

for j, (m, ttl) in enumerate([(np.ones(len(X), bool), '(b) pooled sample'),
                              (mask, '(c) one regime')]):
    ax = axs[j + 1]
    tot = X[m].sum(1)
    for cls, col, ls, lab in ((1, C0, '-', 'working hours'), (0, C1, '--', 'non-working')):
        v = tot[y[m] == cls]
        xs = np.linspace(tot.min(), tot.max(), 300)
        ax.plot(xs, gaussian_kde(v)(xs), color=col, ls=ls, lw=1.2, label=lab)
    ax.set_xlabel('total supplied power'); ax.set_title(ttl)
    if j == 0: ax.set_ylabel('density'); ax.legend(fontsize=7)
for a in axs:
    for sp in ('top', 'right'): a.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig('../figures/space_density.pdf'); plt.close(fig)

# ------------------------------------------------------------- silhouettes
def sil_profile(ax, Xs, ys, title):
    Z = StandardScaler().fit_transform(Xs)
    sv = silhouette_samples(Z, ys)
    pos = 0
    for cls, col in ((0, C1), (1, C0)):
        v = np.sort(sv[ys == cls])
        ax.fill_betweenx(np.arange(pos, pos + len(v)), 0, v, color=col, lw=0)
        pos += len(v) + max(30, len(v) // 20)
    ax.axvline(sv.mean(), color='#333', ls=':', lw=0.9)
    ax.set_title(title); ax.set_xlabel('silhouette value'); ax.set_yticks([])
    ax.set_xlim(-0.6, 1.0)
    return sv.mean()

rng = np.random.default_rng(0)
sub = rng.choice(len(X), 4000, replace=False)
subm = np.where(mask)[0]
subm = rng.choice(subm, min(4000, len(subm)), replace=False)
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.35))
a1 = sil_profile(axs[0], X[sub], y[sub], '(a) pooled sample')
a2 = sil_profile(axs[1], X[subm], y[subm], '(b) one regime')
ax = axs[2]
names, vals = [], []
for lname, r in lv.items():
    from protocol import swc
    if lname.startswith('L0'):
        names.append('$L_0$'); vals.append(swc(X, y)); continue
    sw = [swc(X[r == k], y[r == k]) for k in range(int(r.max()) + 1) if (r == k).sum() > 50]
    names.append('$L_' + lname[1] + '$'); vals.append(np.nanmean(sw))
ax.bar(range(len(vals)), vals, color=C0, zorder=3)
ax.set_xticks(range(len(vals))); ax.set_xticklabels(names)
ax.set_ylabel('mean $SWC$'); ax.set_title('(c) refinement level')
ax.grid(axis='y', lw=0.4, color='#ddd', zorder=0)
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig('../figures/silhouette.pdf'); plt.close(fig)
print('SWC pooled %.3f regime %.3f' % (a1, a2))
print('levels', dict(zip(names, np.round(vals, 3))))
