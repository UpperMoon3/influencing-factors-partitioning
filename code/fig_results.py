"""Figures built from the experimental results."""
import json
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['TeX Gyre Pagella', 'Palatino', 'DejaVu Serif'],
                     'font.size': 8.5, 'mathtext.fontset': 'stix', 'axes.linewidth': 0.6,
                     'axes.edgecolor': '#555', 'legend.frameon': False})
C0, C1, CA = '#7A9CC6', '#E3A587', '#8FBF9F'
CLFS = ['LDA', 'QDA', 'NB', 'KNN', 'RF', 'HGB', 'MLP']
PARAM = {'LDA', 'QDA', 'NB'}

AGG = json.load(open('../results/agg.json'))

# --------------------------------------------- paired gains, both baselines
datasets = list(AGG)
fig, axs = plt.subplots(1, len(datasets), figsize=(7.2, 2.6), sharey=True)
if len(datasets) == 1:
    axs = [axs]
for ax, ds in zip(axs, datasets):
    levels = list(AGG[ds])
    x = np.arange(len(CLFS)); w = 0.38
    # deepest level
    lv = levels[-1]
    recs = {r['clf']: r for r in AGG[ds][lv]['recs']}
    dp = [recs[c]['d_pool'] if c in recs else 0 for c in CLFS]
    df = [recs[c]['d_feat'] if c in recs else 0 for c in CLFS]
    ax.bar(x - w / 2, dp, w, color=C0, label='vs. pooled', zorder=3)
    ax.bar(x + w / 2, df, w, color=C1, label='vs. pooled + regime feature', zorder=3)
    ax.axhline(0, color='#555', lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(CLFS, rotation=0)
    ax.set_title(f'({"ab"[datasets.index(ds)]}) {ds}, {lv}')
    ax.grid(axis='y', lw=0.4, color='#ddd', zorder=0)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    for i, c in enumerate(CLFS):
        if c in PARAM:
            ax.axvspan(i - 0.5, i + 0.5, color='#f3f3f3', zorder=0)
axs[0].set_ylabel('paired gain, p.p.')
axs[0].legend(fontsize=7, loc='upper left')
fig.tight_layout(); fig.savefig('../figures/gains.pdf'); plt.close(fig)

# ---------------------------------- depth of the partition: gain per level
fig, ax = plt.subplots(figsize=(3.5, 2.4))
for ds, col in zip(datasets, (C0, C1)):
    levels = list(AGG[ds])
    y = [np.mean([r['d_pool'] for r in AGG[ds][lv]['recs']]) for lv in levels]
    m = [AGG[ds][lv]['n_regimes'] for lv in levels]
    ax.plot(m, y, 'o-', color=col, lw=1.2, ms=4, label=ds)
ax.axhline(0, color='#555', lw=0.7)
ax.set_xlabel('number of regimes $M$'); ax.set_ylabel('mean gain over pooled, p.p.')
ax.grid(lw=0.4, color='#ddd'); ax.legend(fontsize=7)
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig('../figures/depth.pdf'); plt.close(fig)

# ------------------------------------- silhouette gain against actual gain
try:
    cells = json.load(open('../results/swc_cells.json'))
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    d = np.array([c['dswc'] for c in cells]); a = np.array([c['dacc'] for c in cells])
    par = np.array([c['clf'] in PARAM for c in cells])
    ax.scatter(d[par], a[par], s=12, c=C0, lw=0, alpha=.75, label='parametric')
    ax.scatter(d[~par], a[~par], s=12, c=C1, lw=0, alpha=.75, label='flexible')
    ax.axhline(0, color='#555', lw=0.7); ax.axvline(0, color='#555', lw=0.7)
    ax.set_xlabel('silhouette gain of the regime, $SWC_i-SWC_{\\mathrm{pool}}$')
    ax.set_ylabel('accuracy gain, p.p.')
    ax.grid(lw=0.4, color='#ddd'); ax.legend(fontsize=7)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig('../figures/swc_vs_gain.pdf'); plt.close(fig)
except FileNotFoundError:
    print('swc cells not available yet')
print('figures written')
