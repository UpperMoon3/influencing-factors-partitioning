"""LaTeX tables for the analyses added in the revision.

stream.json      -> tables/stream.tex   (prequential comparison with drift adaptation)
factors.json     -> tables/factors.tex  (choice of factors)
size.json        -> tables/pmin.tex     (minimum subsample size)
sim_extra.json   -> tables/sim_depth2.tex, tables/sim_ind.tex
Every file is written only if its input exists.
"""
import json
import os
import numpy as np

R = '../results/'
T = '../tables/'
BATCH = ('LDA', 'QDA', 'NB', 'KNN')
POL = [('pooled', 'Pooled'), ('feature', 'Pooled + regime feature'), ('window', 'Sliding window'),
       ('ddm', 'DDM'), ('adwin', 'ADWIN'), ('partitioned', '\\textbf{Partitioned (proposed)}'),
       ('part+adwin', 'Partitioned + ADWIN')]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    out = np.empty(len(p))
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (len(p) - k) * p[i])
        out[i] = min(1.0, run)
    return out


def stream():
    S = json.load(open(R + 'stream.json'))
    lines = ['\\begin{tabular}{@{}lcccc@{}}', '\\toprule',
             '\\textbf{Policy} & \\textbf{LDA} & \\textbf{QDA} & \\textbf{NB} & \\textbf{KNN}\\\\']
    for ds, o in S.items():
        res, tests = o['results'], o['tests']
        sig = {}
        for c in BATCH:
            keys = [p for p, _ in POL if p != 'partitioned']
            adj = holm([tests[f'{c}|{p}']['p'] for p in keys])
            for p, a in zip(keys, adj):
                sig[(c, p)] = a < 0.05
        best = {c: max(res[f'{c}|{p}']['acc'] for p, _ in POL) for c in BATCH}
        lines += ['\\midrule', f'\\multicolumn{{5}}{{l}}{{\\textit{{{ds}}}, $L_3$, $M={o["n_regimes"]}$, '
                  f'{o["months"]} months}}\\\\']
        for p, name in POL:
            cells = []
            for c in BATCH:
                a = res[f'{c}|{p}']['acc']
                s = f'{a:.2f}'
                if abs(a - best[c]) < 1e-9:
                    s = f'\\textbf{{{s}}}'
                if p != 'partitioned' and sig[(c, p)]:
                    s += '\\,$^{*}$'
                else:
                    s += '\\phantom{\\,$^{*}$}'
                cells.append(s)
            lines.append(f'{name} & ' + ' & '.join(cells) + '\\\\')
        inc = ' \\quad '.join(f'{k} {res[k + "|incremental"]["acc"]:.2f}'
                              for k in ('ARF', 'ARF + feature', 'HAT'))
        lines.append(f'\\multicolumn{{5}}{{l}}{{Incremental learners: {inc}}}\\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    open(T + 'stream.tex', 'w').write('\n'.join(lines) + '\n')
    print('tables/stream.tex')


def sim_extra():
    from scipy.stats import spearmanr
    d = json.load(open(R + 'sim_extra.json'))
    lines = ['\\begin{tabular}{@{}cccccccc@{}}', '\\toprule',
             '$\\boldsymbol{p}$ & $\\boldsymbol{M}$ & \\textbf{Objects per part} & \\textbf{Pooled} & '
             '\\textbf{True regimes} & \\textbf{Partitioned} & $\\boldsymbol{p_{\\min}=50}$ & $\\boldsymbol{p_{\\min}=100}$\\\\',
             '\\midrule']
    last = None
    for r in d['part_a']:
        if last is not None and r['p'] != last:
            lines.append('\\midrule')
        last = r['p']
        lines.append(f"{r['p']} & {r['M']} & {r['n_per_part']} & {r['pool']:.2f} & {r['true2']:.2f} & "
                     f"{r['part']:.2f} & {r['pmin50']:.2f} & {r['pmin100']:.2f}\\\\")
    lines += ['\\bottomrule', '\\end{tabular}']
    open(T + 'sim_depth2.tex', 'w').write('\n'.join(lines) + '\n')

    c = d['part_b_cells']
    a = np.array([x['gain'] for x in c])
    s = np.array([x['dswc'] for x in c])
    g = np.array([x['gap'] for x in c])
    conc = np.array([x['concept'] > 0.5 and x['delta'] < 0.3 for x in c])
    ok = ~np.isnan(g)

    def st(ind, m):
        rho = spearmanr(ind[m], a[m])[0]
        pred, tr = ind[m] > 0, a[m] > 0
        tp, fp = (pred & tr).sum(), (pred & ~tr).sum()
        fn, tn = (~pred & tr).sum(), (~pred & ~tr).sum()
        return rho, tp / max(tp + fp, 1), tp / max(tp + fn, 1), (tp + tn) / m.sum()

    rows = [('All 300', '$\\Delta SWC$', st(s, np.ones(len(a), bool)), (a > 0).mean()),
            ('216 with $n\\ge 50$', '$\\Delta SWC$', st(s, ok), (a[ok] > 0).mean()),
            ('', '$\\widehat{\\Gamma}$', st(g, ok), (a[ok] > 0).mean()),
            (f'{int((ok & conc).sum())} with concept change', '$\\Delta SWC$', st(s, ok & conc), (a[ok & conc] > 0).mean()),
            ('', '$\\widehat{\\Gamma}$', st(g, ok & conc), (a[ok & conc] > 0).mean())]
    lines = ['\\begin{tabular}{@{}llccccc@{}}', '\\toprule',
             '\\textbf{Configurations} & \\textbf{Indicator} & $\\boldsymbol{\\rho}$ & \\textbf{Precision} & '
             '\\textbf{Recall} & \\textbf{Rule acc.} & \\textbf{Always}\\\\', '\\midrule']
    for name, ind, (rho, pr, rc, acc), base in rows:
        b = f'{base:.3f}' if base is not None else '---'
        lines.append(f'{name} & {ind} & {rho:+.2f} & {pr:.3f} & {rc:.3f} & {acc:.3f} & {b}\\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    open(T + 'sim_ind.tex', 'w').write('\n'.join(lines) + '\n')
    print('tables/sim_depth2.tex, tables/sim_ind.tex')


CLF7 = ('LDA', 'QDA', 'NB', 'KNN', 'RF', 'HGB', 'MLP')
PARAM = ('LDA', 'QDA', 'NB')


def factors():
    F = {}
    for f in ('factors.json', 'factors_Elec2.json'):
        if os.path.exists(R + f):
            for k, v in json.load(open(R + f)).items():
                F.setdefault(k, {}).update(v)
    lines = ['\\begin{tabular}{@{}lcccccc@{}}', '\\toprule',
             '& & & \\multicolumn{2}{c}{\\textbf{Randomised}} & \\multicolumn{2}{c}{\\textbf{Chronological}}\\\\',
             '\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}',
             '\\textbf{Factor set} & $\\boldsymbol{M}$ & \\textbf{Smallest} & $\\boldsymbol{\\Delta}$\\textbf{ pool} & '
             '$\\boldsymbol{\\Delta}$\\textbf{ feat} & $\\boldsymbol{\\Delta}$\\textbf{ pool} & $\\boldsymbol{\\Delta}$\\textbf{ feat}\\\\']
    summary = {}
    for ds, sets in F.items():
        lines += ['\\midrule', f'\\multicolumn{{7}}{{l}}{{\\textit{{{ds}}}}}\\\\']
        rnd = {k: v for k, v in sets.items() if k.startswith('random')}
        for name, v in sets.items():
            if name in rnd:
                continue
            m = lambda proto, key: np.mean([v[proto][c][key] for c in CLF7])
            vals = [m('random', 'd_pool'), m('random', 'd_feat'), m('chrono', 'd_pool'), m('chrono', 'd_feat')]
            summary[(ds, name)] = vals
            lines.append(f"{name} & {v['M']} & {v['min_size']:,} & " + ' & '.join(f'{x:+.2f}' for x in vals) + '\\\\')
        if rnd:
            arr = np.array([[np.mean([v[p][c][k] for c in CLF7]) for p, k in
                             (('random', 'd_pool'), ('random', 'd_feat'), ('chrono', 'd_pool'), ('chrono', 'd_feat'))]
                            for v in rnd.values()])
            v0 = next(iter(rnd.values()))
            summary[(ds, 'random')] = arr.mean(0).tolist()
            lines.append(f"random, shape of $L_3$ ({len(rnd)} draws) & {v0['M']} & {v0['min_size']:,} & "
                         + ' & '.join(f'{a:+.2f}$\\pm${s:.2f}' for a, s in zip(arr.mean(0), arr.std(0, ddof=1))) + '\\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    txt = '\n'.join(lines).replace(' x ', ' $\\times$ ')
    for k in (1, 2, 3):
        txt = txt.replace(f'(L{k})', f'($L_{k}$)')
    txt = txt.replace('(too fine)', '(fine)')
    open(T + 'factors.tex', 'w').write(txt + '\n')
    print('tables/factors.tex')
    return summary


def size():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['TeX Gyre Pagella', 'Palatino', 'DejaVu Serif'],
                         'font.size': 8.5, 'mathtext.fontset': 'stix', 'axes.linewidth': 0.6,
                         'axes.edgecolor': '#555', 'legend.frameon': False})
    S = json.load(open(R + 'size.json'))
    grid = ['5', '250', '500', '1000', '2000', '4000', '8000', '16000', '1000000000']
    show = ['5', '250', '500', '1000', '2000', '4000', '8000', '1000000000']
    lab = lambda g: '$\\infty$' if g == '1000000000' else f'{int(g):,}'
    lines = ['\\begin{tabular}{@{}l' + 'c' * len(show) + '@{}}', '\\toprule',
             '$\\boldsymbol{p_{\\min}}$ & ' + ' & '.join(lab(g) for g in show) + '\\\\']
    for ds, o in S.items():
        for proto, key in (('randomised', 'pmin_randomised'), ('chronological', 'pmin_chrono')):
            d = o[key]
            lines.append('\\midrule' if proto == 'randomised' else '')
            for grp, members in (('parametric', PARAM), ('flexible', tuple(c for c in CLF7 if c not in PARAM))):
                base = np.mean([np.mean(d[c]['1000000000']) for c in members])
                vals = [np.mean([np.mean(d[c][g]) for c in members]) - base for g in show]
                lines.append(f'{ds}, {proto}, {grp} & ' + ' & '.join(f'{v:+.2f}' for v in vals) + '\\\\')
    lines = [l for l in lines if l != '']
    lines += ['\\bottomrule', '\\end{tabular}']
    open(T + 'pmin.tex', 'w').write('\n'.join(lines) + '\n')

    C = {'pool': ('#999999', '-', 'pooled'), 'feat': ('#E3A587', '--', 'pooled + regime feature'),
         'part': ('#7A9CC6', '-', 'partitioned'), 'part_pmin': ('#2E5A88', ':', 'partitioned, $p_{\\min}=100$')}
    dss = [ds for ds in S if 'curve' in S[ds]]
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.8))
    axs = axs.ravel()
    xs = [5, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000]
    for j, ds in enumerate(S):
        ax = axs[j]
        for proto, key, ls in (('randomised', 'pmin_randomised', '-'), ('chronological', 'pmin_chrono', '--')):
            d = S[ds][key]
            for grp, members, col in (('parametric', PARAM, '#E3A587'),
                                      ('flexible', tuple(c for c in CLF7 if c not in PARAM), '#7A9CC6')):
                base = np.mean([np.mean(d[c]['1000000000']) for c in members])
                ys = [np.mean([np.mean(d[c][g]) for c in members]) - base for g in grid]
                ax.plot(xs, ys, ls=ls, color=col, marker='o', ms=2.5, lw=1.1, label=f'{grp}, {proto}')
        ax.set_xscale('log'); ax.axhline(0, color='#555', lw=0.6)
        ax.set_xticks([5, 100, 1000, 10000, 32000]); ax.set_xticklabels(['5', '100', '1k', '10k', '$\\infty$'])
        ax.set_xlabel('$p_{\\min}$ (training records)'); ax.set_title(f'({"abcd"[j]}) {ds}: size rule')
        if j == 0:
            ax.set_ylabel('gain over pooled, p.p.'); ax.legend(fontsize=6)
    for j, ds in enumerate(dss):
        ax = axs[2 + j]
        cur = S[ds]['curve']
        fr = sorted(cur['LDA'], key=float)
        for k, (col, ls, name) in C.items():
            ys = [np.mean([np.mean(cur[c][f][k]) for c in CLF7]) for f in fr]
            ax.plot([float(f) * 100 for f in fr], ys, color=col, ls=ls, lw=1.2, marker='o', ms=2.5, label=name)
        ax.set_xscale('log'); ax.set_xlabel('training data used, %')
        ax.set_title(f'({"abcd"[2 + j]}) {ds}: learning curve')
        if j == 0:
            ax.set_ylabel('accuracy, %'); ax.legend(fontsize=6)
    for a in axs:
        a.grid(lw=0.4, color='#ddd')
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig('../figures/pmin.pdf')
    print('tables/pmin.tex, figures/pmin.pdf')


if __name__ == '__main__':
    os.makedirs(T, exist_ok=True)
    if os.path.exists(R + 'factors.json') or os.path.exists(R + 'factors_Elec2.json'):
        factors()
    if os.path.exists(R + 'size.json'):
        size()
    if os.path.exists(R + 'stream.json'):
        stream()
    if os.path.exists(R + 'sim_extra.json'):
        sim_extra()
