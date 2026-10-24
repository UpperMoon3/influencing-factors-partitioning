import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import sys; sys.path.insert(0,'.')
plt.rcParams.update({'font.family':'serif','font.serif':['TeX Gyre Pagella','DejaVu Serif'],'mathtext.fontset':'stix','font.size':9,'axes.linewidth':0.6})
R=json.load(open('sim_results.json')); K=['LDA','QDA','NB','KNN']
rows=[]
for o in R:
    g=np.mean([o[k]['glob'] for k in K]); p=np.mean([o[k]['part'] for k in K]); d=np.mean([o[k]['diff'] for k in K]); sd=np.mean([o[k]['diff_sd'] for k in K])
    sig_pos=sum(o[k]['p']<0.05 and o[k]['diff']>0 for k in K); sig_neg=sum(o[k]['p']<0.05 and o[k]['diff']<0 for k in K)
    best=max(o[k]['part'] for k in K)
    rows.append((o['shift'],o['n'],o['swc_pool'],o['swc_reg'],g,p,d,sd,sig_pos,sig_neg,o['sel'],best))
    print('%.1f & %d & %.3f & %.3f & %.1f & %.1f & %+.2f & %d/%d & %.1f & %.1f'%(o['shift'],o['n'],o['swc_pool'],o['swc_reg'],g,p,d,sig_pos,sig_neg,o['sel'],best))
# data illustration
from sim import gen
rng=np.random.default_rng(3); X,Y,Rg=gen(300,rng,0.5)
fig,axs=plt.subplots(1,2,figsize=(7.2,2.9))
ax=axs[0]; cols={0:'#7A9CC6',1:'#E3A587'}; mk={0:'o',1:'^'}
for r in (0,1):
    for y in (0,1):
        m=(Rg==r)&(Y==y); ax.scatter(X[m,0],X[m,1],s=9,marker=mk[r],facecolor=cols[y],edgecolor='none',alpha=0.75,label=f'Class {y}, regime {r+1}')
ax.set_xlabel('Feature 1'); ax.set_ylabel('Feature 2'); ax.set_title('(a) Synthetic sample, $\\delta=0.5$',fontsize=9)
ax.legend(fontsize=6.5,frameon=False,loc='upper left',markerscale=1.3); ax.grid(lw=0.4,color='#ddd')
for s in ['top','right']: ax.spines[s].set_visible(False)
ax=axs[1]; lc={0.0:'#8C8C8C',0.5:'#7A9CC6',1.0:'#E3A587'}
for sh in (0.0,0.5,1.0):
    rr=[r for r in rows if r[0]==sh]; n=[r[1] for r in rr]; d=[r[6] for r in rr]; sd=[r[7] for r in rr]
    ax.errorbar(n,d,yerr=sd,marker='o',ms=4,lw=1.2,capsize=2.5,color=lc[sh],label=f'$\\delta={sh}$')
    for xi,yi in zip(n,d): ax.annotate((f'{yi:+.1f}' if abs(yi)>=0.05 else '0.0'),(xi,yi),textcoords='offset points',xytext=(0,6),ha='center',fontsize=6.5,color='#333')
ax.axhline(0,color='k',lw=0.6); ax.set_xscale('log'); ax.set_xticks([20,50,200,1000]); ax.set_xticklabels(['20','50','200','1000'])
ax.set_xlabel('Training tuples per regime'); ax.set_ylabel('$\\Delta$ accuracy, p.p.'); ax.set_title('(b) Partitioned minus global model',fontsize=9)
ax.grid(lw=0.4,color='#ddd'); ax.legend(fontsize=7.5,frameon=False,loc='center right')
for s in ['top','right']: ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig('../figures/simulation.pdf')
