import numpy as np, json
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import cross_val_score
from scipy.stats import wilcoxon
CL={'LDA':lambda:LDA(),'QDA':lambda:QDA(reg_param=1e-3),'NB':lambda:GaussianNB(),'KNN':lambda:KNeighborsClassifier(15)}
def gen(n, rng, shift):
    # regimes 0,1; features = two correlated readings; class 1 = higher load
    cov=np.array([[1,0.9],[0.9,1]])*0.35
    X=[];Y=[];R=[]
    for r in (0,1):
        base = np.array([0,0]) + shift*r*np.array([1.6,1.6])
        for y,delta in ((0,0.0),(1,1.6)):
            m=n//2
            X.append(rng.multivariate_normal(base+delta,cov,m)); Y+= [y]*m; R+=[r]*m
    return np.vstack(X),np.array(Y),np.array(R)
def run(n_per_regime, shift, reps=50, seed=0):
    rng=np.random.default_rng(seed)
    res={k:{'glob':[],'part':[]} for k in CL}; sel_part=[]; best_single=[]; swc_pool=[]; swc_reg=[]
    for rep in range(reps):
        Xtr,Ytr,Rtr=gen(n_per_regime,rng,shift); Xte,Yte,Rte=gen(2000,rng,shift)
        swc_pool.append(silhouette_score(Xtr,Ytr)); swc_reg.append(np.mean([silhouette_score(Xtr[Rtr==r],Ytr[Rtr==r]) for r in (0,1)]))
        for k,f in CL.items():
            g=f().fit(Xtr,Ytr); res[k]['glob'].append(accuracy_score(Yte,g.predict(Xte)))
            pred=np.empty_like(Yte)
            for r in (0,1):
                m=f().fit(Xtr[Rtr==r],Ytr[Rtr==r]); pred[Rte==r]=m.predict(Xte[Rte==r])
            res[k]['part'].append(accuracy_score(Yte,pred))
        # argmin selection per regime via 5-fold CV
        pred=np.empty_like(Yte)
        for r in (0,1):
            Xs,Ys=Xtr[Rtr==r],Ytr[Rtr==r]
            sc={k:cross_val_score(f(),Xs,Ys,cv=5).mean() for k,f in CL.items()}
            kb=max(sc,key=sc.get); pred[Rte==r]=CL[kb]().fit(Xs,Ys).predict(Xte[Rte==r])
        sel_part.append(accuracy_score(Yte,pred))
        best_single.append(max(res[k]['part'][-1] for k in CL))
    out={'n':n_per_regime,'shift':shift,'swc_pool':float(np.mean(swc_pool)),'swc_reg':float(np.mean(swc_reg))}
    for k in CL:
        g=np.array(res[k]['glob']);p=np.array(res[k]['part']); d=p-g
        pv = wilcoxon(p,g).pvalue if np.any(d!=0) else 1.0
        out[k]={'glob':float(g.mean()*100),'glob_sd':float(g.std(ddof=1)*100),'part':float(p.mean()*100),'part_sd':float(p.std(ddof=1)*100),'diff':float(d.mean()*100),'diff_sd':float(d.std(ddof=1)*100),'p':float(pv),'wins':int((d>0).sum())}
    out['sel']=float(np.mean(sel_part)*100)
    return out
if __name__=="__main__":
  allres=[]
  for shift in (0.0,0.5,1.0):
      for n in (20,50,200,1000):
          o=run(n,shift); allres.append(o); print(shift,n,round(o['swc_pool'],3),round(o['swc_reg'],3),{k:(round(o[k]['glob'],1),round(o[k]['part'],1),round(o[k]['diff'],2),'%.1e'%o[k]['p'],o[k]['wins']) for k in CL},round(o['sel'],1))
  json.dump(allres,open('sim_results.json','w'),indent=1)
