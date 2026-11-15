from common import *
import argparse, math
from scipy.stats import norm

def hac(x,lag):
    x=np.asarray(x);n=len(x);c=x-x.mean();v=c@c/n
    for j in range(1,min(lag,n-1)+1):v+=2*(1-j/(lag+1))*(c[j:]@c[:-j])/n
    se=math.sqrt(max(v,0)/n);z=x.mean()/se
    return dict(delta=float(x.mean()),se=se,z=float(z),p=float(2*norm.sf(abs(z))))

def holm(ps):
    order=np.argsort(ps);out=np.empty(len(ps));prev=0
    for j,i in enumerate(order):prev=max(prev,(len(ps)-j)*ps[i]);out[i]=min(1,prev)
    return out

def blocks(x,length,seed=20261110):
    rng=np.random.default_rng(seed);n=len(x);means=[]
    for _ in range(5000):
        starts=rng.integers(n-length+1,size=math.ceil(n/length));ix=(starts[:,None]+np.arange(length)).ravel()[:n];means.append(x[ix].mean())
    return np.quantile(means,[.025,.975])

def main():
    setup();rows=[];series={};summaries=[];ablation=[];selector_rows=[];ensemble_rows=[]
    for u in UNIVERSES:
        for s in SEEDS:
            if not predpath(u,s,'master').exists() or not predpath(u,s,'full').exists():continue
            mp=loadnp(predpath(u,s,'master'));cp=loadnp(predpath(u,s,'full'))
            variants={'MASTER':mp,'B+D+S':cp,'CoSTER':fusion(mp,cp)}
            for kind,name in [('base','B'),('dense','B+D')]:
                path=predpath(u,s,kind)
                if path.exists():
                    pr=loadnp(path);variants[name]=pr;variants['M+'+name]=fusion(mp,pr)
            path=OUT/'ablation'/u/f'seed{s}/base_sparse/test.npz'
            if path.exists() and (path.parent/'result.json').exists():
                pr=loadnp(path);variants['B+S']=pr;variants['M+B+S']=fusion(mp,pr)
            for name,pr in variants.items():
                ds=daily(pr);series[u,s,name]=ds;ms=summary(ds);rows.append(dict(universe=u,seed=s,model=name,**ms))
            for mode in ('last','fixed_random','top2','softmax'):
                path=OUT/'selectors'/u/f'seed{s}'/mode/'test.npz'
                if path.exists() and (path.parent/'result.json').exists():
                    pr=loadnp(path)
                    for name,a in [(mode,pr),('M+'+mode,fusion(mp,pr))]:
                        ds=daily(a);series[u,s,name]=ds;selector_rows.append(dict(universe=u,seed=s,model=name,**summary(ds)))
            for kind in ('gru','ridge'):
                path=OUT/'ensembles'/u/f'seed{s}'/f'master_{kind}.npz'
                if path.exists():
                    ds=daily(loadnp(path));series[u,s,'MASTER+'+kind]=ds
                    ensemble_rows.append(dict(universe=u,seed=s,model='MASTER+'+kind,**summary(ds)))
        for a,b in [(i,i+1) for i in range(0,10,2)]:
            pa,pb=predpath(u,a,'master'),predpath(u,b,'master')
            if pa.exists() and pb.exists():
                mixed=fusion(loadnp(pa),loadnp(pb));ds=daily(mixed);series[u,a,'MASTER_pair']=ds
                ensemble_rows.append(dict(universe=u,seed=a,model=f'MASTER_pair_{a}_{b}',**summary(ds)))
    csvout(OUT/'tables/per_seed.csv',rows);csvout(OUT/'tables/selectors.csv',selector_rows);csvout(OUT/'tables/ensembles.csv',ensemble_rows)
    for u in UNIVERSES:
        names=sorted({r['model'] for r in rows if r['universe']==u})
        for name in names:
            rr=[r for r in rows if r['universe']==u and r['model']==name]
            v=np.array([[r[k] for k in METRICS] for r in rr]);n=len(rr)
            summaries.append(dict(universe=u,model=name,n_seeds=n,**{k+'_mean':float(v[:,j].mean()) for j,k in enumerate(METRICS)},**{k+'_sd':float(v[:,j].std(ddof=1)) if n>1 else None for j,k in enumerate(METRICS)}))
    csvout(OUT/'tables/summary.csv',summaries)
    nw=[];boot=[]
    common_seeds=[s for s in SEEDS if all((u,s,'CoSTER') in series for u in UNIVERSES)]
    for u in UNIVERSES:
        for metric in ('IC','RankIC'):
            data=[];dates=None
            for s in common_seeds:
                a=series[u,s,'CoSTER'];b=series[u,s,'MASTER'];ds=[r['date'] for r in a]
                assert ds==[r['date'] for r in b]
                if dates is not None:assert dates==ds
                dates=ds;data.append([x[metric]-y[metric] for x,y in zip(a,b)])
            if not data:continue
            x=np.mean(data,axis=0)
            csvout(OUT/'statistics'/f'{u}_{metric}_daily.csv',[dict(date=d,mean_delta=float(v)) for d,v in zip(dates,x)])
            for lag in (5,10,20):nw.append(dict(universe=u,metric=metric,n_seeds=len(data),n_days=len(x),lag=lag,**hac(x,lag)))
            for length in (10,20,40):
                ci=blocks(x,length);boot.append(dict(universe=u,metric=metric,n_seeds=len(data),n_days=len(x),block=length,draws=5000,delta=float(x.mean()),ci_low=float(ci[0]),ci_high=float(ci[1])))
    for lag in (5,10,20):
        ix=[i for i,r in enumerate(nw) if r['lag']==lag]
        for i,p in zip(ix,holm([nw[i]['p'] for i in ix])):nw[i]['holm_p']=float(p)
    csvout(OUT/'tables/newey_west.csv',nw);csvout(OUT/'tables/bootstrap.csv',boot)
    # Exploratory stage contrasts: datewise seed means, explicitly separate from primary family.
    for u in UNIVERSES:
        for left,right in [('M+B+D','M+B'),('M+B+S','M+B'),('CoSTER','M+B+D'),('CoSTER','M+B+S')]:
            ss=[s for s in SEEDS if (u,s,left) in series and (u,s,right) in series]
            if not ss:continue
            for metric in ('IC','RankIC'):
                x=[]
                for s in ss:
                    a,b=series[u,s,left],series[u,s,right];assert [r['date'] for r in a]==[r['date'] for r in b]
                    x.append([p[metric]-q[metric] for p,q in zip(a,b)])
                v=np.mean(x,axis=0);ci=blocks(v,20)
                ablation.append(dict(universe=u,contrast=left+' minus '+right,metric=metric,n_seeds=len(ss),**hac(v,10),ci_low=float(ci[0]),ci_high=float(ci[1]),inference='exploratory_unadjusted'))
    csvout(OUT/'tables/ablation_contrasts.csv',ablation)
    contrasts=[]
    for u in UNIVERSES:
        for comparator in ('MASTER+gru','MASTER+ridge','MASTER_pair','M+last','M+fixed_random','M+top2','M+softmax'):
            ss=[s for s in SEEDS if (u,s,comparator) in series and (u,s,'CoSTER') in series]
            if not ss:continue
            for metric in ('IC','RankIC'):
                values=[]
                for s in ss:
                    a,b=series[u,s,'CoSTER'],series[u,s,comparator]
                    assert [r['date'] for r in a]==[r['date'] for r in b]
                    values.append([p[metric]-q[metric] for p,q in zip(a,b)])
                x=np.mean(values,axis=0);ci=blocks(x,20)
                contrasts.append(dict(universe=u,contrast='CoSTER minus '+comparator,metric=metric,n_pairs=len(ss),seeds=';'.join(map(str,ss)),**hac(x,10),ci_low=float(ci[0]),ci_high=float(ci[1]),inference='exploratory_unadjusted'))
    csvout(OUT/'tables/control_contrasts.csv',contrasts)
    dump(OUT/'analysis_status.json',dict(status='complete_for_available_results',primary_paired_seeds=common_seeds,target_seeds=SEEDS,ablation_rows=len(ablation),selector_rows=len(selector_rows),ensemble_rows=len(ensemble_rows),paper_modified=False))
    print('ANALYZED paired seeds',common_seeds,flush=True)

if __name__=='__main__':main()
