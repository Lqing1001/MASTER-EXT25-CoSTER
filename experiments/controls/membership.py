from common import *
from analyze import hac,blocks

def restricted(m,store,other):
    chunks={k:[] for k in ('dates','instruments','labels','predictions')}
    for day in store.dates_between(B.TEST_START,B.TEST_END):
        x,y,ids=store.batch(int(day));_,_,otherids=other._day_arrays(int(day));keep=np.isin(ids,otherids)
        with torch.inference_mode():p=m(torch.from_numpy(x[keep]).to(DEVICE)).cpu().numpy()
        for k,v in [('dates',np.full(keep.sum(),day,dtype=np.int32)),('instruments',ids[keep].astype('S8')),('labels',y[keep]),('predictions',p)]:chunks[k].append(v)
    return {k:np.concatenate(v) for k,v in chunks.items()}

def main():
    setup();u='csi300';store=B.UniverseStore(DATA,u);other=B.UniverseStore(DATA,'csi800');rows=[];deltas={};details=[]
    for s in SEEDS:
        if not predpath(u,s,'full').exists():continue
        mp=loadnp(predpath(u,s,'master'));cp=loadnp(predpath(u,s,'full'));fp=fusion(mp,cp)
        keep=np.ones(len(mp['dates']),dtype=bool)
        for day in np.unique(mp['dates']):
            ix=np.flatnonzero(mp['dates']==day);_,_,ids=other._day_arrays(int(day));keep[ix]=np.isin(mp['instruments'][ix],ids.astype('S8'))
        assert (~keep).sum()==890 and ((~keep)&np.isfinite(mp['labels'])).sum()==866
        if s==0:
            details=[dict(date=int(mp['dates'][i]),instrument=bytes(mp['instruments'][i]).decode(),finite_target=bool(np.isfinite(mp['labels'][i]))) for i in np.flatnonzero(~keep)]
        arrays={}
        for name,a in [('MASTER',mp),('CoSTER',fp)]:
            arrays['original',name]=daily(a);arrays['fixed_prediction_filter',name]=daily(a,keep)
        re={}
        for kind in ('master','full'):
            p=OUT/'membership'/f'seed{s}'/f'{kind}_restricted.npz'
            if p.exists():re[kind]=loadnp(p)
            else:
                m=model(u,s,kind);a=restricted(m,store,other);del m
                p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**a);re[kind]=a
        for name,a in [('MASTER',re['master']),('CoSTER',fusion(re['master'],re['full']))]:arrays['restricted_context',name]=daily(a)
        for scenario in ['original','fixed_prediction_filter','restricted_context']:
            for name in ['MASTER','CoSTER']:
                ds=arrays[scenario,name];rows.append(dict(universe=u,seed=s,scenario=scenario,model=name,**summary(ds)))
            a,b=arrays[scenario,'CoSTER'],arrays[scenario,'MASTER'];assert [x['date'] for x in a]==[x['date'] for x in b]
            deltas.setdefault(scenario,[]).append(np.array([[x[k]-y[k] for k in ('IC','RankIC')] for x,y in zip(a,b)]))
        print('MEMBERSHIP',s,flush=True)
    stats=[]
    for scenario,v in deltas.items():
        arr=np.mean(v,axis=0)
        for j,k in enumerate(('IC','RankIC')):
            for measure,x in [('model_gap',arr[:,j]),('change_in_gap_vs_original',arr[:,j]-np.mean(deltas['original'],axis=0)[:,j])]:
                if measure=='change_in_gap_vs_original' and scenario=='original':continue
                ci=blocks(x,20);stats.append(dict(scenario=scenario,metric=k,measure=measure,n_seeds=len(v),delta=float(x.mean()),ci_low=float(ci[0]),ci_high=float(ci[1])))
    csvout(OUT/'membership/per_seed.csv',rows);csvout(OUT/'membership/inference.csv',stats);csvout(OUT/'membership/discrepancies_local.csv',details)
    dump(OUT/'membership/status.json',dict(status='complete',test_discrepancies=890,finite_target_discrepancies=866,scope='CSI300 intersection with contemporaneous CSI800 records; no retraining; fixed prediction filtering and changed cross-stock context are separate.',n_seeds=len(deltas['original']),raw_data_not_published=True))
if __name__=='__main__':main()
