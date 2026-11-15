from common import *
import sqlite3,pandas as pd
sys.path.insert(0,str(ROOT/'experiments/master_ext_reproduction'))
from recompute_long_short_20260917 import prices_for,metric,backtest
from supplement_portfolios_20260920 import longonly,regression
from benchmark_compare import benchmark_returns

def simulate(scores,prices,trade,mandate,replacement):
    T,N=scores.shape;prev=np.zeros(N);cohorts=[];ideal_cohorts=[];rows=[];weights=[];frozen_age=np.zeros(N,dtype=int)
    for t in range(1,T):
        ret=np.divide(prices[t],prices[t-1],out=np.ones(N),where=np.isfinite(prices[t])&np.isfinite(prices[t-1]))-1
        gross=float(prev@ret);drift=prev*(1+ret)/(1+gross)
        intended=np.zeros(N);w=np.zeros(N);failed=0;orders=0
        if t+4<T:
            eligible=np.flatnonzero(np.isfinite(scores[t-1]));n=int(len(eligible)*.1)
            if n:
                ix=eligible[np.argsort(scores[t-1,eligible],kind='stable')]
                intended[ix[-n:]]=(1 if mandate=='longonly' else .5)/n
                if mandate=='longshort':intended[ix[:n]]=-.5/n
                orders=int(np.count_nonzero(intended));failed=int(np.count_nonzero(intended[~trade[t]]))
                if replacement:
                    available=ix[trade[t,ix]];longix=available[-min(n,len(available)):]
                    w[longix]=(1 if mandate=='longonly' else .5)/n
                    if mandate=='longshort':
                        shortix=available[:min(n,max(0,len(available)-len(longix)))];w[shortix]=-.5/n
                else:w=intended.copy();w[~trade[t]]=0
        cohorts.append(w);ideal_cohorts.append(intended)
        if len(cohorts)>4:cohorts.pop(0);ideal_cohorts.pop(0)
        target=np.sum(cohorts,axis=0)/4;ideal=np.sum(ideal_cohorts,axis=0)/4;pre_freeze=target.copy()
        blocked=int(np.count_nonzero((np.abs(target-drift)>1e-12)&~trade[t]));target[~trade[t]]=drift[~trade[t]]
        if mandate=='longonly':
            frozen=target[~trade[t]].sum();liquid=target[trade[t]].sum();assert frozen<=1+1e-10
            if liquid>1-frozen:target[trade[t]]*=max(0,1-frozen)/liquid
            assert target.min()>=0 and target.sum()<=1+1e-10
        assert np.allclose(target[~trade[t]],drift[~trade[t]])
        frozen_mask=(~trade[t])&(np.abs(target)>1e-12);frozen_age=np.where(frozen_mask,frozen_age+1,0)
        traded=float(np.abs(target-drift).sum())
        rows.append(dict(gross=gross,net=gross-.001*traded,turnover=.5*traded,
            intended_long=float(ideal[ideal>0].sum()),intended_short=float(-ideal[ideal<0].sum()),
            actual_long=float(target[target>0].sum()),actual_short=float(-target[target<0].sum()),
            gross_exposure=float(np.abs(target).sum()),net_exposure=float(target.sum()),
            longonly_cash=float(1-target.sum()) if mandate=='longonly' else None,
            frozen_abs_exposure=float(np.abs(target[frozen_mask]).sum()),max_frozen_duration=int(frozen_age.max()),
            failed_original_entries=failed,intended_entry_names=orders,blocked_security_orders=blocked,
            target_actual_L1=float(np.abs(target-ideal).sum()),freeze_adjustment_L1=float(np.abs(target-pre_freeze).sum())))
        weights.append(target.copy());prev=target
    return rows,np.array(weights)

def checks():
    scores=np.tile(np.arange(20.),(14,1));prices=np.ones_like(scores);prices[:,18:]=1.01**np.arange(14)[:,None];prices[:,:2]=.99**np.arange(14)[:,None]
    tr=np.ones_like(scores,dtype=bool);tr[4:7,19]=False;prices[4:7,19]=prices[3,19]
    for mandate,fn in [('longshort',backtest),('longonly',longonly)]:
        r,w=simulate(scores,prices,tr,mandate,False);ref,_=fn(scores,prices,tr)
        assert np.allclose([v['gross'] for v in r],ref[:,0],rtol=0,atol=1e-12)
        assert np.allclose([v['turnover'] for v in r],ref[:,1],rtol=0,atol=1e-12)
        assert abs(r[0]['gross'])<1e-12 and abs(r[0]['turnover']-.125)<1e-12
        assert np.abs(w[-1]).sum()<1e-12
        r2,w2=simulate(scores,prices,tr,mandate,True)
        assert all(v['failed_original_entries']>=0 for v in r2)

def main():
    setup();checks();d=OUT/'portfolio';d.mkdir(exist_ok=True);summ=[];expo=[];regs=[];daily_records=[]
    con=sqlite3.connect(Path(os.environ.get('COSTER_PRICE_DB',str(ROOT/'datasets/master_ext_clean_v1/intermediate/master_ext_clean.sqlite'))).as_uri()+'?mode=ro',uri=True)
    factors=pd.read_excel(Path(os.environ.get('COSTER_FACTOR_FILE',str(ROOT/'results/additional_experiments_20260920/CH3_factors_daily_202608.xlsx')))).set_index('date');factors.index=factors.index.astype(int)
    tr_returns=benchmark_returns()
    available_index_codes=pd.read_sql_query('select distinct index_code from index_daily',con).index_code.tolist()
    for u in UNIVERSES:
        ref=loadnp(predpath(u,0,'master'));dates=np.unique(ref['dates']);symbols=sorted(set(bytes(x).decode()[-6:] for x in ref['instruments']));pos={x:i for i,x in enumerate(symbols)}
        ri=np.searchsorted(dates,ref['dates']);ci=np.array([pos[bytes(x).decode()[-6:]] for x in ref['instruments']]);prices,trade=prices_for(con,dates,symbols)
        for s in SEEDS:
            if not predpath(u,s,'full').exists():continue
            mp=loadnp(predpath(u,s,'master'));fp=fusion(mp,loadnp(predpath(u,s,'full')));aligned(mp,ref)
            for name,p in [('MASTER',mp),('CoSTER',fp)]:
                sc=np.full(prices.shape,np.nan);sc[ri,ci]=p['predictions']
                for mandate,fn in [('longshort',backtest),('longonly',longonly)]:
                    for replace in (False,True):
                        scenario='replacement' if replace else 'freeze_cash';rs,w=simulate(sc,prices,trade,mandate,replace)
                        if not replace:
                            old,_=fn(sc,prices,trade)
                            assert np.allclose(old[:,0],[r['gross'] for r in rs],atol=1e-12,rtol=0)
                            assert np.allclose(old[:,1],[r['turnover'] for r in rs],atol=1e-12,rtol=0)
                        tag=dict(universe=u,seed=s,model=name,mandate=mandate,scenario=scenario)
                        for cost in (0,5,10,20,30):
                            net=np.array([r['gross']-2*cost/10000*r['turnover'] for r in rs])
                            bm=dict(TR_index_CAGR=None,net_excess_annual=None,net_IR=None)
                            if mandate=='longonly':
                                br=tr_returns[u].reindex(dates[1:]);assert br.notna().all()
                                active=net-br.to_numpy()
                                bm=dict(TR_index_CAGR=metric(br.to_numpy())['CAGR'],net_excess_annual=float(252*active.mean()),net_IR=float(np.sqrt(252)*active.mean()/active.std(ddof=1)))
                            summ.append(dict(**tag,cost_bps=cost,**metric(net),**bm))
                        for col in ('actual_long','actual_short','gross_exposure','net_exposure','frozen_abs_exposure','target_actual_L1','max_frozen_duration'):
                            v=np.array([r[col] for r in rs]);expo.append(dict(**tag,quantity=col,mean=float(v.mean()),p95=float(np.quantile(v,.95)),max=float(v.max()),terminal=float(v[-1])))
                        for day,r in zip(dates[1:],rs):daily_records.append(dict(**tag,date=int(day),**r))
                        wd=d/u/f'seed{s}';wd.mkdir(parents=True,exist_ok=True)
                        np.savez_compressed(wd/f'{name}_{mandate}_{scenario}_weights.npz',dates=dates[1:],symbols=np.array(symbols),weights=w)
            print('PORTFOLIO',u,s,flush=True)
    con.close();df=pd.DataFrame(daily_records)
    for (u,mandate,scenario),g in df.groupby(['universe','mandate','scenario']):
        a=g[g.model=='CoSTER'].groupby('date').net.mean();b=g[g.model=='MASTER'].groupby('date').net.mean();assert a.index.equals(b.index)
        ff=factors.reindex(a.index);assert ff[['mktrf','SMB','VMG']].notna().all().all()
        for lag in (5,10,20):regs.append(dict(universe=u,mandate=mandate,scenario=scenario,**regression((a-b).to_numpy(),ff[['mktrf','SMB','VMG']].to_numpy(),lag)))
    csvout(d/'performance.csv',summ);csvout(d/'exposure_summary.csv',expo);csvout(d/'incremental_alpha.csv',regs);df.to_csv(d/'daily_ledger.csv',index=False,encoding='utf-8-sig')
    dump(d/'status.json',dict(status='execution_and_attribution_complete',parity='synthetic and every freeze_cash run reproduce original functions',total_return_benchmark='Raw CSMAR H00300 and H00906; full 969-price/968-return coverage; see data_audit/total_return_audit.json',local_filtered_database_index_codes=available_index_codes,stock_adjustment_definition='CSMAR BwardFactor = previous close / exchange reference previous close on adjustment day; cumulative product. This alone is not proof of an exact dividend-reinvestment total-return convention.',short_selling='theoretical unlimited borrowing; no point-in-time loan inventory/fees available',paper_modified=False))
if __name__=='__main__':main()
