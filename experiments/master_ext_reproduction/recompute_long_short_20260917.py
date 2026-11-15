"""Recompute theoretical long-short portfolios from the final manuscript predictions.
Run from the workspace root. No training or source data are modified.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='1'
from pathlib import Path
import json, hashlib, sqlite3, math
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/long_short_20260917'
COSTS=[0,5,10,20,30]

def metric(r):
    assert np.isfinite(r).all() and np.all(r>-1)
    nav=np.cumprod(1+r)
    return {'CAGR':float(nav[-1]**(252/len(r))-1),'annual_arithmetic':float(r.mean()*252),'Sharpe':float(r.mean()/r.std(ddof=1)*np.sqrt(252)) if r.std(ddof=1)>0 else 0.,'max_drawdown':float((nav/np.maximum.accumulate(np.r_[1,nav])[1:]-1).min())}

def backtest(scores, prices, tradeable):
    T,N=scores.shape
    cohort=[];previous=np.zeros(N); rows=[];blocked=0; skipped=0
    for t in range(1,T):
        ret=np.divide(prices[t],prices[t-1],out=np.ones(N),where=np.isfinite(prices[t])&np.isfinite(prices[t-1]))-1
        assert np.isfinite(ret).all()
        gross=float(previous@ret)
        drift=previous*(1+ret)/(1+gross)
        w=np.zeros(N)
        # Signal t-1 must have a full four-interval holding horizon within the data.
        if t+4<T:
            eligible=np.flatnonzero(np.isfinite(scores[t-1]))
            n=int(len(eligible)*.1)
            if n:
                ix=eligible[np.argsort(scores[t-1,eligible],kind='stable')]
                w[ix[:n]]=-.5/n;w[ix[-n:]]=.5/n
                skipped+=int(np.count_nonzero(w[~tradeable[t]]))
                # No reranking using next-day availability; failed entries stay cash.
                w[~tradeable[t]]=0
        cohort.append(w)
        if len(cohort)>4:cohort.pop(0)
        target=np.sum(cohort,axis=0)/4
        blocked+=int(np.count_nonzero((np.abs(target-drift)>1e-12)&~tradeable[t]))
        # Existing positions cannot be traded on missing/zero-volume days.
        target[~tradeable[t]]=drift[~tradeable[t]]
        traded=float(np.abs(target-drift).sum())
        rows.append([gross,.5*traded,traded,float(np.abs(target).sum()),float(target.sum()),float(np.abs(target-drift)[~tradeable[t]].sum())])
        previous=target
    a=np.asarray(rows)
    assert np.max(a[:,5])==0
    return a,{'blocked_security_days':blocked,'skipped_entry_names':skipped,'terminal_gross_exposure':float(np.abs(previous).sum()),'max_gross_exposure':float(a[:,3].max())}

def prices_for(con,dates,symbols):
    calendar=pd.to_datetime(dates.astype(str))
    q=pd.read_sql_query("SELECT trade_date,symbol,close,volume,amount FROM quotes WHERE trade_date>='2021-12-01' AND trade_date<='2025-12-31'",con)
    q=q[q.symbol.isin(symbols)];q['trade_date']=pd.to_datetime(q.trade_date)
    close=q.pivot(index='trade_date',columns='symbol',values='close').reindex(columns=symbols)
    volume=q.pivot(index='trade_date',columns='symbol',values='volume').reindex_like(close)
    amount=q.pivot(index='trade_date',columns='symbol',values='amount').reindex_like(close)
    f=pd.read_sql_query('SELECT trade_date,symbol,cumulative_backward_factor FROM factors',con)
    f=f[f.symbol.isin(symbols)];f['trade_date']=pd.to_datetime(f.trade_date)
    fac=f.pivot(index='trade_date',columns='symbol',values='cumulative_backward_factor').reindex(columns=symbols)
    fac=fac.reindex(fac.index.union(close.index)).sort_index().ffill().reindex(close.index).fillna(1.)
    valid=close.gt(0)&volume.gt(0)&amount.gt(0)
    marked=(close*fac).where(valid).ffill().reindex(calendar)
    trade=valid.reindex(calendar).fillna(False)
    return marked.to_numpy(),trade.to_numpy(dtype=bool)

def tests():
    # Constant signals, +1% long stock and -1% short stock daily: signed P&L +1% at full exposure.
    scores=np.tile(np.arange(20.),(12,1));p=np.ones((12,20));p[:,18:]=1.01**np.arange(12)[:,None];p[:,:2]=.99**np.arange(12)[:,None]
    a,_=backtest(scores,p,np.ones_like(p,dtype=bool))
    assert abs(a[0,0])<1e-14 and abs(a[0,2]-.25)<1e-14
    assert abs(a[4,0]-.01)<1e-12 and abs(a[-1,3])<1e-12
    assert np.allclose(a[:,2],2*a[:,1])
    # Suspension blocks turnover; changing signal only affects following close's target.
    tr=np.ones_like(p,dtype=bool);tr[5,19]=False
    b,info=backtest(scores,p,tr);assert info['blocked_security_days']>0 and np.all(b[:,5]==0)
    flat,_=backtest(scores,np.ones_like(p),np.ones_like(tr));assert np.all(flat[:,0]==0)
    assert abs(metric(np.full(252,.001))['CAGR']-((1.001)**252-1))<1e-12

def main():
    tests();OUT.mkdir(parents=True,exist_ok=True)
    sources=json.loads((ROOT/'audit_outputs/final_github_20260916/source_map.json').read_text())
    (OUT/'source_map.json').write_text(json.dumps(sources,indent=2))
    manifest={'predictions':[],'universes':{},'tests':'PASS'};summary=[];daily=[];label_audit={}
    con=sqlite3.connect((ROOT/'datasets/master_ext_clean_v1/intermediate/master_ext_clean.sqlite').as_uri()+'?mode=ro',uri=True)
    for u in ['csi300','csi800']:
        arrays={}
        for model in ['MASTER','TOP1','XGB']:
            for seed in range(3):
                p=Path(sources[u][model][str(seed)])
                with np.load(p,allow_pickle=False) as z:arrays[model,seed]={k:z[k] for k in z.files}
                manifest['predictions'].append({'universe':u,'model':model,'seed':seed,'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        ref=arrays['MASTER',0];dates=np.unique(ref['dates']);assert len(dates)==969
        for a in arrays.values():
            assert np.array_equal(a['dates'],ref['dates']) and np.array_equal(a['instruments'],ref['instruments'])
            assert np.allclose(a['labels'],ref['labels'],equal_nan=True)
            assert np.isfinite(a['predictions']).all()
        inst=np.array([x.decode()[2:] for x in ref['instruments']]);symbols=sorted(set(inst));spos={s:i for i,s in enumerate(symbols)}
        ri=np.searchsorted(dates,ref['dates']);ci=np.array([spos[s] for s in inst]);assert len(set(zip(ri,ci)))==len(ri)
        prices,tradeable=prices_for(con,dates,symbols)
        # Validate adjusted prices against the actual unnormalized forward-return labels.
        use=(ri+5<len(dates))&np.isfinite(ref['labels']);rr=ri[use];cc=ci[use]
        implied=prices[rr+5,cc]/prices[rr+1,cc]-1;err=np.abs(implied-ref['labels'][use]);finite=np.isfinite(err)
        label_audit[u]={'n':int(finite.sum()),'maximum_error':float(err[finite].max()),'fraction_error_lt_1e_6':float((err[finite]<1e-6).mean())}
        assert finite.all() and err.max()<1e-5,label_audit[u]
        raw={};norm={}
        for key,a in arrays.items():
            matrix=np.full(prices.shape,np.nan);matrix[ri,ci]=a['predictions'].astype(float);raw[key]=matrix
            norm[key]=(matrix-np.nanmean(matrix,axis=1,keepdims=True))/np.maximum(np.nanstd(matrix,axis=1,keepdims=True),1e-12)
        manifest['universes'][u]={'dates':len(dates),'start':int(dates[1]),'end':int(dates[-1]),'last_signal':int(dates[-6]),'symbols':len(symbols),'runs':{}}
        for seed in range(3):
            m=norm['MASTER',seed];e=norm['TOP1',seed];x=norm['XGB',seed]
            variants={'MASTER':raw['MASTER',seed],'MX':.5*(m+x),'HOM':.5*(m+norm['MASTER',(seed+1)%3]),'DynaFuse':.5*(m+e),'TOP1':raw['TOP1',seed]}
            for model,scores in variants.items():
                a,info=backtest(scores,prices,tradeable);manifest['universes'][u]['runs'][f'{model}_{seed}']=info
                for bps in COSTS:
                    net=a[:,0]-bps/10000*a[:,2]
                    v={'universe':u,'model':model,'seed':seed,'cost_bps_one_way':bps,'days':len(a),**metric(net),'daily_turnover':float(a[:,1].mean()),'annual_turnover':float(a[:,1].mean()*252),'mean_gross_exposure':float(a[:,3].mean())};summary.append(v)
                for i,row in enumerate(a):daily.append({'universe':u,'model':model,'seed':seed,'date':int(dates[i+1]),'gross_return':row[0],'turnover_half_L1':row[1],'traded_notional_L1':row[2],'net_return_10bps':row[0]-.001*row[2],'gross_exposure':row[3],'net_exposure':row[4]})
            print(u,'seed',seed,'done',flush=True)
    con.close();manifest['label_price_alignment']=label_audit
    (OUT/'audit.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    df=pd.DataFrame(summary);df.to_csv(OUT/'per_seed_costs.csv',index=False,encoding='utf-8-sig');pd.DataFrame(daily).to_csv(OUT/'daily_returns.csv',index=False,encoding='utf-8-sig')
    stats=df.groupby(['universe','model','cost_bps_one_way'])[['CAGR','annual_arithmetic','Sharpe','daily_turnover','annual_turnover','max_drawdown','mean_gross_exposure']].agg(['mean','std']);stats.columns=['_'.join(c) for c in stats.columns];stats.reset_index().to_csv(OUT/'summary_costs.csv',index=False,encoding='utf-8-sig')
    lines=['| 股票池 | 模型 | 年化收益（%） | Sharpe | 日均换手（%） | 扣费后年化收益（%） |','|---|---|---:|---:|---:|---:|'];latex=[]
    for u in ['csi300','csi800']:
        for m in ['MASTER','MX','HOM','DynaFuse','TOP1']:
            g=stats.loc[u,m,0];n=stats.loc[u,m,10]
            vals=[(g['CAGR_mean']*100,g['CAGR_std']*100),(g['Sharpe_mean'],g['Sharpe_std']),(g['daily_turnover_mean']*100,g['daily_turnover_std']*100),(n['CAGR_mean']*100,n['CAGR_std']*100)]
            text=[f'{a:.2f} ± {b:.2f}' for a,b in vals]
            lines.append('| '+' | '.join([u.upper(),m]+text)+' |');latex.append(u.upper()+' & '+m+' & '+' & '.join(f'${a:.2f}\\pm{b:.2f}$' for a,b in vals)+r' \\')
    (OUT/'table.md').write_text('\n'.join(lines),encoding='utf8')
    tex=r'''% Requires booktabs. Values are three-seed mean +/- sample SD.
\begin{table}[htbp]
\centering\small
\caption{Theoretical long--short portfolio performance during 2022--2025. Returns and turnover are percentages. Net CAGR assumes a one-way transaction cost of 10 basis points.}
\label{tab:long_short}
\begin{tabular}{llrrrr}
\toprule
Universe & Model & CAGR & Sharpe & Daily turnover & Net CAGR \\
\midrule
'''+ '\n'.join(latex)+r'''
\bottomrule
\end{tabular}
\end{table}
'''
    (OUT/'table.tex').write_text(tex,encoding='utf8')
    print('\n'.join(lines));print(label_audit)
if __name__=='__main__':main()
