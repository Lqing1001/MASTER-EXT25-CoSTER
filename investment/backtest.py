"""Final four-cohort portfolio accounting and CH-3 HAC regression."""
import math
import numpy as np
import pandas as pd

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

def test_longshort():
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

def regression(y,f,lag=10):
 X=np.column_stack([np.ones(len(y)),f]);n,k=X.shape
 assert np.isfinite(X).all() and np.isfinite(y).all() and np.linalg.matrix_rank(X)==k
 beta=np.linalg.lstsq(X,y,rcond=None)[0];e=y-X@beta;xe=X*e[:,None]
 meat=xe.T@xe
 for l in range(1,lag+1):
  cross=xe[l:].T@xe[:-l];meat+=(1-l/(lag+1))*(cross+cross.T)
 bread=np.linalg.inv(X.T@X);cov=bread@meat@bread*n/(n-k);se=np.sqrt(np.diag(cov));t=beta/se
 return {'n':n,'lag':lag,'alpha_daily':beta[0],'alpha_annual':252*beta[0],'alpha_se_daily':se[0],'t_alpha':t[0],'p_two_sided_normal':math.erfc(abs(t[0])/math.sqrt(2)),'alpha_ci95_annual_low':252*(beta[0]-1.96*se[0]),'alpha_ci95_annual_high':252*(beta[0]+1.96*se[0]),'beta_MKT':beta[1],'beta_SMB':beta[2],'beta_VMG':beta[3],'R2':1-e@e/((y-y.mean())@(y-y.mean()))}

def longonly(scores,prices,trade):
 T,N=scores.shape;cohorts=[];prev=np.zeros(N);rows=[];blocked=0
 for t in range(1,T):
  ret=np.divide(prices[t],prices[t-1],out=np.ones(N),where=np.isfinite(prices[t])&np.isfinite(prices[t-1]))-1
  gross=prev@ret;drift=prev*(1+ret)/(1+gross);w=np.zeros(N)
  if t+4<T:
   elig=np.flatnonzero(np.isfinite(scores[t-1]));n=int(.1*len(elig));assert n>0
   ix=elig[np.argsort(scores[t-1,elig],kind='stable')][-n:];w[ix]=1/n;w[~trade[t]]=0
  cohorts.append(w)
  if len(cohorts)>4:cohorts.pop(0)
  target=np.sum(cohorts,axis=0)/4
  blocked+=np.count_nonzero((np.abs(target-drift)>1e-12)&~trade[t]);target[~trade[t]]=drift[~trade[t]]
  # Reserve frozen positions first, scale liquid targets to avoid implicit borrowing.
  frozen=target[~trade[t]].sum();liquid=target[trade[t]].sum()
  assert frozen<=1+1e-10
  if liquid>1-frozen:target[trade[t]]*=max(0,1-frozen)/liquid
  assert target.min()>=0 and target.sum()<=1+1e-10
  q=np.abs(target-drift).sum();rows.append([gross,q/2,q,target.sum(),1-prev.sum()]);prev=target
 a=np.array(rows);return a,{'blocked_security_days':int(blocked),'terminal_stock_weight':float(prev.sum()),'max_stock_weight':float(a[:,3].max())}

def test_longonly_and_hac():
 rng=np.random.default_rng(17);f=rng.normal(size=(200,3));y=.002+f@np.array([.1,.2,.3])+rng.normal(scale=.01,size=200)
 r=regression(y,f,0);X=np.column_stack([np.ones(200),f]);b=np.linalg.lstsq(X,y,rcond=None)[0];e=y-X@b;inv=np.linalg.inv(X.T@X)
 brute=sum(e[i]**2*np.outer(X[i],X[i]) for i in range(200));v=inv@brute@inv*200/196
 assert abs(r['alpha_se_daily']-np.sqrt(v[0,0]))<1e-12
 # HAC10 check via an independently built time covariance kernel.
 K=np.maximum(0,1-np.abs(np.arange(200)[:,None]-np.arange(200)[None,:])/11)
 xe=X*e[:,None];v=inv@(xe.T@K@xe)@inv*200/196
 assert abs(regression(y,f,10)['alpha_se_daily']-np.sqrt(v[0,0]))<1e-12
 s=np.tile(np.arange(20.),(12,1));p=np.ones_like(s);p[:,18:]=1.01**np.arange(12)[:,None]
 a,_=longonly(s,p,np.ones_like(s,dtype=bool));assert abs(a[0,2]-.25)<1e-12 and abs(a[4,0]-.01)<1e-12 and abs(a[-1,3])<1e-12
