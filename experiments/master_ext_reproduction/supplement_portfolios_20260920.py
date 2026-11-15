"""CH-3 attribution and long-only decile evaluation. Run from workspace root."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,sqlite3,hashlib,math,sys
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent))
from recompute_long_short_20260917 import prices_for,metric
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/additional_experiments_20260920'
MODELS=['MASTER','MX','HOM','DynaFuse','TOP1']

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

def tests():
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

def main():
 tests();OUT.mkdir(exist_ok=True)
 factorfile=OUT/'CH3_factors_daily_202608.xlsx';f=pd.read_excel(factorfile);f['date']=f.date.astype(int);assert f.date.is_unique
 f=f.set_index('date')[['rf_dly','mktrf','SMB','VMG']].astype(float)
 assert f.abs().max().max()<.3 and .000001<f.rf_dly.mean()<.001
 ls=pd.read_csv(ROOT/'results/long_short_20260917/daily_returns.csv');regrows=[]
 # Inference on the datewise average of paired runs, not on 3N pooled observations or mean t values.
 for (u,m),g in ls.groupby(['universe','model']):
  for seed in ['seed_average',0,1,2]:
   h=g.groupby('date')[['gross_return','net_return_10bps']].mean() if seed=='seed_average' else g[g.seed==seed].set_index('date')[['gross_return','net_return_10bps']]
   h=h.sort_index();ff=f.reindex(h.index);assert ff.notna().all().all() and len(h)==968
   for cost,col in [(0,'gross_return'),(10,'net_return_10bps')]:
    for lag in ([5,10,20] if seed=='seed_average' else [10]):
     for convention in (['spread_pnl','subtract_full_rf'] if seed=='seed_average' and lag==10 else ['spread_pnl']):
      y=h[col].to_numpy()-(ff.rf_dly.to_numpy() if convention=='subtract_full_rf' else 0)
      regrows.append({'universe':u,'model':m,'seed':seed,'cost_bps':cost,'convention':convention,**regression(y,ff[['mktrf','SMB','VMG']].to_numpy(),lag)})
 reg=pd.DataFrame(regrows);reg.to_csv(OUT/'factor_regressions.csv',index=False,encoding='utf-8-sig')
 sources=json.loads((ROOT/'results/long_short_20260917/source_map.json').read_text());runs={};daily=[];summ=[]
 con=sqlite3.connect((ROOT/'datasets/master_ext_clean_v1/intermediate/master_ext_clean.sqlite').as_uri()+'?mode=ro',uri=True)
 for u,idx in [('csi300','000300'),('csi800','000906')]:
  arrays={}
  for model in ['MASTER','TOP1','XGB']:
   for seed in range(3):
    with np.load(sources[u][model][str(seed)],allow_pickle=False) as z:arrays[model,seed]={k:z[k] for k in z.files}
  ref=arrays['MASTER',0];dates=np.unique(ref['dates']);assert len(dates)==969
  inst=np.array([x.decode()[2:] for x in ref['instruments']]);symbols=sorted(set(inst));spos={s:i for i,s in enumerate(symbols)};ri=np.searchsorted(dates,ref['dates']);ci=np.array([spos[s] for s in inst])
  prices,trade=prices_for(con,dates,symbols);raw={};normed={}
  for key,z in arrays.items():
   assert np.array_equal(z['dates'],ref['dates']) and np.array_equal(z['instruments'],ref['instruments']) and np.allclose(z['labels'],ref['labels'],equal_nan=True)
   v=np.full(prices.shape,np.nan);v[ri,ci]=z['predictions'].astype(float);raw[key]=v;normed[key]=(v-np.nanmean(v,axis=1,keepdims=True))/np.maximum(np.nanstd(v,axis=1,keepdims=True),1e-12)
  bench=pd.read_sql_query('SELECT trade_date,close FROM index_daily WHERE index_code=? ORDER BY trade_date',con,params=[idx]);bench['date']=bench.trade_date.str.replace('-','').astype(int);b=bench.set_index('date')['close'].reindex(dates);assert b.notna().all();br=b.pct_change().iloc[1:].to_numpy();bm=metric(br)
  for seed in range(3):
   m=normed['MASTER',seed];e=normed['TOP1',seed];x=normed['XGB',seed]
   variants={'MASTER':raw['MASTER',seed],'TOP1':raw['TOP1',seed],'DynaFuse':.5*(m+e),'MX':.5*(m+x),'HOM':.5*(m+normed['MASTER',(seed+1)%3])}
   for model,sc in variants.items():
    a,info=longonly(sc,prices,trade);runs[f'{u}_{model}_{seed}']=info
    for cost in [0,5,10,20,30]:
     ret=a[:,0]-cost/10000*a[:,2];active=ret-br;v=metric(ret)
     summ.append({'universe':u,'model':model,'seed':seed,'cost_bps':cost,**v,'benchmark_CAGR':bm['CAGR'],'CAGR_difference':v['CAGR']-bm['CAGR'],'annual_active_mean':active.mean()*252,'information_ratio':active.mean()/active.std(ddof=1)*np.sqrt(252),'relative_wealth_CAGR':(np.prod((1+ret)/(1+br)))**(252/len(ret))-1,'daily_turnover':a[:,1].mean()})
    for i,row in enumerate(a):daily.append({'universe':u,'model':model,'seed':seed,'date':int(dates[i+1]),'gross_return':row[0],'net_return_10bps':row[0]-.001*row[2],'daily_turnover':row[1],'benchmark_return':br[i],'stock_weight':row[3],'cash_weight_before_return':row[4]})
   print('longonly',u,seed,'done',flush=True)
 con.close();d=pd.DataFrame(daily);s=pd.DataFrame(summ);d.to_csv(OUT/'longonly_daily.csv',index=False,encoding='utf-8-sig');s.to_csv(OUT/'longonly_per_seed.csv',index=False,encoding='utf-8-sig')
 numeric=['CAGR','Sharpe','daily_turnover','benchmark_CAGR','annual_active_mean','information_ratio','CAGR_difference','relative_wealth_CAGR']
 agg=s.groupby(['universe','model','cost_bps'])[numeric].agg(['mean','std']);agg.columns=['_'.join(c) for c in agg.columns];agg.reset_index().to_csv(OUT/'longonly_summary.csv',index=False,encoding='utf-8-sig')
 regmain=reg[(reg.seed=='seed_average')&(reg.lag==10)&(reg.convention=='spread_pnl')]
 lines=['| 股票池 | 模型 | 毛 alpha（年化%） | HAC t | 净 alpha（年化%） | HAC t |','|---|---|---:|---:|---:|---:|']
 lo=['| 股票池 | 模型 | 毛 CAGR% | 指数 CAGR% | 毛超额年化% | 净 CAGR% | 净超额年化% | 净 IR |','|---|---|---:|---:|---:|---:|---:|---:|']
 for u in ['csi300','csi800']:
  for m in MODELS:
   r=regmain[(regmain.universe==u)&(regmain.model==m)].set_index('cost_bps');g=agg.loc[u,m,0];n=agg.loc[u,m,10]
   lines.append(f'| {u.upper()} | {m} | {100*r.loc[0,"alpha_annual"]:.2f} | {r.loc[0,"t_alpha"]:.2f} | {100*r.loc[10,"alpha_annual"]:.2f} | {r.loc[10,"t_alpha"]:.2f} |')
   lo.append(f'| {u.upper()} | {m} | {100*g.CAGR_mean:.2f} | {100*g.benchmark_CAGR_mean:.2f} | {100*g.annual_active_mean_mean:.2f} | {100*n.CAGR_mean:.2f} | {100*n.annual_active_mean_mean:.2f} | {n.information_ratio_mean:.2f} |')
 (OUT/'factor_table.md').write_text('\n'.join(lines),encoding='utf8');(OUT/'longonly_table.md').write_text('\n'.join(lo),encoding='utf8')
 assert len(d)==29040 and d.notna().all().all() and d.stock_weight.between(0,1+1e-10).all()
 assert np.allclose(d.gross_return-d.net_return_10bps,.002*d.daily_turnover)
 audit={'factor_source':'https://www.mingshiim.com/static/CH3_factors_daily_202608.xlsx','factor_page':'https://www.mingshiim.com/database','factor_sha256':hashlib.sha256(factorfile.read_bytes()).hexdigest(),'factor_units':'decimal returns, no percent scaling','factor_first_date':int(f.index.min()),'factor_last_date':int(f.index.max()),'regression_dates':[20220105,20251231],'n_dates':968,'missing_factor_dates':0,'HAC':'Bartlett kernel, lag 10, n/(n-k) correction; asymptotic normal p-values','longonly_runs':runs,'validation':'HAC kernel cross-check, lag0 HC1, synthetic longonly holding timing, full factor coverage, long-only nonnegative unlevered weights, cost identity PASS','source_map':sources}
 (OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print('\n'.join(lines));print('\n'.join(lo))
if __name__=='__main__':main()

