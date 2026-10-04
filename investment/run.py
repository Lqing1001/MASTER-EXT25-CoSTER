"""Portfolio evaluation from a manifest of final prediction archives."""
import argparse,json,sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from investment.backtest import metric,backtest,longonly,prices_for,regression

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--database',type=Path,required=True);p.add_argument('--factors',type=Path);p.add_argument('--output-dir',type=Path,required=True)
 a=p.parse_args();manifest=pd.read_csv(a.manifest)
 if not {'universe','model','seed','path'}.issubset(manifest):raise ValueError('Manifest columns: universe,model,seed,path')
 if manifest.duplicated(['universe','model','seed']).any():raise ValueError('Duplicate runs')
 a.output_dir.mkdir(parents=True,exist_ok=True);summ=[];daily=[];audit={}
 con=sqlite3.connect(a.database.resolve().as_uri()+'?mode=ro',uri=True)
 for r in manifest.itertuples():
  path=Path(r.path);path=path if path.is_absolute() else a.manifest.resolve().parent/path
  with np.load(path,allow_pickle=False) as z:v={k:z[k] for k in ['dates','instruments','labels','predictions']}
  dates=np.unique(v['dates']);inst=np.array([s.decode() if isinstance(s,bytes) else str(s) for s in v['instruments']]);inst=np.array([s[2:] for s in inst]);symbols=sorted(set(inst));lookup={s:i for i,s in enumerate(symbols)};ri=np.searchsorted(dates,v['dates']);ci=np.array([lookup[s] for s in inst])
  if len(set(zip(ri,ci)))!=len(ri) or not np.isfinite(v['predictions']).all():raise ValueError('Duplicate or nonfinite predictions')
  prices,trade=prices_for(con,dates,symbols)
  use=(ri+5<len(dates))&np.isfinite(v['labels']);rr=ri[use];cc=ci[use];err=np.abs(prices[rr+5,cc]/prices[rr+1,cc]-1-v['labels'][use])
  if not len(err) or not np.isfinite(err).all() or err.max()>1e-5:raise ValueError('Price/label alignment failed')
  scores=np.full(prices.shape,np.nan);scores[ri,ci]=v['predictions']
  index={'csi300':'000300','csi800':'000906'}[r.universe]
  b=pd.read_sql_query('SELECT trade_date,close FROM index_daily WHERE index_code=? ORDER BY trade_date',con,params=[index]);b['date']=b.trade_date.str.replace('-','').astype(int);b=b.set_index('date')['close'].reindex(dates)
  if b.isna().any():raise ValueError('Missing benchmark prices')
  br=b.pct_change().iloc[1:].to_numpy();bm=metric(br)
  for strategy,fn in [('longshort',backtest),('longonly',longonly)]:
   x,info=fn(scores,prices,trade);audit[f'{r.universe}_{r.model}_{r.seed}_{strategy}']={'price_label_max_error':float(err.max()),**info}
   for cost in [0,5,10,20,30]:
    ret=x[:,0]-cost/10000*x[:,2];row=dict(universe=r.universe,model=r.model,seed=r.seed,strategy=strategy,cost=cost,**metric(ret),turnover=float(x[:,1].mean()))
    if strategy=='longonly':
     active=ret-br;row.update(benchmark_CAGR=bm['CAGR'],annual_active_mean=float(active.mean()*252),IR=float(active.mean()/active.std(ddof=1)*np.sqrt(252)))
    summ.append(row)
   for i,row in enumerate(x):daily.append(dict(universe=r.universe,model=r.model,seed=r.seed,strategy=strategy,date=int(dates[i+1]),gross_return=row[0],net_return_10bps=row[0]-.001*row[2],turnover=row[1],benchmark_return=br[i]))
 con.close();s=pd.DataFrame(summ);d=pd.DataFrame(daily);s.to_csv(a.output_dir/'per_seed.csv',index=False);d.to_csv(a.output_dir/'daily.csv',index=False)
 s.groupby(['universe','model','strategy','cost']).agg({k:['mean','std'] for k in ['CAGR','Sharpe','max_drawdown','turnover']}).to_csv(a.output_dir/'summary.csv')
 if a.factors:
  f=pd.read_excel(a.factors) if a.factors.suffix=='.xlsx' else pd.read_csv(a.factors);f['date']=f.date.astype(int);f=f.set_index('date');regs=[]
  ls=d[d.strategy=='longshort']
  for (u,m),g in ls.groupby(['universe','model']):
   h=g.groupby('date')[['gross_return','net_return_10bps']].mean();ff=f.reindex(h.index)[['mktrf','SMB','VMG']]
   if ff.isna().any().any():raise ValueError('Missing CH-3 factor dates')
   for cost,col in [(0,'gross_return'),(10,'net_return_10bps')]:regs.append(dict(universe=u,model=m,cost=cost,seed='seed_average',**regression(h[col].to_numpy(),ff.to_numpy(),10)))
  # Same-seed pairing precedes daily averaging for incremental alpha.
  for u,g in ls.groupby('universe'):
   for cost,col in [(0,'gross_return'),(10,'net_return_10bps')]:
    wide=g.pivot(index=['seed','date'],columns='model',values=col)
    if {'CoSTER','MASTER'}.issubset(wide.columns):
     pair=wide[['CoSTER','MASTER']]
     if pair.isna().any().any():raise ValueError('Incomplete paired portfolio runs')
     diff=(pair.CoSTER-pair.MASTER).groupby('date').mean();ff=f.reindex(diff.index)[['mktrf','SMB','VMG']]
     regs.append(dict(universe=u,model='CoSTER-minus-MASTER',cost=cost,seed='seed_average',**regression(diff.to_numpy(),ff.to_numpy(),10)))
  pd.DataFrame(regs).to_csv(a.output_dir/'factor_regressions.csv',index=False)
 (a.output_dir/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
if __name__=='__main__':main()
