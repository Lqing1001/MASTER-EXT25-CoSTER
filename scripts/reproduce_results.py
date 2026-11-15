"""Recompute current paired inference and figures from aggregate public evidence."""
from pathlib import Path
import argparse,json,math
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT=Path(__file__).resolve().parents[1]
def hac(x,lag=10):
 n=len(x);c=x-x.mean();v=c@c/n
 for j in range(1,min(lag,n-1)+1):v+=2*(1-j/(lag+1))*(c[j:]@c[:-j])/n
 se=math.sqrt(max(v,0)/n);z=x.mean()/se
 return dict(delta=float(x.mean()),se=se,z=z,p=float(2*norm.sf(abs(z))))
def blocks(x):
 rng=np.random.default_rng(20261110);n=len(x);means=[]
 for _ in range(5000):
  start=rng.integers(n-20+1,size=math.ceil(n/20));ix=(start[:,None]+np.arange(20)).ravel()[:n];means.append(x[ix].mean())
 return np.quantile(means,[.025,.975])
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'runs/reproduced');a=p.parse_args();out=a.output_dir;out.mkdir(parents=True,exist_ok=True)
 source=ROOT/'reference';fd=source/'figure_data';daily=pd.read_csv(fd/'recomputed_daily_metrics.csv')
 assert set(daily.seed)==set(range(10))
 sizes=daily.groupby(['universe','seed','model']).size();assert len(sizes)==60 and (sizes==964).all()
 rows=[]
 for u in ['csi300','csi800']:
  for metric in ['IC','RankIC']:
   q=daily[daily.universe==u].pivot(index=['date','seed'],columns='model',values=metric)
   x=(q['M+B']-q.MASTER).groupby('date').mean().to_numpy();ci=blocks(x)
   rows.append(dict(universe=u,metric=metric,**hac(x),ci_low=ci[0],ci_high=ci[1]))
 result=pd.DataFrame(rows);expected=pd.read_csv(fd/'base_fusion_contrasts.csv')
 delta=np.max(np.abs(result[['delta','p','ci_low','ci_high']].values-expected[['delta','p','ci_low','ci_high']].values));assert delta<1e-10
 result.to_csv(out/'base_fusion_contrasts.csv',index=False)
 plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
 colors={'MASTER':'#1f77b4','CoSTER':'#ff7f0e'}
 def save(fig,name):fig.tight_layout();fig.savefig(out/(name+'.pdf'));fig.savefig(out/(name+'.png'),dpi=200);plt.close(fig)
 def dates(s):return pd.to_datetime(np.asarray(s).astype(str),format='%Y%m%d')
 def timeaxis(ax):ax.xaxis.set_major_locator(mdates.YearLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'));ax.grid(alpha=.18)
 fig,axs=plt.subplots(2,2,figsize=(8,5.6),sharey='row')
 for j,u in enumerate(['csi300','csi800']):
  for i,metric in enumerate(['IC','RankIC']):
   ax=axs[i,j];q=daily[daily.universe==u].groupby(['seed','model'])[metric].mean().unstack();vals=q[['MASTER','CoSTER']].values
   for v in vals:ax.plot([0,1],v,'o-',color='#999999',alpha=.6,lw=.8,ms=3)
   ax.plot([0,1],vals.mean(0),'o-',color='#ff7f0e',lw=2);ax.set_xticks([0,1],['MASTER','CoSTER']);ax.set_ylabel(metric);ax.set_title('CSI '+u[-3:]);ax.grid(axis='y',alpha=.18)
 save(fig,'fig_paired_seeds')
 fig,ax=plt.subplots(figsize=(8,3.8));rolling=[]
 for u,c in zip(['csi300','csi800'],colors.values()):
  q=daily[daily.universe==u].pivot(index='date',columns=['seed','model'],values='RankIC');diff=pd.DataFrame({s:q[s,'CoSTER']-q[s,'MASTER'] for s in range(10)})
  r=diff.rolling(252,min_periods=252).mean();x=dates(r.index)
  ax.plot(x,r.mean(axis=1),label='CSI '+u[-3:],color=c);ax.fill_between(x,r.min(axis=1),r.max(axis=1),color=c,alpha=.12)
  for dt,v in r.dropna().iterrows():rolling.append(dict(universe=u,date=dt,mean=v.mean(),min=v.min(),max=v.max()))
 ax.axhline(0,color='gray',ls='--');ax.set_ylabel('252-day mean RankIC difference');ax.legend(frameon=False);timeaxis(ax);save(fig,'fig_rolling_rankic');pd.DataFrame(rolling).to_csv(out/'rolling_rankic.csv',index=False)
 summ=pd.read_csv(source/'tables/summary.csv');fig,axs=plt.subplots(1,2,figsize=(8,3.9))
 for ax,u in zip(axs,['csi300','csi800']):
  q=summ[summ.universe==u].set_index('model');v=np.array([q.loc[m,'RankIC_mean'] for m in ['MASTER','M+B','M+B+D','CoSTER']])
  ax.bar([0,4],[v[0],v[-1]],color=list(colors.values()),width=.62)
  for j,x in enumerate(np.diff(v),1):
   ax.bar(j,x,bottom=v[j-1],color='#66a89e',width=.62);ax.text(j,v[j]+.0006,f'+{x:.6f}',ha='center',fontsize=8,rotation=20)
  for j,x in [(0,v[0]),(4,v[-1])]:ax.text(j,x+.0006,f'{x:.5f}',ha='center',fontsize=8)
  ax.set_xticks(range(5),['MASTER','+B','+D','+S','CoSTER']);ax.set_ylim(0,.079);ax.set_ylabel('Mean test RankIC');ax.set_title('CSI '+u[-3:]);ax.grid(axis='y',alpha=.18)
 save(fig,'fig_rankic_bridge')
 wealth=pd.read_csv(fd/'net_wealth_by_seed.csv');bench=pd.read_csv(fd/'benchmark_wealth.csv');fig,axs=plt.subplots(2,2,figsize=(8,6.4))
 for i,mandate in enumerate(['longshort','longonly']):
  for j,u in enumerate(['csi300','csi800']):
   ax=axs[i,j]
   for m,c in colors.items():
    q=wealth[(wealth.universe==u)&(wealth.mandate==mandate)&(wealth.model==m)].pivot(index='date',columns='seed',values='wealth');assert q.shape==(969,10);assert np.allclose(q.iloc[0],1)
    x=dates(q.index);ax.plot(x,q.mean(axis=1),label=m,color=c);ax.fill_between(x,q.min(axis=1),q.max(axis=1),color=c,alpha=.13)
   if i==1:ax.plot(dates(bench.date),bench[u],label='Total-return index',color='#555555',ls='--')
   ax.set_title('CSI '+u[-3:]+' '+mandate);ax.set_ylabel('Net wealth');timeaxis(ax);ax.legend(frameon=False,fontsize=8)
 save(fig,'fig_net_wealth_10seeds')
 costs=pd.read_csv(fd/'recomputed_cost_performance.csv');fig,axs=plt.subplots(1,2,figsize=(8,3.5))
 for ax,u in zip(axs,['csi300','csi800']):
  for m,c in colors.items():
   q=costs[(costs.universe==u)&(costs.mandate=='longshort')&(costs.model==m)].groupby('cost_bps').CAGR.agg(['mean','std'])
   ax.errorbar(q.index,100*q['mean'],100*q['std'],fmt='o-',capsize=3,color=c,label=m)
  ax.axhline(0,color='gray',ls='--');ax.set_xticks([0,5,10,20,30]);ax.set_xlabel('One-way cost (bp)');ax.set_ylabel('Net CAGR (%)');ax.set_title('CSI '+u[-3:]);ax.legend(frameon=False);ax.grid(alpha=.18)
 save(fig,'fig_cost_sensitivity')
 report=dict(paired_statistic_max_error=float(delta),seeds=10,dates=964,figures=5)
 (out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report))
if __name__=='__main__':main()
