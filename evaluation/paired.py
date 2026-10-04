"""Matched per-seed daily differences, averaged before time-series inference."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from evaluation.statistics import moving_block_interval,hac_mean_test

def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,nargs='+',required=True);p.add_argument('--baseline',type=Path,nargs='+',required=True);p.add_argument('--output-dir',type=Path,required=True)
 a=p.parse_args()
 if len(a.candidate)!=len(a.baseline):p.error('Provide one paired file per seed, in the same seed order')
 out={};a.output_dir.mkdir(parents=True,exist_ok=True)
 for metric in ['IC','RankIC']:
  ds=[];common=None
  for cp,bp in zip(a.candidate,a.baseline):
   c=pd.read_csv(cp).set_index('date')[metric];b=pd.read_csv(bp).set_index('date')[metric]
   if not c.index.is_unique or not c.index.equals(b.index):raise ValueError('Daily dates must match exactly')
   d=(c-b).replace([np.inf,-np.inf],np.nan).dropna()
   if common is None:common=d.index
   elif not common.equals(d.index):raise ValueError('Finite dates must match across seeds')
   ds.append(d.to_numpy())
  delta=np.mean(ds,axis=0);out[metric]={'ci95':moving_block_interval(delta),**hac_mean_test(delta),'training_seeds_count':len(ds),'block':20,'replicates':5000,'bootstrap_seed':20261001}
  df=pd.DataFrame(np.array(ds).T,index=common,columns=[f'pair{i}_delta' for i in range(len(ds))]);df['mean_delta']=delta;df.to_csv(a.output_dir/f'{metric}_paired_daily.csv')
 (a.output_dir/'paired.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
