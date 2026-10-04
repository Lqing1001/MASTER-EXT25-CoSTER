"""Validation-only Top-k and score-weight sensitivity (CSI 300, seed 0)."""
import argparse,csv,json,copy
from pathlib import Path
import numpy as np
import torch
import CoSTER as C

def fused_metrics(p,m,alpha):
 if not np.array_equal(p['dates'],m['dates']) or not np.array_equal(p['instruments'],m['instruments']) or not np.allclose(p['labels'],m['labels'],equal_nan=True):raise ValueError('Unmatched arrays')
 rows=[];ds=p['dates'];bounds=np.r_[0,np.flatnonzero(np.diff(ds))+1,len(ds)]
 for a,b in zip(bounds[:-1],bounds[1:]):
  x=p['predictions'][a:b].astype(float);z=m['predictions'][a:b].astype(float);x=(x-x.mean())/max(x.std(),1e-12);z=(z-z.mean())/max(z.std(),1e-12);v=alpha*x+(1-alpha)*z;y=p['labels'][a:b]
  rows.append({'IC':C.correlation(v,y),'RankIC':C.correlation(v,y,True)})
 r=C.summarize_daily(rows);return {'IC':r['IC'],'RankIC':r['RankIC'],'V':(r['IC']+r['RankIC'])/2}

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--device',default='cuda');a=p.parse_args();torch.set_num_threads(1)
 if a.output_dir.exists() and any(a.output_dir.iterdir()):raise FileExistsError('Use an empty output directory')
 a.output_dir.mkdir(parents=True,exist_ok=True);device=torch.device(a.device);model=C.CoSTER.from_run_dir(a.run_dir,'csi300',0,device);store=C.UniverseStore(a.dataset_root,'csi300')
 _,mp=C.evaluate(model.master,store,C.VALID_START,C.VALID_END,device,None);_,ep=C.evaluate(model.continuous,store,C.VALID_START,C.VALID_END,device,None);rows=[]
 for k in [1,2,4,8]:
  if k==1:pred=ep
  else:
   anchor=copy.deepcopy(model.continuous.base);C.seed_all(0);expert=C.SparseExpert(anchor,'deformable',selected=k).to(device);ps=[p for p in expert.parameters() if p.requires_grad];opt=torch.optim.AdamW(ps,lr=1e-4,weight_decay=1e-4)
   C.train_predictor(expert,f'Top-{k}',store,store.dates_between(C.TRAIN_START,C.TRAIN_END),opt,device,10,10,0,None,None)
   torch.save(expert.state_dict(),a.output_dir/f'top{k}.pt');_,pred=C.evaluate(expert,store,C.VALID_START,C.VALID_END,device,None)
  rows.append({'parameter':'Top-k','setting':k,**fused_metrics(pred,mp,.5)})
 for alpha in [0,.25,.5,.75,1]:rows.append({'parameter':'alpha','setting':alpha,**fused_metrics(ep,mp,alpha)})
 with (a.output_dir/'sensitivity.csv').open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=['parameter','setting','IC','RankIC','V']);w.writeheader();w.writerows(rows)
 (a.output_dir/'selection.json').write_text(json.dumps({'selected_k':max(rows[:4],key=lambda r:r['V'])['setting'],'selection':'Full CoSTER validation V; sparse stage checkpoint selected by branch V','universe':'csi300','seed':0,'alpha':.5,'test_used':False},indent=2),encoding='utf8')
if __name__=='__main__':main()
