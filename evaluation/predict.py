"""Evaluate final CoSTER and its two components on complete cross-sections."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import CoSTER as C

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True)
 p.add_argument('--universe',choices=['csi300','csi800'],required=True);p.add_argument('--seed',type=int,default=0)
 p.add_argument('--split',choices=['validation','test'],default='test');p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--max-eval-days',type=int)
 a=p.parse_args();torch.set_num_threads(1)
 if a.output_dir.exists() and any(a.output_dir.iterdir()):raise FileExistsError('Use an empty output directory')
 model=C.CoSTER.from_run_dir(a.run_dir,a.universe,a.seed,a.device);store=C.UniverseStore(a.dataset_root,a.universe)
 start,end=(C.VALID_START,C.VALID_END) if a.split=='validation' else (C.TEST_START,C.TEST_END)
 a.output_dir.mkdir(parents=True,exist_ok=True);summary={}
 for name,module in [('MASTER',model.master),('Continuous',model.continuous),('CoSTER',model)]:
  rows,arrays=C.evaluate(module,store,start,end,torch.device(a.device),a.max_eval_days)
  C.write_daily_csv(a.output_dir/f'{name}_daily.csv',rows);np.savez_compressed(a.output_dir/f'{name}_predictions.npz',**arrays)
  summary[name]=C.metrics_by_period(rows)
 (a.output_dir/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
