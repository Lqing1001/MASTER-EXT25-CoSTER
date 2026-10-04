"""Serial reproduction of the final model and baseline protocol."""
import argparse,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--output-root',type=Path,default=ROOT/'runs/coster');p.add_argument('--checkpoint-root',type=Path,help='Published checkpoints root: use each frozen encoder for matched training')
 p.add_argument('--universes',nargs='+',choices=['csi300','csi800'],default=['csi300','csi800']);p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);p.add_argument('--include-baselines',action='store_true');p.add_argument('--dry-run',action='store_true')
 a=p.parse_args();env=os.environ.copy();env['MASTER_EXT_DATASET_ROOT']=str(a.dataset_root.resolve())
 def run(args):
  cmd=[sys.executable]+list(map(str,args));print(subprocess.list2cmdline(cmd),flush=True)
  if not a.dry_run:subprocess.run(cmd,cwd=ROOT,env=env,check=True)
 for u in a.universes:
  for seed in a.seeds:
   base=a.output_root.resolve()/u/f'seed{seed}';common=['--dataset-root',a.dataset_root.resolve(),'--universe',u,'--seed',seed]
   run(['CoSTER.py','train-master',*common,'--output-dir',base/'master'])
   encoder=[]
   if a.checkpoint_root:encoder=['--encoder-checkpoint',a.checkpoint_root.resolve()/u/f'seed{seed}/continuous_prism'/f'no_vq_{u}_seed{seed}_base_best.pt']
   run(['CoSTER.py','train-continuous',*common,*encoder,'--output-dir',base/'continuous_prism'])
   run(['CoSTER.py','train-sparse',*common,'--selected',1,'--continuous-dir',base/'continuous_prism','--output-dir',base/'sparse_top1'])
   run(['-m','evaluation.predict',*common,'--run-dir',base,'--output-dir',base/'evaluation','--device','cuda'])
  if a.include_baselines:
   for name in ['ridge','random_forest']:run(['-m','experiments.baselines.traditional','--model',name,'--universe',u,'--seed',0,'--output-dir',a.output_root.resolve()/'baselines'/u/name])
   run(['-m','experiments.baselines.xgboost','--universe',u,'--seed',0,'--output-dir',a.output_root.resolve()/'baselines'/u/'xgboost'])
   run(['-m','experiments.baselines.act','--universe',u,'--seed',0,'--output-dir',a.output_root.resolve()/'baselines'/u/'act'])
   for name in ['lstm','gru','transformer','lightgbm']:run(['-m','experiments.baselines.run_additional_baselines','--model',name,'--universe',u,'--seed',0,'--device','cuda','--dataset-root',a.dataset_root.resolve(),'--output-dir',a.output_root.resolve()/'baselines'])
   for name in ['run_stockmamba_paper','run_prism_official_adapted']:run(['-m','experiments.baselines.'+name,'--universe',u,'--seed',0])
if __name__=='__main__':main()
