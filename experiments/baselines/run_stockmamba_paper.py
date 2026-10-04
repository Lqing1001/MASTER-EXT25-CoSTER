"""Seed-0 StockMamba reconstruction, fixed validation-only protocol, 20 epochs."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import argparse,json,hashlib,math,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from .stockmamba_paper import StockMambaPaper,rank_position_loss,ROOT
sys.path.insert(0,str(ROOT))
import CoSTER as protocol
from .run_additional_baselines import dump

def run(args):
    out=ROOT/'runs/stockmamba'/('smoke' if args.smoke else 'full')/f'{args.universe}_seed{args.seed}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():print('already complete',out,flush=True);return
    status_path=out/'status.json';started=time.time()
    try:
        protocol.seed_all(args.seed); device=torch.device(args.device)
        store=protocol.UniverseStore(protocol.DATASET_ROOT,args.universe)
        model=StockMambaPaper(args.universe).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
        dates=store.dates_between(protocol.TRAIN_START,protocol.TRAIN_END)
        if args.smoke: dates=dates[:3]
        rng=np.random.RandomState(args.seed);history=[];best=-float('inf');best_epoch=-1
        config=dict(arguments=vars(args),dataset_root=str(store.base),splits={'train':[protocol.TRAIN_START,protocol.TRAIN_END],'validation':[protocol.VALID_START,protocol.VALID_END],'test':[protocol.TEST_START,protocol.TEST_END]},optimizer='Adam lr=1e-5; weight_decay=0',clip='clip_grad_value 3',selection='max validation mean(IC,RankIC); all 20 epochs; no training-loss threshold',beta=model.beta,ffn_hidden=1024,ffn_note='article unspecified; conventional 4*d, inferred not author-confirmed',parameters=sum(p.numel() for p in model.parameters()),source='reconstructed from published equations; not author source',initialization='A_log=-1; dt_bias Uniform(0.001,0.1); D=1; literal paper specification',sha256=hashlib.sha256((Path(__file__).parent/'stockmamba_paper.py').read_bytes()).hexdigest())
        dump(out/'config.json',config)
        for epoch in range(1,(1 if args.smoke else args.epochs)+1):
            model.train();order=dates.copy();rng.shuffle(order);losses=[];epoch_start=time.time()
            for n,day in enumerate(order,1):
                x,y,_=store.batch(int(day),training=True)
                idx,target=protocol.drop_extreme_and_zscore(torch.tensor(y))
                pred=model(torch.from_numpy(x[idx.numpy()]).to(device));loss=rank_position_loss(pred,target.to(device))
                if not torch.isfinite(loss): raise FloatingPointError(f'nonfinite loss {day}')
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_value_(model.parameters(),3);optimizer.step();losses.append(loss.item())
                if n%250==0 or n==len(order):
                    dump(status_path,dict(state='training',epoch=epoch,day=n,total_days=len(order),elapsed_seconds=time.time()-started))
                    print(f'{args.universe} epoch={epoch} {n}/{len(order)} loss={np.mean(losses):.6f}',flush=True)
            rows,_=protocol.evaluate(model,store,protocol.VALID_START,protocol.VALID_END,device,3 if args.smoke else None)
            metrics=protocol.summarize_daily(rows);score=(metrics['IC']+metrics['RankIC'])/2
            if not math.isfinite(score):raise RuntimeError('invalid validation score')
            if score>best:
                best=score;best_epoch=epoch;torch.save(model.state_dict(),out/'best.pt')
            history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),validation=metrics,score=score,best_epoch=best_epoch,seconds=time.time()-epoch_start))
            dump(out/'history.json',history)
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,numpy_rng=rng.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),out/'last_training.pt')
            print(f'VALID {args.universe} epoch={epoch} IC={metrics["IC"]:.6f} RankIC={metrics["RankIC"]:.6f} best={best_epoch}',flush=True)
        model.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=True))
        rows,arrays=protocol.evaluate(model,store,protocol.TEST_START,protocol.TEST_END,device,3 if args.smoke else None)
        if not args.smoke:assert len(rows)==969 and sum(np.isfinite(r['IC']) for r in rows)==964
        protocol.write_daily_csv(out/'test_daily.csv',rows);np.savez_compressed(out/'test_predictions.npz',**arrays)
        result=dict(model='StockMamba paper reconstruction',universe=args.universe,seed=args.seed,smoke=args.smoke,config=config,best_epoch=best_epoch,metrics=protocol.metrics_by_period(rows),history=history,elapsed_seconds=time.time()-started)
        dump(out/'result.json',result);dump(status_path,dict(state='completed',best_epoch=best_epoch,elapsed_seconds=time.time()-started))
        print('COMPLETE',out,flush=True)
    except Exception as exc:
        dump(status_path,dict(state='failed',error=str(exc)));raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--universe',choices=['csi300','csi800'],default='csi300');p.add_argument('--suite',action='store_true')
    p.add_argument('--seed',type=int,default=0);p.add_argument('--device',default='cuda');p.add_argument('--epochs',type=int,default=20);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    for universe in (['csi300','csi800'] if args.suite else [args.universe]):args.universe=universe;run(args)
if __name__=='__main__':main()
