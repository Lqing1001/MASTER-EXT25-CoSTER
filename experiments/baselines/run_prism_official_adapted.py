"""Train the PRISM-VQ adaptation on MASTER-EXT25."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .prism_official_adapter import build_spatial,PrismOfficialAdapter,CONFIG,ROOT
sys.path.insert(0,str(ROOT))
import CoSTER as protocol
from .run_additional_baselines import dump

def main():
    p=argparse.ArgumentParser(); p.add_argument('--universe',choices=['csi300','csi800'],default='csi300')
    p.add_argument('--seed',type=int,default=0); p.add_argument('--device',default='cuda')
    p.add_argument('--stage1-epochs',type=int,default=4); p.add_argument('--epochs',type=int,default=12); p.add_argument('--patience',type=int,default=4)
    p.add_argument('--smoke',action='store_true'); args=p.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1); protocol.seed_all(args.seed)
    out=ROOT/'runs/prism_vq'/('smoke' if args.smoke else 'full')/f'{args.universe}_seed{args.seed}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists(): print('already complete',flush=True); return
    dump(out/'config.json',dict(arguments=vars(args),architecture=CONFIG,source='https://github.com/finxlab/PRISM-VQ',commit='7d02635d0cdeec2f4e7278a9e73acd0eb67a8fa6',adaptations=['8-day window','Market63 replaces JKP13','single five-day auxiliary target','common normalized training labels and validation IC selection','frozen VQ kept in eval mode; RMSNorm fused fastpath disabled']))
    store=protocol.UniverseStore(protocol.DATASET_ROOT,args.universe)
    days=store.dates_between(protocol.TRAIN_START,protocol.TRAIN_END)
    if args.smoke: days=days[:3]
    rng=np.random.RandomState(args.seed); device=torch.device(args.device)
    spatial=build_spatial().to(device); optimizer=torch.optim.AdamW(spatial.parameters(),lr=1e-4,weight_decay=1e-5)
    started=time.time(); prehistory=[]
    for epoch in range(1,(1 if args.smoke else args.stage1_epochs)+1):
        spatial.train(); order=days.copy(); rng.shuffle(order); losses=[]
        for day in order:
            x,y,_=store.batch(int(day),training=True)
            idx,target=protocol.drop_extreme_and_zscore(torch.tensor(y))
            x=torch.from_numpy(x[idx.numpy()]).to(device); target=target.to(device).unsqueeze(-1)
            recon,vq,pred,loss,*_=spatial(x[:,:,:158],x[:,-1,158:],target)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite pretrain loss')
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(spatial.parameters(),1); optimizer.step()
            losses.append([recon.item(),vq.item(),pred.item(),loss.item()])
        entry=dict(epoch=epoch,mean_losses=np.mean(losses,axis=0).tolist()); prehistory.append(entry)
        dump(out/'pretrain_history.json',prehistory); torch.save(spatial.state_dict(),out/'spatial.pt')
        print(f'PRETRAIN {args.universe} {entry}',flush=True)
    model=PrismOfficialAdapter(spatial).to(device)
    trainable=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=1e-4,weight_decay=1e-5)
    history=[]; best=-float('inf'); stale=0; best_epoch=-1
    for epoch in range(1,(1 if args.smoke else args.epochs)+1):
        model.train(); order=days.copy(); rng.shuffle(order); losses=[]; epoch_start=time.time()
        for day in order:
            x,y,_=store.batch(int(day),training=True)
            idx,target=protocol.drop_extreme_and_zscore(torch.tensor(y))
            prediction=model(torch.from_numpy(x[idx.numpy()]).to(device))
            loss=nn.functional.mse_loss(prediction,target.to(device))+.01*model.aux_loss
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite prediction loss')
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(trainable,1); optimizer.step(); losses.append(loss.item())
        rows,_=protocol.evaluate(model,store,protocol.VALID_START,protocol.VALID_END,device,3 if args.smoke else None)
        metrics=protocol.summarize_daily(rows); score=(metrics['IC']+metrics['RankIC'])/2
        if not math.isfinite(score): raise RuntimeError('Invalid validation score')
        if score>best:
            best=score; best_epoch=epoch; stale=0; torch.save(model.state_dict(),out/'best.pt')
        else: stale+=1
        entry=dict(epoch=epoch,train_loss=float(np.mean(losses)),validation=metrics,best_epoch=best_epoch,seconds=time.time()-epoch_start)
        history.append(entry); dump(out/'history.json',history); print(f'PREDICTOR {args.universe} {entry}',flush=True)
        if stale>=args.patience: break
    model.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=True))
    rows,arrays=protocol.evaluate(model,store,protocol.TEST_START,protocol.TEST_END,device,3 if args.smoke else None)
    if not args.smoke: assert len(rows)==969 and sum(np.isfinite(r['IC']) for r in rows)==964
    protocol.write_daily_csv(out/'test_daily.csv',rows); np.savez_compressed(out/'test_predictions.npz',**arrays)
    dump(out/'result.json',dict(model='PRISM-VQ official-module adaptation',universe=args.universe,seed=args.seed,smoke=args.smoke,metrics=protocol.metrics_by_period(rows),pretrain=prehistory,history=history,best_epoch=best_epoch,elapsed_seconds=time.time()-started))
    print('COMPLETE',out,flush=True)
if __name__=='__main__': main()
