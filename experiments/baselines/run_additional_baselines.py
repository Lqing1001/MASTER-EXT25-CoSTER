"""Additional MASTER-EXT25 baselines; fixed protocol, validation-only selection.
Run from project root. Full comparison: --suite. --smoke never yields paper results.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse, copy, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import CoSTER as protocol

class RecurrentRanker(nn.Module):
    def __init__(self, kind, hidden=64):
        super().__init__()
        self.encoder=getattr(nn,kind.upper())(221,hidden,num_layers=2,dropout=0.1,batch_first=True)
        self.head=nn.Linear(hidden,1)
    def forward(self,x):
        return self.head(self.encoder(x)[0][:,-1]).squeeze(-1)

class TransformerRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection=nn.Linear(221,64)
        pos=torch.arange(8).float()[:,None]
        freq=torch.exp(torch.arange(0,64,2).float()*(-math.log(10000)/64))
        pe=torch.zeros(8,64); pe[:,0::2]=torch.sin(pos*freq); pe[:,1::2]=torch.cos(pos*freq)
        self.register_buffer('position',pe)
        layer=nn.TransformerEncoderLayer(64,4,128,0.1,batch_first=True)
        self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.head=nn.Linear(64,1)
    def forward(self,x):
        return self.head(self.encoder(self.projection(x)+self.position)[:,-1]).squeeze(-1)

def build_model(name):
    if name in ('lstm','gru'): return RecurrentRanker(name)
    if name=='transformer': return TransformerRanker()
    raise ValueError(name)

def tabular(x):
    return np.concatenate((x[:,-1,:158],x[:,:,:158].mean(1),x[:,-1,158:]),axis=1).astype(np.float32)

def dump(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    tmp.replace(path)

def neural(args,store,out):
    device=torch.device(args.device)
    model=build_model(args.model).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    rng=np.random.RandomState(args.seed)
    days=store.dates_between(protocol.TRAIN_START,protocol.TRAIN_END)
    if args.smoke: days=days[:3]
    best=-float('inf'); stale=0; history=[]; best_epoch=-1
    for epoch in range(1,(1 if args.smoke else args.epochs)+1):
        start=time.time(); model.train(); order=days.copy(); rng.shuffle(order); losses=[]
        for day in order:
            x,y,_=store.batch(int(day),training=True)
            idx,target=protocol.drop_extreme_and_zscore(torch.from_numpy(y))
            prediction=model(torch.from_numpy(x[idx.numpy()]).to(device))
            loss=nn.functional.mse_loss(prediction,target.to(device))
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); losses.append(loss.item())
        rows,_=protocol.evaluate(model,store,protocol.VALID_START,protocol.VALID_END,device,3 if args.smoke else None)
        metrics=protocol.summarize_daily(rows); score=(metrics['IC']+metrics['RankIC'])/2
        if not math.isfinite(score): raise RuntimeError('Invalid validation score')
        improved=score>best
        if improved:
            best=score; best_epoch=epoch; stale=0
            torch.save(model.state_dict(),out/'best.pt')
        else: stale+=1
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),validation=metrics,score=score,best=improved,seconds=time.time()-start))
        dump(out/'history.json',history)
        print(f'{args.model} {args.universe} epoch={epoch} score={score:.6f} best={best_epoch} seconds={time.time()-start:.1f}',flush=True)
        if stale>=args.patience: break
    model.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=True))
    rows,arrays=protocol.evaluate(model,store,protocol.TEST_START,protocol.TEST_END,device,3 if args.smoke else None)
    return rows,arrays,dict(history=history,best_epoch=best_epoch,parameters=sum(p.numel() for p in model.parameters()),selection='maximum mean of validation daily IC and RankIC')

def materialize(store,start,end,smoke=False):
    xs=[]; ys=[]; dates=store.dates_between(start,end)
    if smoke: dates=dates[:3]
    for i,day in enumerate(dates):
        x,y,_=store.batch(int(day),training=True)
        idx,target=protocol.drop_extreme_and_zscore(torch.from_numpy(y))
        xs.append(tabular(x[idx.numpy()])); ys.append(target.numpy())
        if (i+1)%500==0: print(f'materialize {i+1}/{len(dates)}',flush=True)
    return np.concatenate(xs),np.concatenate(ys)

def evaluate_tree(model,store,start,end,smoke=False):
    rows=[]; chunks={k:[] for k in ('dates','instruments','predictions','labels')}
    dates=store.dates_between(start,end)
    if smoke: dates=dates[:3]
    for day in dates:
        x,y,inst=store.batch(int(day),training=False)
        pred=model.predict(tabular(x),num_threads=1).astype(np.float32)
        if not np.isfinite(pred).all(): raise RuntimeError('nonfinite predictions')
        rows.append(dict(date=int(day),n=len(y),finite_labels=int(np.isfinite(y).sum()),IC=protocol.correlation(pred,y),RankIC=protocol.correlation(pred,y,rank=True)))
        for key,values in zip(chunks,(np.full(len(y),day,dtype=np.int32),inst.astype('S8'),pred,y)):
            chunks[key].append(values)
    return rows,{k:np.concatenate(v) for k,v in chunks.items()}

def lightgbm(args,store,out):
    import lightgbm as lgb
    tx,ty=materialize(store,protocol.TRAIN_START,protocol.TRAIN_END,args.smoke)
    vx,vy=materialize(store,protocol.VALID_START,protocol.VALID_END,args.smoke)
    train=lgb.Dataset(tx,label=ty,free_raw_data=True)
    valid=lgb.Dataset(vx,label=vy,reference=train,free_raw_data=True)
    train.construct(); valid.construct(); del tx,ty,vx,vy
    candidates=[]; best_score=-float('inf')
    for leaves in ([31] if args.smoke else [31,63]):
        config=dict(objective='regression',metric='rmse',learning_rate=0.03,num_leaves=leaves,min_data_in_leaf=100,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l1=0.1,lambda_l2=2.0,num_threads=1,seed=args.seed,deterministic=True,force_col_wise=True,verbosity=-1)
        model=lgb.train(config,train,num_boost_round=3 if args.smoke else 2000,valid_sets=[valid],callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(100)])
        rows,_=evaluate_tree(model,store,protocol.VALID_START,protocol.VALID_END,args.smoke)
        metrics=protocol.summarize_daily(rows); score=(metrics['IC']+metrics['RankIC'])/2
        candidates.append(dict(config=config,rounds=model.best_iteration,validation=metrics,score=score))
        print(f'LightGBM {args.universe} leaves={leaves} rounds={model.best_iteration} validation={score:.6f}',flush=True)
        if score>best_score:
            best_score=score; best=model; model.save_model(str(out/'best.txt'))
    rows,arrays=evaluate_tree(best,store,protocol.TEST_START,protocol.TEST_END,args.smoke)
    return rows,arrays,dict(candidates=candidates,selection='validation RMSE early stopping; leaf count chosen by mean validation IC and RankIC',features='last Alpha158 + 8-day mean Alpha158 + last Market63 (379)',lightgbm_version=lgb.__version__)

def run(args):
    out=args.output_dir/('smoke' if args.smoke else 'full')/f'{args.model}_{args.universe}_seed{args.seed}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():
        print('Already completed:',out,flush=True); return
    protocol.seed_all(args.seed)
    config=vars(args).copy(); config['dataset_root']=str(args.dataset_root.resolve())
    config['splits']={'train':[protocol.TRAIN_START,protocol.TRAIN_END],'validation':[protocol.VALID_START,protocol.VALID_END],'test':[protocol.TEST_START,protocol.TEST_END]}
    config['target_processing']='training: daily 2.5% tails removed, sample-standard-deviation z-score; evaluation: full universe before finite-label scoring'
    config['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dump(out/'config.json',config)
    started=time.time(); store=protocol.UniverseStore(args.dataset_root,args.universe)
    rows,arrays,info=(lightgbm if args.model=='lightgbm' else neural)(args,store,out)
    if not args.smoke:
        assert len(rows)==969 and sum(np.isfinite(r['IC']) for r in rows)==964
    protocol.write_daily_csv(out/'test_daily.csv',rows)
    np.savez_compressed(out/'test_predictions.npz',**arrays)
    dump(out/'result.json',dict(model=args.model,universe=args.universe,seed=args.seed,smoke=args.smoke,metrics=protocol.metrics_by_period(rows),training=info,config=config,elapsed_seconds=time.time()-started))
    print('COMPLETE',out,flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['lstm','gru','transformer','lightgbm'],default='lstm')
    p.add_argument('--universe',choices=['csi300','csi800'],default='csi300')
    p.add_argument('--dataset-root',type=Path,default=protocol.DATASET_ROOT)
    p.add_argument('--output-dir',type=Path,default=ROOT/'runs/classic')
    p.add_argument('--seed',type=int,default=0); p.add_argument('--device',default='cpu')
    p.add_argument('--epochs',type=int,default=40); p.add_argument('--patience',type=int,default=6)
    p.add_argument('--lr',type=float,default=0.001); p.add_argument('--smoke',action='store_true'); p.add_argument('--suite',action='store_true')
    args=p.parse_args(); torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if args.suite:
        for universe in ('csi300','csi800'):
            for model in ('lstm','gru','transformer','lightgbm'):
                args.universe=universe; args.model=model; run(args)
    else: run(args)
if __name__=='__main__': main()
