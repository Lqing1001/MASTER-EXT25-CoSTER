from common import *
import argparse, copy, random
from torch import nn

class RoutedResidual(B.DeformableTemporalResidual):
    def forward(self,stock):
        if self.mode in ('top1','top2','softmax'):
            return super().forward(stock)
        seq=self.input_proj(stock)+self.position[:,:stock.shape[1]]
        ix=torch.full((len(stock),),stock.shape[1]-1,device=stock.device,dtype=torch.long) if self.mode=='last' else self.route
        chosen=self.value(seq)[torch.arange(len(stock),device=stock.device),ix]
        return self.head(chosen).squeeze(-1)

def route(m,day,instruments,seed):
    if not hasattr(m,'residual') or getattr(m.residual,'mode',None)!='fixed_random':return
    ids=np.array([int(bytes(x).decode()[-6:]) for x in instruments],dtype=np.uint64)
    h=(ids*np.uint64(2654435761)+np.uint64(int(day))*np.uint64(2246822519)+np.uint64(seed)*np.uint64(3266489917))&np.uint64(0xffffffff)
    h=((h^(h>>16))*np.uint64(2246822519))&np.uint64(0xffffffff)
    h=((h^(h>>13))*np.uint64(3266489917))&np.uint64(0xffffffff)
    h=h^(h>>16)
    m.residual.route=torch.from_numpy((h%8).astype(np.int64)).to(DEVICE)

class RoutedStore:
    def __init__(self,store,m,seed):self.store=store;self.m=m;self.seed=seed;self.universe=store.universe
    def dates_between(self,*args):return self.store.dates_between(*args)
    def batch(self,day,training=False):
        x,y,ids=self.store.batch(day,training);route(self.m,day,ids,self.seed);return x,y,ids

def sparse(anchor,seed,mode):
    B.seed_all(seed);m=B.SparseExpert(anchor,'deformable',selected={'top2':2,'softmax':8}.get(mode,1)).to(DEVICE)
    m.residual.__class__=RoutedResidual;m.residual.mode=mode
    return m

def fit(m,store,d,epochs,seed,smoke=False):
    d.mkdir(parents=True,exist_ok=True)
    if (d/'result.json').exists():
        m.load_state_dict(torch.load(d/'best.pt',map_location=DEVICE,weights_only=True));return
    ps=[p for p in m.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(ps,lr=1e-4,weight_decay=1e-4)
    frozen={n:p.detach().clone() for n,p in m.named_parameters() if not p.requires_grad}
    rng=np.random.RandomState(seed);dates=store.dates_between(B.TRAIN_START,B.TRAIN_END)
    hist=[];best=-np.inf;start_epoch=0;best_epoch=-1
    resume=d/'resume.pt'
    if resume.exists():
        r=torch.load(resume,map_location=DEVICE,weights_only=False)
        assert r['source_hash']==sha(Path(__file__)), 'Training source changed; explicit migration required'
        m.load_state_dict(r['model']);opt.load_state_dict(r['optimizer']);rng.set_state(r['numpy_rng'])
        torch.set_rng_state(r['cpu_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in r['cuda_rng']])
        hist=r['history'];best=r['best'];best_epoch=r['best_epoch'];start_epoch=r['next_epoch']
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    for epoch in range(start_epoch,1 if smoke else epochs):
        m.train();order=dates.copy();rng.shuffle(order)
        if smoke:order=order[:3]
        losses=[];tick=time.perf_counter()
        for i,day in enumerate(order):
            x,y,ids=store.batch(int(day),training=True)
            x=torch.from_numpy(x).to(DEVICE);y=torch.from_numpy(y).to(DEVICE)
            keep,target=B.drop_extreme_and_zscore(y);route(m,int(day),ids[keep.cpu().numpy()],seed)
            p=m(x[keep]);loss=nn.functional.mse_loss(p,target)+.1*B.pearson_loss(p,target)
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(ps,1.,error_if_nonfinite=True);opt.step();losses.append(loss.item())
            if (i+1)%500==0:print(d.name,epoch,i+1,len(order),'loss',np.mean(losses),flush=True)
        vr,vp=B.evaluate(m,RoutedStore(store,m,seed),B.VALID_START,B.VALID_END,DEVICE,3 if smoke else None)
        val=B.summarize_daily(vr);v=.5*(val['IC']+val['RankIC']);assert np.isfinite(v)
        improved=v>best
        if improved:best=v;best_epoch=epoch;torch.save(m.state_dict(),d/'best.pt')
        hist.append(dict(epoch=epoch,loss=float(np.mean(losses)),validation=val,selected=improved,seconds=time.perf_counter()-tick))
        dump(d/'history.json',hist)
        torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),numpy_rng=rng.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),history=hist,best=best,best_epoch=best_epoch,next_epoch=epoch+1,source_hash=sha(Path(__file__))),d/'resume.tmp')
        (d/'resume.tmp').replace(resume)
        print('EPOCH',d,epoch,'validation',v,'seconds',hist[-1]['seconds'],flush=True)
    for n,p in m.named_parameters():
        if n in frozen:assert torch.equal(p,frozen[n]),'Frozen anchor changed: '+n
    m.load_state_dict(torch.load(d/'best.pt',map_location=DEVICE,weights_only=True))
    for split,a,b in [('validation',B.VALID_START,B.VALID_END),('test',B.TEST_START,B.TEST_END)]:
        rows,pr=B.evaluate(m,RoutedStore(store,m,seed),a,b,DEVICE,3 if smoke else None)
        B.write_daily_csv(d/f'{split}_daily.csv',rows);np.savez_compressed(d/f'{split}.npz',**pr)
    dump(d/'result.json',dict(status='complete',seed=seed,smoke=smoke,best_epoch=best_epoch,validation_score=best,selection='branch_validation',epochs=len(hist),history=hist,seconds_this_invocation=time.perf_counter()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated(),registered_parameters=sum(p.numel() for p in m.parameters()),trainable_parameters=sum(p.numel() for p in ps),frozen_parameters_unchanged=True,source_hash=sha(Path(__file__))))

def new_main(u,s):
    assert s>=3
    import subprocess
    d=OUT/'training'/u/f'seed{s}';paths=source(u,s)
    if not paths['master'].exists():
        subprocess.run([sys.executable,str(ROOT/'CoSTER.py'),'train-master','--dataset-root',str(DATA),'--universe',u,'--seed',str(s),'--output-dir',str(paths['master'].parent)],cwd=ROOT,check=True)
    store=B.UniverseStore(DATA,u);spfile=d/'encoder.pt'
    if not spfile.exists():
        B.seed_all(s);sp=B.ContinuousEncoder('no_vq').to(DEVICE)
        opt=torch.optim.AdamW(sp.parameters(),lr=1e-4,weight_decay=1e-4)
        h=B.train_spatial(sp,'no_vq',store,store.dates_between(B.TRAIN_START,B.TRAIN_END),opt,DEVICE,4,None)
        torch.save(sp.state_dict(),spfile);dump(d/'encoder_history.json',h);del sp
    # Match the final submission's position-ranker initialization after encoder loading.
    B.seed_all(s);sp=B.ContinuousEncoder('no_vq').to(DEVICE)
    sp.load_state_dict(torch.load(spfile,map_location=DEVICE,weights_only=True))
    for p in sp.parameters():p.requires_grad_(False)
    b=B.MixtureRanker(sp,'no_vq').to(DEVICE);fit(b,store,d/'base',12,s)
    a=B.TemporalAdapter(b).to(DEVICE);fit(a,store,d/'dense',10,s)
    m=sparse(a,s,'top1');fit(m,store,d/'sparse',10,s)

def ablation(u,s,smoke=False):
    store=B.UniverseStore(DATA,u)
    for kind in ['master','base','dense','full']:
        for split in ['validation','test']:predict(u,s,kind,split,store)
    d=OUT/('smoke' if smoke else 'ablation')/u/f'seed{s}/base_sparse'
    anchor=model(u,s,'base');m=sparse(anchor,s,'top1')
    dump(d/'config.json',dict(universe=u,seed=s,anchor=str(source(u,s)['base']),anchor_sha256=sha(source(u,s)['base']),mode='top1',epochs=10))
    fit(m,store,d,10,s,smoke)

def selector(u,s,mode,smoke=False):
    store=B.UniverseStore(DATA,u);d=OUT/('smoke' if smoke else 'selectors')/u/f'seed{s}'/mode
    m=sparse(model(u,s,'dense'),s,mode)
    dump(d/'config.json',dict(universe=u,seed=s,anchor=str(source(u,s)['dense']),anchor_sha256=sha(source(u,s)['dense']),mode=mode,epochs=10))
    fit(m,store,d,10,s,smoke)

def diagnostic():
    store=B.UniverseStore(DATA,'csi300');day=20190102;x,y,ids=store.batch(day,True)
    keep,target=B.drop_extreme_and_zscore(torch.from_numpy(y).to(DEVICE));x=torch.from_numpy(x).to(DEVICE)[keep];target=target.detach()
    rows=[];initials={};curves=[]
    for mode in ['top1','last','fixed_random','top2','softmax']:
        m=sparse(model('csi300',0,'dense'),0,mode);m.eval();route(m,day,ids[keep.cpu().numpy()],0)
        with torch.no_grad():assert torch.equal(m(x),m.base(x)), 'Zero-head initial prediction must equal anchor'
        initial={n:p.detach().clone() for n,p in m.residual.named_parameters()};initials[mode]=initial
        opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
        initial_positions=None
        for step in range(12):
            m.train();p=m(x);loss=nn.functional.mse_loss(p,target)+.1*B.pearson_loss(p,target)
            opt.zero_grad(set_to_none=True);loss.backward()
            for n,p in m.residual.named_parameters():
                rows.append(dict(mode=mode,step=step,parameter=n,grad_none=p.grad is None,grad_norm=None if p.grad is None else float(p.grad.norm()),delta_before_step=float((p-initial[n]).abs().max())))
            nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.);opt.step()
            with torch.no_grad():
                seq=m.residual.input_proj(x[:,:,:158])+m.residual.position
                pos=m.residual.selector(seq).squeeze(-1).argmax(1)
                if initial_positions is None:initial_positions=pos.clone()
                curves.append(dict(mode=mode,step=step,loss=float(loss),argmax_switch_fraction=float((pos!=initial_positions).float().mean()),selector_max_delta=max(float((p-initial[n]).abs().max()) for n,p in m.residual.named_parameters() if n.startswith('selector.'))))
        if mode=='top1':assert all(r['grad_norm']==0 for r in rows if r['mode']==mode and r['parameter'].startswith('selector.'))
        if mode in ('top2','softmax'):assert any(r['grad_norm'] and r['grad_norm']>0 for r in rows if r['mode']==mode and r['parameter'].startswith('selector.'))
    for mode,init in initials.items():
        for n in init:assert torch.equal(init[n],initials['top1'][n]), 'Unmatched residual initialization'
    csvout(OUT/'diagnostics/gradients.csv',rows);csvout(OUT/'diagnostics/optimization.csv',curves)
    dump(OUT/'diagnostics/status.json',dict(status='passed',data='training date 20190102 only; diagnostic, not predictive evaluation',steps=12,initial_anchor_equality=True,matched_initializations=True,top1_scorer_task_gradient_zero=True,soft_and_top2_nonzero_after_zero_head=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['ablation','selector','new_main','diagnostic']);p.add_argument('--universe',default='csi300');p.add_argument('--seed',type=int,default=0);p.add_argument('--mode',default='top2');p.add_argument('--smoke',action='store_true');a=p.parse_args();setup()
    if a.task=='diagnostic':diagnostic()
    elif a.task=='new_main':new_main(a.universe,a.seed)
    elif a.task=='ablation':ablation(a.universe,a.seed,a.smoke)
    else:selector(a.universe,a.seed,a.mode,a.smoke)
