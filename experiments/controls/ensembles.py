from common import *
import argparse,subprocess
sys.path.insert(0,str(ROOT/'experiments/master_ext_reproduction'))
from run_additional_baselines import RecurrentRanker

def benchmark(m,x,repeats=100):
    m.eval()
    with torch.inference_mode():
        for _ in range(20):m(x)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();ms=[]
        for _ in range(repeats):
            t=time.perf_counter();m(x);torch.cuda.synchronize();ms.append((time.perf_counter()-t)*1000)
    return dict(median_ms=float(np.median(ms)),p95_ms=float(np.quantile(ms,.95)),peak_cuda_bytes=torch.cuda.max_memory_allocated(),registered_parameters=sum(p.numel() for p in m.parameters()))

class Pair(torch.nn.Module):
    def __init__(self,a,b):super().__init__();self.a=a;self.b=b
    def forward(self,x):return .5*(B.daily_zscore(self.a(x))+B.daily_zscore(self.b(x)))

def main(u,s):
    setup();d=OUT/'ensembles'/u/f'seed{s}';d.mkdir(parents=True,exist_ok=True)
    if (d/'result.json').exists():return
    mp=predict(u,s,'master');baseline=ROOT/'results/coster_additional_baselines/full'/f'gru_{u}_seed0'
    if s==0:
        gpath=baseline
        cfg=json.loads((gpath/'config.json').read_text());assert str(DATA)==str(Path(cfg['dataset_root']))
        assert cfg['smoke'] is False
    else:
        gpath=OUT/'simple_baselines/full'/f'gru_{u}_seed{s}'
        subprocess.run([sys.executable,str(ROOT/'experiments/master_ext_reproduction/run_additional_baselines.py'),'--model','gru','--universe',u,'--seed',str(s),'--device','cpu','--dataset-root',str(DATA),'--output-dir',str(OUT/'simple_baselines')],cwd=ROOT,check=True)
    gp=loadnp(gpath/'test_predictions.npz');aligned(mp,gp)
    ridge=ROOT/'results'/('protocol_strict_norm_20191224'+('_csi800' if u=='csi800' else ''))/'traditional_baselines'/f'ridge_{u}_seed0_predictions.npz'
    cfg=json.loads((ridge.parent/'_protocol.json').read_text());assert Path(cfg['dataset_root']).resolve()==DATA.resolve()
    rp=loadnp(ridge);aligned(mp,rp)
    metrics={}
    for name,pr in [('gru',gp),('ridge',rp)]:
        fp=fusion(mp,pr);np.savez_compressed(d/f'master_{name}.npz',**fp);metrics[name]=summary(daily(fp))
    store=B.UniverseStore(DATA,u);x,_,_=store.batch(20240102);x=torch.from_numpy(x).to(DEVICE)
    cost=[]
    for kind in ('master','base','dense','full'):
        m=model(u,s,kind)
        cost.append(dict(model=kind,**benchmark(m,x)))
        if kind!='master':cost.append(dict(model='MASTER+'+kind,**benchmark(Pair(model(u,s,'master'),m).to(DEVICE),x)))
        del m
    gr=RecurrentRanker('gru').to(DEVICE);gr.load_state_dict(torch.load(gpath/'best.pt',map_location=DEVICE,weights_only=True))
    cost.append(dict(model='MASTER+GRU',**benchmark(Pair(model(u,s,'master'),gr).to(DEVICE),x)))
    if s<9 and source(u,s+1)['master'].exists():cost.append(dict(model='MASTER+MASTER',**benchmark(Pair(model(u,s,'master'),model(u,s+1,'master')).to(DEVICE),x)))
    csvout(d/'inference_cost.csv',cost)
    dump(d/'result.json',dict(status='complete',universe=u,seed=s,metrics=metrics,gru_source=str(gpath),ridge_source=str(ridge),ridge_note='Deterministic fitted Ridge reused across MASTER seeds; not independent Ridge retraining.',inference_note='Same GPU, one full 20240102 cross-section, 20 warmups, 100 synchronized repetitions; excludes data preparation/transfer; registered parameters include unused pretraining heads.',n_stocks=len(x),gru_training_device='CPU, one thread',source_hashes={str(gpath/'test_predictions.npz'):sha(gpath/'test_predictions.npz'),str(ridge):sha(ridge)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--universe',required=True);p.add_argument('--seed',required=True,type=int);a=p.parse_args();main(a.universe,a.seed)
