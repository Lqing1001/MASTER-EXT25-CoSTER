"""Controlled experiments for CoSTER."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import sys, json, hashlib, time, csv
import numpy as np
import torch
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import CoSTER as B
OUT=Path(os.environ.get('COSTER_EXPERIMENT_ROOT',str(ROOT/'runs/experiments')))
OLD=Path(os.environ.get('COSTER_LEGACY_ROOT',str(ROOT/'results/coster_nextgen_20260930')))
DATA=Path(os.environ.get('MASTER_EXT_DATASET_ROOT',str(ROOT/'datasets/master_ext_strict_20191224_v1')))
DEVICE=torch.device(os.environ.get('COSTER_DEVICE','cuda:0'))
SEEDS=list(range(10))
UNIVERSES=['csi300','csi800']
METRICS=['IC','RankIC','ICIR','RankICIR']

def setup():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8');temp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def csvout(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def loadnp(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def source(u,s):
    checkpoint_root=os.environ.get('COSTER_CHECKPOINT_ROOT')
    if checkpoint_root:
        folder=Path(checkpoint_root)/u/f'seed{s}'
        return {kind:folder/(kind+'.pt') for kind in ('master','base','dense','full')}

    if s<3:
        d=ROOT/'results'/('csi800_seed12_matched_seed0_20260915' if u=='csi800' and s in (1,2) else 'protocol_strict_norm_20191224'+('_csi800' if u=='csi800' else ''))
        p=OLD/'full'/u/f'seed{s}/position'
        return dict(master=d/'master'/f'master_full_{u}_seed{s}_best.pt',base=p/'ranker_best.pt',dense=p/'adapter_best.pt',full=p/'final.pt')
    p=OUT/'training'/u/f'seed{s}'
    return dict(master=p/'master'/f'master_full_{u}_seed{s}_best.pt',base=p/'base/best.pt',dense=p/'dense/best.pt',full=p/'sparse/best.pt')

def build(u,kind):
    if kind=='master':return B.MasterExpert('full',u)
    b=B.MixtureRanker(B.ContinuousEncoder('no_vq'),'no_vq')
    if kind=='base':return b
    d=B.TemporalAdapter(b)
    if kind=='dense':return d
    return B.SparseExpert(d,'deformable',selected=1)

def model(u,s,kind):
    m=build(u,kind).to(DEVICE)
    m.load_state_dict(torch.load(source(u,s)[kind],map_location=DEVICE,weights_only=True),strict=True)
    return m.eval()

def aligned(a,b):
    for k in ('dates','instruments'):assert np.array_equal(a[k],b[k]),k
    assert np.allclose(a['labels'],b['labels'],equal_nan=True)

def daily(a,mask=None):
    ds=a['dates'];assert np.all(ds[1:]>=ds[:-1]);cuts=np.r_[0,np.flatnonzero(np.diff(ds))+1,len(ds)];rows=[]
    for l,r in zip(cuts[:-1],cuts[1:]):
        ix=np.arange(l,r)
        if mask is not None:ix=ix[mask[ix]]
        y=a['labels'][ix].astype(float);p=a['predictions'][ix].astype(float)
        assert np.isfinite(p).all()
        valid=np.isfinite(y);y=y[valid];p=p[valid]
        if len(y)<3 or np.std(y)==0 or np.std(p)==0:continue
        rows.append(dict(date=int(ds[l]),IC=float(np.corrcoef(p,y)[0,1]),RankIC=float(np.corrcoef(rankdata(p),rankdata(y))[0,1])))
    return rows

def summary(rows):
    x=np.array([[r['IC'],r['RankIC']] for r in rows]);assert np.isfinite(x).all()
    return dict(zip(METRICS,np.r_[x.mean(0),x.mean(0)/x.std(0)]))

def fusion(a,b):
    aligned(a,b);out=dict(a);p=np.empty(len(a['dates']),dtype=float)
    cuts=np.r_[0,np.flatnonzero(np.diff(a['dates']))+1,len(p)]
    for l,r in zip(cuts[:-1],cuts[1:]):
        x=a['predictions'][l:r].astype(float);y=b['predictions'][l:r].astype(float)
        p[l:r]=.5*((x-x.mean())/max(x.std(),1e-12)+(y-y.mean())/max(y.std(),1e-12))
    out['predictions']=p;return out

def predpath(u,s,kind,split='test'):
    return OUT/'predictions'/u/f'seed{s}'/f'{kind}_{split}.npz'

def predict(u,s,kind,split='test',store=None):
    path=predpath(u,s,kind,split)
    if path.exists():return loadnp(path)
    if s<3 and split=='test' and kind in ('master','full'):
        src=(OLD/'test_evaluation'/u/f'seed{s}/master_predictions.npz') if kind=='master' else (OLD/'final_paper_evidence'/u/f'seed{s}/continuous_test.npz')
        a=loadnp(src)
    elif s<3 and split=='validation' and kind in ('master','full'):
        src=OLD/'full'/u/f'seed{s}'/('master_validation.npz' if kind=='master' else 'position/validation_predictions.npz')
        a=loadnp(src);a={k:a[k] for k in ('dates','instruments','labels','predictions')}
    else:
        store=store or B.UniverseStore(DATA,u);m=model(u,s,kind)
        start,end=(B.TEST_START,B.TEST_END) if split=='test' else (B.VALID_START,B.VALID_END)
        _,a=B.evaluate(m,store,start,end,DEVICE);del m
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**a)
    return a
