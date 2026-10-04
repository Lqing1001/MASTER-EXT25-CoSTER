"""CSI300 membership sensitivity using fixed final predictions."""
import argparse,json
from pathlib import Path
import numpy as np
import CoSTER as C

def main():
 p=argparse.ArgumentParser();p.add_argument('--raw-features',type=Path,required=True);p.add_argument('--predictions',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();excluded=set()
 for f in a.raw_features.glob('*.npz'):
  with np.load(f) as z:excluded.update((int(d),f.stem) for d in z['dates'][z['is_csi300']&~z['is_csi800']])
 out={'all_mismatches':len(excluded),'models':{}}
 for f in a.predictions:
  with np.load(f) as z:v={k:z[k] for k in z.files}
  keep=np.array([(int(d),i.decode() if isinstance(i,bytes) else str(i)) not in excluded for d,i in zip(v['dates'],v['instruments'])]);bounds=np.r_[0,np.flatnonzero(np.diff(v['dates']))+1,len(keep)];m={}
  for name,mask in [('before',np.ones_like(keep)),('after',keep)]:
   rows=[]
   for l,r in zip(bounds[:-1],bounds[1:]):
    ix=np.arange(l,r)[mask[l:r]];x=v['predictions'][ix];y=v['labels'][ix];rows.append({'IC':C.correlation(x,y),'RankIC':C.correlation(x,y,True)})
   m[name]=C.summarize_daily(rows)
  out['models'][f.stem]=m;out['excluded_test']=int((~keep).sum());out['excluded_finite_test']=int((~keep&np.isfinite(v['labels'])).sum())
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2),encoding='utf8')
if __name__=='__main__':main()
