import json
from collections import Counter,defaultdict
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.nn.attention import sdpa_kernel,SDPBackend
from CoSTER import CoSTER

class Count(TorchDispatchMode):
 def __init__(self):super().__init__();self.counts={};self.ops={}
 def __torch_dispatch__(self,func,types,args=(),kwargs=None):
  y=func(*args,**(kwargs or {}));name=str(func);self.ops[name]=self.ops.get(name,0)+1;c=0
  if name in ('aten.mm.default','aten.bmm.default'):c=2*y.numel()*args[0].shape[-1]
  elif name=='aten.addmm.default':c=2*y.numel()*args[1].shape[-1]
  elif name=='aten.baddbmm.default':c=2*y.numel()*args[1].shape[-1]
  if c:self.counts[name]=self.counts.get(name,0)+int(c)
  return y

def main():
 torch.set_num_threads(1);torch.backends.mha.set_fastpath_enabled(False);model=CoSTER().eval();out=[]
 for n in [300,800]:
  row={'stocks':n}
  for name,m in [('MASTER',model.master),('Continuous',model.continuous)]:
   counter=Count()
   with torch.no_grad(),sdpa_kernel(SDPBackend.MATH),counter:m(torch.randn(n,8,221))
   row[name]={'parameters':sum(p.numel() for p in m.parameters()),'matrix_FLOPs':sum(counter.counts.values())}
  row['increment_pct']=100*row['Continuous']['matrix_FLOPs']/row['MASTER']['matrix_FLOPs'];out.append(row)
 print(json.dumps({'scope':'Matrix MAC=2 FLOPs; excludes elementwise operations, activation, normalization, sorting and score combination','counts':out},indent=2))
if __name__=='__main__':main()
