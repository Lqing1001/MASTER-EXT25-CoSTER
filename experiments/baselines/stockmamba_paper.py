"""StockMamba reconstructed from DOI 10.3390/math14111859, Eqs. 2--25.
Not author-provided StockMamba source. Core SSD comes from state-spaces/mamba.
No mamba_ssm/Triton extension dependency. See accompanying reproduction notes.
"""
from __future__ import annotations
import math,sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from CoSTER import PositionalEncoding,TAttention,SAttention,TemporalAttention
from .stockmamba_ssd_reference import ssd_minimal_discrete

class Mamba2MarketBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj=nn.Linear(64,386,bias=False) # z128 + x128 + B64 + C64 + dt2
        self.conv=nn.Conv1d(256,256,4,padding=3,groups=256,bias=True)
        self.dt_bias=nn.Parameter(torch.empty(2).uniform_(.001,.1))
        self.A_log=nn.Parameter(torch.full((2,),-1.0))
        self.D=nn.Parameter(torch.ones(2))
        self.norm_weight=nn.Parameter(torch.ones(128))
        self.out_proj=nn.Linear(128,64,bias=False)
    def forward(self,u):
        batch,length,_=u.shape
        z,xbc,dt=self.in_proj(u).split([128,256,2],dim=-1)
        dt=F.softplus(dt+self.dt_bias)
        xbc=F.silu(self.conv(xbc.transpose(1,2))[:,:,:length].transpose(1,2))
        x,b,c=xbc.split([128,64,64],dim=-1)
        x=x.reshape(batch,length,2,64)
        b=b.unsqueeze(2).expand(-1,-1,2,-1)
        c=c.unsqueeze(2).expand(-1,-1,2,-1)
        # For T=8, removing the 56 trailing zeros from the first chunk is exact:
        # a causal scan cannot send trailing padding information into earlier outputs.
        chunk=min(64,length)
        padded=math.ceil(length/chunk)*chunk
        xx=x*dt.unsqueeze(-1); aa=-self.A_log.exp()*dt
        if padded>length:
            xx=F.pad(xx,(0,0,0,0,0,padded-length));aa=F.pad(aa,(0,0,0,padded-length))
            b=F.pad(b,(0,0,0,0,0,padded-length));c=F.pad(c,(0,0,0,0,0,padded-length))
        y,_=ssd_minimal_discrete(xx,aa,b,c,chunk)
        y=y[:,:length]+x*self.D[None,None,:,None]
        y=y.reshape(batch,length,128)*F.silu(z)
        y=y*torch.rsqrt(y.square().mean(-1,keepdim=True)+1e-5)*self.norm_weight
        return self.out_proj(y)

class StockMambaPaper(nn.Module):
    def __init__(self,universe='csi300',ffn_hidden=1024):
        super().__init__()
        self.beta=5.0 if universe=='csi300' else 2.0
        self.market_in=nn.Linear(63,64,bias=False)
        self.mamba=Mamba2MarketBlock()
        self.market_out=nn.Linear(64,63,bias=False)
        self.gate=nn.Linear(63,158,bias=False)
        self.projection=nn.Linear(158,256)
        self.position=PositionalEncoding(256)
        self.temporal=TAttention(256,4,.5)
        self.spatial=SAttention(256,2,.5)
        # FFN width is not specified in the article. Use conventional 4*d;
        # total ~1.60M is close to its reported ~1.63M, unlike old ~0.67M proxy.
        for block in (self.temporal,self.spatial):
            block.ffn=nn.Sequential(nn.Linear(256,ffn_hidden),nn.ReLU(),nn.Dropout(.5),nn.Linear(ffn_hidden,256),nn.Dropout(.5))
        self.distill=TemporalAttention(256)
        self.head=nn.Linear(256,1)
    def factor_gates(self,market):
        regime=self.market_out(self.mamba(self.market_in(market)))
        return 158*torch.softmax(self.gate(regime)/self.beta,dim=-1)
    def forward(self,x):
        gates=self.factor_gates(x[:,:,158:221])
        z=self.position(self.projection(x[:,:,:158]*gates))
        z=self.spatial(self.temporal(z))
        return self.head(self.distill(z)).squeeze(-1)

def rank_position_loss(pred,target):
    pc=pred-pred.mean();tc=target-target.mean()
    pn=pc.norm();tn=tc.norm()
    if pred.numel()<30 or pn.detach()<1e-6 or tn.detach()<1e-6:
        return F.mse_loss(pred,target)
    with torch.no_grad():
        # Equal labels receive equal average ranks; indices are zero-based.
        _,inverse,counts=torch.unique(target,sorted=True,return_inverse=True,return_counts=True)
        last=counts.cumsum(0)-1
        average_rank=(last-(counts-1)/2)[inverse]
        percentile=average_rank/(target.numel()-1)
        weights=1+(2*percentile-1).square();weights=weights/weights.mean()
    weighted=((pred-target).square()*weights).mean()
    ic=1-(pc*tc).sum()/(pn*tn+1e-8)
    return .5*weighted+.5*ic
