"""PRISM-VQ official-module adapter for MASTER-EXT25 (NOT exact reproduction).
Upstream: finxlab/PRISM-VQ, commit 7d02635d0cdeec2f4e7278a9e73acd0eb67a8fa6 (MIT).
Reuses the upstream VQVAE and LoadingGenerator on the common benchmark.
Declared adaptations: 8-day window; Market63 instead of JKP13; single 5-day
training target in the auxiliary decoder; common MASTER-EXT25 splits/labels.
"""
from __future__ import annotations
import copy, sys
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
UPSTREAM=ROOT/'third_party/PRISM-VQ'
if not (UPSTREAM/'module/autoencoder.py').exists():
    raise FileNotFoundError('Missing bundled third_party/PRISM-VQ/module')
sys.path.insert(0,str(UPSTREAM))
from module.autoencoder import VQVAE
from module.bidirectional import LoadingGenerator

# Upstream replaces LayerNorm with RMSNorm. PyTorch's fused encoder fast path
# assumes LayerNorm semantics; disable it to preserve the actual RMSNorm forward.
torch.backends.mha.set_fastpath_enabled(False)

CONFIG={
 'vqvae':dict(num_features=158,seq_len=8,hidden_size=128,num_prior_factors=63,vq_embed_dim=128,num_embed=512,
   encoder=dict(num_heads=2,num_layers=1),
   quantizer=dict(decay=.95,commit_weight=.25,distance='l2',anchor='probrandom',first_batch=False,contras_loss=True),
   decoder=dict(norm_type='none',num_groups=8,hidden_channels=128,initial_T=2),
   predictor=dict(pred_len=1,num_gru_layers=1,dropout=.1,output_dim=1,pred_weight=.0001)),
 'predictor':dict(num_features=158,individual=False,aux_weight=.01,kernel_size=3,k=1,n_expert=2,pred_len=8,moe_hidden=64,dropout=.1,
   transformer=dict(pe_kind='rope',num_heads=2,num_layers=1,d_model=64,dim_feedforward=128,dropout=.1,batch_first=True,prepend_structure_token=True),
   rank=0,target_day=5,use_prior=True,aux_imp=3)
}

class PrismOfficialAdapter(nn.Module):
    def __init__(self,spatial):
        super().__init__()
        self.spatial=spatial
        for p in spatial.parameters(): p.requires_grad_(False)
        self.loadings=LoadingGenerator(copy.deepcopy(CONFIG))
        self.prior_norm=nn.LayerNorm(63)
        self.latent_value=nn.Sequential(nn.Linear(128,128),nn.LayerNorm(128),nn.GELU(),nn.Linear(128,128))
        self.aux_loss=None
        self.spatial.eval()
    def train(self,mode=True):
        super().train(mode)
        # A frozen quantizer must not re-anchor codewords during stage 2.
        self.spatial.eval()
        return self
    def forward(self,x):
        feature=x[:,:,:158]; prior=x[:,-1,158:]
        with torch.no_grad():
            normalized=self.spatial.revin(feature,mode='norm')
            h=self.spatial.spatial_encoder(normalized)
            z,_,_=self.spatial.quantizer(h)
        alpha,beta_p,beta_l,aux=self.loadings(feature,z)
        latent=self.latent_value(z)
        self.aux_loss=3*torch.log1p(torch.relu(aux)/3)
        return alpha+(beta_p*self.prior_norm(prior)).sum(-1)+(beta_l*latent).sum(-1)

def build_spatial(): return VQVAE(copy.deepcopy(CONFIG))
