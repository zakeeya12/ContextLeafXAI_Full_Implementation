from __future__ import annotations
import torch
from torch import nn


class BidirectionalInteraction(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.c2v=nn.Linear(dim,dim*2); self.v2c=nn.Linear(dim,dim*2)
    def forward(self,v,c):
        gv,bv=self.c2v(c).chunk(2,1); gc,bc=self.v2c(v).chunk(2,1)
        v2=v + torch.sigmoid(gv)*v + bv
        c2=c + torch.sigmoid(gc)*c + bc
        return v2,c2


class ReliabilityAwareFusion(nn.Module):
    def __init__(self, dim: int, quality_dim: int):
        super().__init__()
        q=max(quality_dim,1)
        self.visual_rel=nn.Sequential(nn.Linear(dim,64),nn.GELU(),nn.Linear(64,1),nn.Sigmoid())
        self.context_rel=nn.Sequential(nn.Linear(dim+q,64),nn.GELU(),nn.Linear(64,1),nn.Sigmoid())
        self.gate=nn.Sequential(nn.Linear(dim*2+2+q,dim),nn.Sigmoid())
        self.residual=nn.Linear(dim*2,dim); self.norm=nn.LayerNorm(dim)
        self.quality_dim=quality_dim
    def forward(self,v,c,quality):
        if self.quality_dim==0:
            quality=torch.zeros(v.size(0),1,device=v.device,dtype=v.dtype)
        rv=self.visual_rel(v); rc=self.context_rel(torch.cat([c,quality],1))
        g=self.gate(torch.cat([v,c,rv,rc,quality],1))
        fused=g*v+(1-g)*c
        fused=self.norm(fused+self.residual(torch.cat([v,c],1)))
        return fused,{"fusion_gate":g,"visual_reliability":rv,"context_reliability":rc}
