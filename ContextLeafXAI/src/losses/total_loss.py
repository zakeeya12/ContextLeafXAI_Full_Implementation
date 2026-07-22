from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


def js_divergence(p_logits,q_logits):
    p=F.softmax(p_logits,1); q=F.softmax(q_logits,1); m=0.5*(p+q)
    return 0.5*(F.kl_div(m.log(),p,reduction="batchmean")+F.kl_div(m.log(),q,reduction="batchmean"))


class ContextLeafLoss(nn.Module):
    def __init__(self,class_weights=None,lambda_context=0.2,lambda_agreement=0.1):
        super().__init__(); self.ce=nn.CrossEntropyLoss(weight=class_weights)
        self.lc=lambda_context; self.la=lambda_agreement
    def forward(self,out,label,perturbed_out=None):
        cls=self.ce(out["logits"],label)
        ctx=torch.tensor(0.,device=cls.device)
        if perturbed_out is not None: ctx=js_divergence(out["logits"],perturbed_out["logits"])
        v=F.normalize(out["visual_embedding"],dim=1); c=F.normalize(out["context_embedding"],dim=1)
        agr=(1-(v*c).sum(1)).mean()
        total=cls+self.lc*ctx+self.la*agr
        return total,{"classification":cls.detach(),"context_consistency":ctx.detach(),"agreement":agr.detach()}
