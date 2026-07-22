from __future__ import annotations
import torch
from torch import nn
from .encoders import HybridVisualEncoder, ContextEncoder
from .fusion import BidirectionalInteraction, ReliabilityAwareFusion


class LeafContextReasonNet(nn.Module):
    def __init__(self, num_classes:int, context_input_dim:int, quality_dim:int=0,
                 cnn_backbone="efficientnet_b3", vit_backbone="vit_base_patch16_224",
                 pretrained=True, visual_dim=512, context_dim=128, fusion_dim=256, dropout=0.3,
                 mode="multimodal"):
        super().__init__(); self.mode=mode
        self.visual=HybridVisualEncoder(cnn_backbone,vit_backbone,visual_dim,pretrained)
        self.context=ContextEncoder(max(context_input_dim,1),out_dim=context_dim,dropout=dropout)
        self.vproj=nn.Linear(visual_dim,fusion_dim); self.cproj=nn.Linear(context_dim,fusion_dim)
        self.interaction=BidirectionalInteraction(fusion_dim)
        self.fusion=ReliabilityAwareFusion(fusion_dim,quality_dim)
        self.classifier=nn.Sequential(nn.Linear(fusion_dim,128),nn.GELU(),nn.Dropout(dropout),nn.Linear(128,num_classes))
        self.visual_head=nn.Linear(fusion_dim,num_classes); self.context_head=nn.Linear(fusion_dim,num_classes)
        self.context_input_dim=context_input_dim
    def forward(self,image,context,quality):
        v,vaux=self.visual(image); v=self.vproj(v)
        if self.context_input_dim==0:
            context=torch.zeros(image.size(0),1,device=image.device,dtype=image.dtype)
        c=self.cproj(self.context(context))
        if self.mode=="visual_only": z=v; faux={}
        elif self.mode=="context_only": z=c; faux={}
        elif self.mode=="early_fusion": z=0.5*(v+c); faux={}
        else:
            v,c=self.interaction(v,c); z,faux=self.fusion(v,c,quality)
        logits=self.classifier(z)
        return {"logits":logits,"visual_embedding":v,"context_embedding":c,"fused_embedding":z,
                "visual_logits":self.visual_head(v),"context_logits":self.context_head(c),**vaux,**faux}
