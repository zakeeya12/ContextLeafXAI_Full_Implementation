from __future__ import annotations
import torch
from torch import nn
import timm


class HybridVisualEncoder(nn.Module):
    def __init__(self, cnn_name="efficientnet_b3", vit_name="vit_base_patch16_224", out_dim=512, pretrained=True):
        super().__init__()
        self.cnn = timm.create_model(cnn_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        self.vit = timm.create_model(vit_name, pretrained=pretrained, num_classes=0, global_pool="token")
        self.cnn_proj = nn.Sequential(nn.Linear(self.cnn.num_features, out_dim), nn.LayerNorm(out_dim), nn.GELU())
        self.vit_proj = nn.Sequential(nn.Linear(self.vit.num_features, out_dim), nn.LayerNorm(out_dim), nn.GELU())
        self.branch_gate = nn.Sequential(nn.Linear(out_dim*2, out_dim), nn.Sigmoid())
        self._last_conv = None
        target = getattr(self.cnn, "conv_head", None)
        if target is None:
            for m in reversed(list(self.cnn.modules())):
                if isinstance(m, nn.Conv2d): target = m; break
        if target is not None:
            target.register_forward_hook(lambda _m,_i,o: setattr(self,"_last_conv",o))

    def forward(self, x):
        c = self.cnn_proj(self.cnn(x)); v = self.vit_proj(self.vit(x))
        a = self.branch_gate(torch.cat([c,v], dim=1))
        return a*c + (1-a)*v, {"cnn": c, "vit": v, "branch_gate": a, "feature_map": self._last_conv}


class ContextEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden=(256,128), out_dim=128, dropout=0.3):
        super().__init__()
        dims=[input_dim,*hidden,out_dim]; layers=[]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i],dims[i+1]))
            if i < len(dims)-2:
                layers += [nn.BatchNorm1d(dims[i+1]), nn.GELU(), nn.Dropout(dropout)]
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)
