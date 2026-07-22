import torch
from src.models.leafcontextreasonnet import LeafContextReasonNet

def test_forward():
 m=LeafContextReasonNet(3,4,1,cnn_backbone="resnet18",vit_backbone="vit_tiny_patch16_224",pretrained=False,visual_dim=64,context_dim=32,fusion_dim=32)
 o=m(torch.randn(2,3,224,224),torch.randn(2,4),torch.randn(2,1));assert o["logits"].shape==(2,3)
