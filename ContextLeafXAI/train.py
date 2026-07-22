from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.utils.config import load_config
from src.utils.seed import seed_everything
from src.datasets.transforms import build_transforms
from src.datasets.multimodal_dataset import MultimodalPlantDataset
from src.models.leafcontextreasonnet import LeafContextReasonNet
from src.losses.total_loss import ContextLeafLoss
from src.engine.trainer import fit


def make_loader(cfg,path,train):
    d=cfg["data"]
    ds=MultimodalPlantDataset(path,build_transforms(d["image_size"],train),d["image_col"],d["label_col"],d.get("context_cols",[]),d.get("missingness_cols",[]),d.get("quality_cols",[]))
    return DataLoader(ds,batch_size=cfg["training"]["batch_size"],shuffle=train,num_workers=d.get("num_workers",2),pin_memory=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);a=ap.parse_args();cfg=load_config(a.config);seed_everything(cfg.get("seed",42))
    tr=make_loader(cfg,cfg["data"]["train_manifest"],True);va=make_loader(cfg,cfg["data"]["val_manifest"],False)
    m=cfg["model"]; context_dim=len(cfg["data"].get("context_cols",[]))+len(cfg["data"].get("missingness_cols",[])); quality_dim=len(cfg["data"].get("quality_cols",[]))
    model=LeafContextReasonNet(m["num_classes"],context_dim,quality_dim,m["cnn_backbone"],m["vit_backbone"],m["pretrained"],m["visual_dim"],m["context_dim"],m["fusion_dim"],m["dropout"],m.get("mode","multimodal"))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    weights=None
    if cfg["training"].get("class_weighting",True):
        y=pd.read_csv(cfg["data"]["train_manifest"])[cfg["data"]["label_col"]].values; counts=pd.Series(y).value_counts().sort_index(); w=len(y)/(len(counts)*counts.values);weights=torch.tensor(w,dtype=torch.float32,device=device)
    crit=ContextLeafLoss(weights,cfg["training"]["lambda_context_consistency"],cfg["training"]["lambda_explanation_agreement"])
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["training"]["lr"],weight_decay=cfg["training"]["weight_decay"])
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg["training"]["epochs"],eta_min=cfg["scheduler"]["min_lr"])
    out=Path(cfg["output_dir"])/cfg["experiment_name"]
    fit(model,tr,va,crit,opt,sch,device,cfg["training"]["epochs"],cfg["training"]["patience"],out,
        grad_clip=cfg["training"]["grad_clip"],amp=cfg["training"]["amp"],mask_prob=cfg["training"]["context_mask_prob"],noise_std=cfg["training"]["context_noise_std"])
if __name__=="__main__":main()
