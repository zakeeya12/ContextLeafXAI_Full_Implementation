from __future__ import annotations
from pathlib import Path
import json, math
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from src.evaluation.metrics import classification_metrics


def perturb_context(x,mask_prob,noise_std):
    if x.numel()==0:return x
    mask=torch.rand_like(x)<mask_prob
    return torch.where(mask,torch.zeros_like(x),x+torch.randn_like(x)*noise_std)


def run_epoch(model,loader,criterion,device,optimizer=None,scaler=None,grad_clip=5.0,amp=True,mask_prob=0.15,noise_std=0.05):
    train=optimizer is not None; model.train(train); total=0.; ys=[]; probs=[]
    for b in tqdm(loader,leave=False):
        image=b["image"].to(device); context=b["context"].to(device); quality=b["quality"].to(device); label=b["label"].to(device)
        if train: optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp and device.type=="cuda"):
            out=model(image,context,quality)
            pout=model(image,perturb_context(context,mask_prob,noise_std),quality) if train and model.mode not in ("visual_only","context_only") else None
            loss,_=criterion(out,label,pout)
        if train:
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip); scaler.step(optimizer); scaler.update()
        total+=loss.item()*label.size(0); ys.extend(label.detach().cpu().tolist()); probs.extend(torch.softmax(out["logits"],1).detach().cpu().tolist())
    return total/len(loader.dataset),classification_metrics(ys,probs)


def fit(model,train_loader,val_loader,criterion,optimizer,scheduler,device,epochs,patience,out_dir,**kwargs):
    out=Path(out_dir); (out/"checkpoints").mkdir(parents=True,exist_ok=True); (out/"logs").mkdir(parents=True,exist_ok=True)
    scaler=GradScaler(enabled=kwargs.get("amp",True) and device.type=="cuda"); best=math.inf; wait=0; hist=[]
    for epoch in range(1,epochs+1):
        tr_loss,tr_m=run_epoch(model,train_loader,criterion,device,optimizer,scaler,**kwargs)
        va_loss,va_m=run_epoch(model,val_loader,criterion,device,**kwargs)
        if scheduler:scheduler.step()
        rec={"epoch":epoch,"train_loss":tr_loss,"val_loss":va_loss,"train_macro_f1":tr_m["macro_f1"],"val_macro_f1":va_m["macro_f1"]}; hist.append(rec); print(rec)
        torch.save({"model":model.state_dict(),"epoch":epoch},out/"checkpoints"/"last.pt")
        if va_loss<best:
            best=va_loss;wait=0;torch.save({"model":model.state_dict(),"epoch":epoch,"metrics":va_m},out/"checkpoints"/"best.pt")
        else:
            wait+=1
            if wait>=patience: break
    json.dump(hist,open(out/"logs"/"history.json","w"),indent=2)
