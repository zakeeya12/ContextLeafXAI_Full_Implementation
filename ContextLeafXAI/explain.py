from __future__ import annotations
import argparse,json,cv2,numpy as np,torch
from pathlib import Path
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from src.utils.config import load_config
from src.datasets.transforms import build_transforms
from src.models.leafcontextreasonnet import LeafContextReasonNet


def main():
 ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);ap.add_argument("--checkpoint",required=True);ap.add_argument("--image",required=True);ap.add_argument("--context-json");a=ap.parse_args();c=load_config(a.config);d=c["data"];m=c["model"];ctx=json.load(open(a.context_json)) if a.context_json else {};cols=d.get("context_cols",[])+d.get("missingness_cols",[]);qcols=d.get("quality_cols",[])
 raw=cv2.cvtColor(cv2.imread(a.image),cv2.COLOR_BGR2RGB);x=build_transforms(d["image_size"],False)(image=raw)["image"].unsqueeze(0);ct=torch.tensor([[float(ctx.get(k,0)) for k in cols]],dtype=torch.float32);qt=torch.tensor([[float(ctx.get(k,0)) for k in qcols]],dtype=torch.float32)
 device=torch.device("cuda" if torch.cuda.is_available() else "cpu");model=LeafContextReasonNet(m["num_classes"],len(cols),len(qcols),m["cnn_backbone"],m["vit_backbone"],False,m["visual_dim"],m["context_dim"],m["fusion_dim"],m["dropout"],m.get("mode","multimodal"));model.load_state_dict(torch.load(a.checkpoint,map_location=device)["model"]);model.to(device).eval();x,ct,qt=x.to(device),ct.to(device),qt.to(device)
 with torch.no_grad(): pred=int(model(x,ct,qt)["logits"].argmax(1))
 if cols:
  ig=IntegratedGradients(lambda z:model(x,z,qt)["logits"][:,pred]);attr=ig.attribute(ct,baselines=torch.zeros_like(ct)).detach().cpu().numpy()[0];print(json.dumps(dict(zip(cols,map(float,attr))),indent=2))
 out=Path(c["output_dir"])/c["experiment_name"]/"explanations";out.mkdir(parents=True,exist_ok=True);plt.figure(figsize=(5,5));plt.imshow(raw);plt.axis("off");plt.title(f"Prediction: {pred}");plt.tight_layout();plt.savefig(out/"explanation_input.png",dpi=300);print(f"Saved to {out}")
if __name__=="__main__":main()
