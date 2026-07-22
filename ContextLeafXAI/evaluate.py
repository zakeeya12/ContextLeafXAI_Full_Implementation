from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.utils.config import load_config
from src.datasets.transforms import build_transforms
from src.datasets.multimodal_dataset import MultimodalPlantDataset
from src.models.leafcontextreasonnet import LeafContextReasonNet
from src.evaluation.metrics import classification_metrics


def main():
 ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);ap.add_argument("--checkpoint",required=True);a=ap.parse_args();c=load_config(a.config);d=c["data"];m=c["model"]
 ds=MultimodalPlantDataset(d["test_manifest"],build_transforms(d["image_size"],False),d["image_col"],d["label_col"],d.get("context_cols",[]),d.get("missingness_cols",[]),d.get("quality_cols",[]));loader=DataLoader(ds,batch_size=c["training"]["batch_size"],shuffle=False)
 model=LeafContextReasonNet(m["num_classes"],len(d.get("context_cols",[]))+len(d.get("missingness_cols",[])),len(d.get("quality_cols",[])),m["cnn_backbone"],m["vit_backbone"],False,m["visual_dim"],m["context_dim"],m["fusion_dim"],m["dropout"],m.get("mode","multimodal"));device=torch.device("cuda" if torch.cuda.is_available() else "cpu");model.load_state_dict(torch.load(a.checkpoint,map_location=device)["model"]);model.to(device).eval();ys=[];ps=[]
 with torch.no_grad():
  for b in loader:
   o=model(b["image"].to(device),b["context"].to(device),b["quality"].to(device));ys+=b["label"].tolist();ps+=torch.softmax(o["logits"],1).cpu().tolist()
 metrics=classification_metrics(ys,ps);out=Path(c["output_dir"])/c["experiment_name"]/"tables";out.mkdir(parents=True,exist_ok=True);json.dump(metrics,open(out/"test_metrics.json","w"),indent=2);print(json.dumps({k:v for k,v in metrics.items() if k not in ("report","confusion_matrix")},indent=2))
if __name__=="__main__":main()
