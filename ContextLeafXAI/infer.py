from __future__ import annotations
import argparse,json,cv2,torch
from src.utils.config import load_config
from src.datasets.transforms import build_transforms
from src.models.leafcontextreasonnet import LeafContextReasonNet

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);ap.add_argument("--checkpoint",required=True);ap.add_argument("--image",required=True);ap.add_argument("--context-json");a=ap.parse_args();c=load_config(a.config);d=c["data"];m=c["model"];ctx=json.load(open(a.context_json)) if a.context_json else {}
 image=cv2.cvtColor(cv2.imread(a.image),cv2.COLOR_BGR2RGB);x=build_transforms(d["image_size"],False)(image=image)["image"].unsqueeze(0);cols=d.get("context_cols",[])+d.get("missingness_cols",[]);qcols=d.get("quality_cols",[]);context=torch.tensor([[float(ctx.get(k,0)) for k in cols]],dtype=torch.float32);quality=torch.tensor([[float(ctx.get(k,0)) for k in qcols]],dtype=torch.float32)
 device=torch.device("cuda" if torch.cuda.is_available() else "cpu");model=LeafContextReasonNet(m["num_classes"],len(cols),len(qcols),m["cnn_backbone"],m["vit_backbone"],False,m["visual_dim"],m["context_dim"],m["fusion_dim"],m["dropout"],m.get("mode","multimodal"));model.load_state_dict(torch.load(a.checkpoint,map_location=device)["model"]);model.to(device).eval()
 with torch.no_grad(): o=model(x.to(device),context.to(device),quality.to(device));p=torch.softmax(o["logits"],1)[0];print(json.dumps({"predicted_class_id":int(p.argmax()),"confidence":float(p.max()),"probabilities":p.cpu().tolist()},indent=2))
if __name__=="__main__":main()
