from pathlib import Path
import numpy as np,pandas as pd,cv2,yaml
from sklearn.model_selection import train_test_split
root=Path(__file__).resolve().parents[1];imgdir=root/"data/raw/synthetic";imgdir.mkdir(parents=True,exist_ok=True);rows=[];rng=np.random.default_rng(42)
for i in range(90):
 y=i%3;im=np.zeros((224,224,3),np.uint8);im[:]=[50+60*y,100,80];cv2.circle(im,(70+30*y,112),25+5*y,(200,40+40*y,40),-1);im=np.clip(im+rng.normal(0,8,im.shape),0,255).astype(np.uint8);p=imgdir/f"{i}.jpg";cv2.imwrite(str(p),im)
 rows.append({"sample_id":i,"image_path":str(p),"class_id":y,"class_name":f"class_{y}","temp_mean_7d":20+4*y+rng.normal(),"rain_sum_14d":5+10*y+rng.normal(),"soil_ph":6+0.2*y+rng.normal(0,.1),"miss_temp":0,"weather_completeness":.95})
df=pd.DataFrame(rows);tr,tmp=train_test_split(df,test_size=.3,stratify=df.class_id,random_state=42);va,te=train_test_split(tmp,test_size=.5,stratify=tmp.class_id,random_state=42);proc=root/"data/processed";proc.mkdir(parents=True,exist_ok=True)
for n,x in (("train",tr),("val",va),("test",te)):x.to_csv(proc/f"synthetic_{n}.csv",index=False)
cfg=yaml.safe_load(open(root/"configs/base.yaml"));cfg["experiment_name"]="synthetic";cfg["data"].update({"train_manifest":str(proc/"synthetic_train.csv"),"val_manifest":str(proc/"synthetic_val.csv"),"test_manifest":str(proc/"synthetic_test.csv"),"context_cols":["temp_mean_7d","rain_sum_14d","soil_ph"],"missingness_cols":["miss_temp"],"quality_cols":["weather_completeness"],"num_workers":0});cfg["model"].update({"num_classes":3,"cnn_backbone":"resnet18","vit_backbone":"vit_tiny_patch16_224","pretrained":False});cfg["training"].update({"epochs":2,"batch_size":4,"amp":False,"patience":2});yaml.safe_dump(cfg,open(root/"configs/synthetic.yaml","w"),sort_keys=False);print("Synthetic demo created")
