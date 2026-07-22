from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--manifest",required=True);ap.add_argument("--out-dir",required=True);ap.add_argument("--label-col",default="class_id");ap.add_argument("--seed",type=int,default=42);a=ap.parse_args();df=pd.read_csv(a.manifest)
 train,temp=train_test_split(df,test_size=0.30,stratify=df[a.label_col],random_state=a.seed);val,test=train_test_split(temp,test_size=0.50,stratify=temp[a.label_col],random_state=a.seed);o=Path(a.out_dir);o.mkdir(parents=True,exist_ok=True)
 for n,x in (("train",train),("val",val),("test",test)):x.to_csv(o/f"{n}.csv",index=False)
 print({"train":len(train),"val":len(val),"test":len(test)})
if __name__=="__main__":main()
