from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,roc_auc_score,confusion_matrix,classification_report


def classification_metrics(y_true,y_prob):
    y_prob=np.asarray(y_prob); y_pred=y_prob.argmax(1)
    p,r,f,_=precision_recall_fscore_support(y_true,y_pred,average="macro",zero_division=0)
    out={"accuracy":float(accuracy_score(y_true,y_pred)),"macro_precision":float(p),"macro_recall":float(r),"macro_f1":float(f)}
    try: out["macro_roc_auc_ovr"]=float(roc_auc_score(y_true,y_prob,multi_class="ovr",average="macro"))
    except ValueError: out["macro_roc_auc_ovr"]=float("nan")
    out["confusion_matrix"]=confusion_matrix(y_true,y_pred).tolist(); out["report"]=classification_report(y_true,y_pred,zero_division=0,output_dict=True)
    return out
