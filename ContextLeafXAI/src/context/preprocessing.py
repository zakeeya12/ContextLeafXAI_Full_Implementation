from __future__ import annotations
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class ContextPreprocessor:
    def __init__(self, numerical_cols):
        self.cols=list(numerical_cols); self.imputer=SimpleImputer(strategy="median",add_indicator=False); self.scaler=StandardScaler()
    def fit(self,df:pd.DataFrame):
        x=self.imputer.fit_transform(df[self.cols]); self.scaler.fit(x); return self
    def transform(self,df): return self.scaler.transform(self.imputer.transform(df[self.cols]))
    def save(self,path): joblib.dump(self,path)
    @staticmethod
    def load(path): return joblib.load(path)
