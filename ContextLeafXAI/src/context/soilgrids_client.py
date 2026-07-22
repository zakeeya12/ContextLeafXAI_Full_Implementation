from __future__ import annotations
import requests


def fetch_soilgrids(lat:float,lon:float,properties=("phh2o","soc","clay","sand","silt","bdod"),depths=("0-5cm","5-15cm")):
    url="https://rest.isric.org/soilgrids/v2.0/properties/query"
    params=[("lat",lat),("lon",lon)] + [("property",p) for p in properties] + [("depth",d) for d in depths] + [("value","mean")]
    r=requests.get(url,params=params,timeout=60); r.raise_for_status(); return r.json()
