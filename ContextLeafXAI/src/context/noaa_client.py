from __future__ import annotations
import requests


def fetch_ghcn_daily(station_id:str,start_date:str,end_date:str,token:str|None=None):
    url="https://www.ncei.noaa.gov/access/services/data/v1"
    params={"dataset":"daily-summaries","stations":station_id,"startDate":start_date,"endDate":end_date,"format":"json","units":"metric","includeAttributes":"false"}
    headers={"token":token} if token else {}
    r=requests.get(url,params=params,headers=headers,timeout=60); r.raise_for_status(); return r.json()
