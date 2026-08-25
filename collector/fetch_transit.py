"""抓 TDX 全台公車站牌/YouBike站點/路外停車場靜態資料，寫入 Data/travel.db。
需要 .env 裡的 TDX_CLIENT_ID / TDX_CLIENT_SECRET（免費會員即可，這三個 API 不吃虛擬點數，
受 5次/分/金鑰 頻率限制，本腳本已內建節流）。
用法：python3 collector/fetch_transit.py
"""
import os
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

from db import get_conn

load_dotenv()

AUTH_URL = "https://tdx.transportdata.tw/auth/realms/TDXConnect/protocol/openid-connect/token"
THROTTLE_SECONDS = 13  # 5次/分 => 至少間隔12秒，留餘裕

# TDX 用英文城市代碼，但 attractions/lodgings 表存的是官方中文縣市名（來自觀光署 PostalAddress.City），
# 這裡統一轉成中文寫進 DB，不然 attraction_transit join 時 city 對不上，最近站點永遠算不出來
# （踩過這個坑：第一版沒轉換，link_transit.py 誤判成「都沒有關聯」，把整張 transit_stations 清空過一次）
CITY_ZH = {
    "Taipei": "臺北市", "NewTaipei": "新北市", "Taoyuan": "桃園市",
    "Taichung": "臺中市", "Tainan": "臺南市", "Kaohsiung": "高雄市",
    "Keelung": "基隆市", "Hsinchu": "新竹市", "HsinchuCounty": "新竹縣",
    "MiaoliCounty": "苗栗縣", "ChanghuaCounty": "彰化縣", "NantouCounty": "南投縣",
    "YunlinCounty": "雲林縣", "Chiayi": "嘉義市", "ChiayiCounty": "嘉義縣",
    "PingtungCounty": "屏東縣", "YilanCounty": "宜蘭縣", "HualienCounty": "花蓮縣",
    "TaitungCounty": "臺東縣", "PenghuCounty": "澎湖縣", "KinmenCounty": "金門縣",
    "LienchiangCounty": "連江縣",
}

# 這份是 TDX 官方實際接受的城市代碼（跟 400 錯誤訊息回傳的清單核對過，
# 不是隨便猜的英文：新竹市是 "Hsinchu" 不是 "HsinchuCity"，嘉義市是 "Chiayi" 不是 "ChiayiCity"）
CITIES = [
    "Taipei", "NewTaipei", "Taoyuan", "Taichung", "Tainan", "Kaohsiung",
    "Keelung", "Hsinchu", "HsinchuCounty", "MiaoliCounty", "ChanghuaCounty",
    "NantouCounty", "YunlinCounty", "Chiayi", "ChiayiCounty", "PingtungCounty",
    "YilanCounty", "HualienCounty", "TaitungCounty", "PenghuCounty",
    "KinmenCounty", "LienchiangCounty",
]

ENDPOINTS = {
    "bus": ("https://tdx.transportdata.tw/api/basic/v2/Bus/Stop/City/{city}", "Stops"),
    "bike": ("https://tdx.transportdata.tw/api/basic/v2/Bike/Station/City/{city}", "Stations"),
    "parking": ("https://tdx.transportdata.tw/api/basic/v1/Parking/OffStreet/CarPark/City/{city}", "CarParks"),
}


def get_token() -> str:
    resp = requests.post(
        AUTH_URL,
        headers={"content-type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "client_credentials",
            "client_id": os.getenv("TDX_CLIENT_ID"),
            "client_secret": os.getenv("TDX_CLIENT_SECRET"),
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _extract(kind: str, item: dict):
    if kind == "bus":
        pos = item.get("StopPosition") or {}
        return item.get("StopName", {}).get("Zh_tw") if isinstance(item.get("StopName"), dict) else item.get("StopName"), \
            item.get("StopUID"), pos.get("PositionLat"), pos.get("PositionLon")
    if kind == "bike":
        pos = item.get("StationPosition") or {}
        name = item.get("StationName")
        name = name.get("Zh_tw") if isinstance(name, dict) else name
        return name, item.get("StationUID"), pos.get("PositionLat"), pos.get("PositionLon")
    if kind == "parking":
        pos = item.get("CarParkPosition") or {}
        name = item.get("CarParkName")
        name = name.get("Zh_tw") if isinstance(name, dict) else name
        return name, item.get("CarParkID"), pos.get("PositionLat"), pos.get("PositionLon")
    raise ValueError(kind)


def fetch_all():
    token = get_token()
    headers = {"authorization": f"Bearer {token}"}
    fetched_at = datetime.now(timezone.utc).isoformat()
    conn = get_conn()
    total = 0
    for city in CITIES:
        for kind, (url_tpl, list_key) in ENDPOINTS.items():
            url = url_tpl.format(city=city)
            try:
                r = requests.get(url, headers=headers, params={"$format": "JSON"}, timeout=30)
            except requests.RequestException as e:
                print(f"[skip] {city}/{kind}: request error {e}")
                time.sleep(THROTTLE_SECONDS)
                continue
            if r.status_code != 200:
                print(f"[skip] {city}/{kind}: HTTP {r.status_code} {r.text[:120]}")
                time.sleep(THROTTLE_SECONDS)
                continue
            body = r.json()
            items = body.get(list_key) if isinstance(body, dict) else body
            items = items or []
            for item in items:
                name, tdx_id, lat, lon = _extract(kind, item)
                if lat is None or lon is None:
                    continue
                conn.execute(
                    """INSERT INTO transit_stations
                       (station_name, city, type, tdx_id, latitude, longitude, source, fetched_at)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (name, CITY_ZH.get(city, city), kind, tdx_id, lat, lon, "TDX", fetched_at),
                )
            conn.commit()
            print(f"{city}/{kind}: {len(items)} rows")
            total += len(items)
            time.sleep(THROTTLE_SECONDS)
    conn.close()
    print(f"done, total rows inserted: {total}")


if __name__ == "__main__":
    fetch_all()
