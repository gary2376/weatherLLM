"""抓交通部觀光署「觀光資訊資料庫」全台景點/旅館民宿資料，寫入 Data/travel.db。
資料源：data.gov.tw 資料集 7777（景點）/7780（旅館民宿），免key免登入，每日更新。
用法：python3 collector/fetch_tourism.py
"""
import io
import json
import zipfile
from datetime import datetime, timezone

import requests

from db import get_conn
from attraction_classes import classify_weather_fit, labels_for
from cuisine_classes import labels_for as cuisine_labels_for

ATTRACTION_ZIP = "https://media.taiwan.net.tw/XMLReleaseAll_public/v2.0/Zh_tw/Attraction-json.zip"
HOTEL_ZIP = "https://media.taiwan.net.tw/XMLReleaseAll_public/v2.0/Zh_tw/Hotel-json.zip"
RESTAURANT_ZIP = "https://media.taiwan.net.tw/XMLReleaseAll_public/v2.0/Zh_tw/Restaurant-json.zip"
SOURCE_ATTRACTION = "data.gov.tw/dataset/7777"
SOURCE_HOTEL = "data.gov.tw/dataset/7780"
SOURCE_RESTAURANT = "data.gov.tw/dataset/7779"


def _clean_latlon(lat, lon):
    """觀光署原始資料有些筆沒座標時填 (0,0) 佔位，不是真的在幾內亞灣，當缺值處理。"""
    if lat in (0, 0.0) and lon in (0, 0.0):
        return None, None
    return lat, lon


def _download_json(url: str, member_suffix: str) -> dict:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    name = next(n for n in zf.namelist() if n.endswith(member_suffix))
    with zf.open(name) as f:
        return json.load(io.TextIOWrapper(f, encoding="utf-8-sig"))


def load_attractions(conn, fetched_at: str) -> int:
    raw = _download_json(ATTRACTION_ZIP, "AttractionList.json")
    rows = raw["Attractions"]
    n = 0
    for a in rows:
        addr = a.get("PostalAddress") or {}
        codes = a.get("AttractionClasses") or []
        lat, lon = _clean_latlon(a.get("PositionLat"), a.get("PositionLon"))
        conn.execute(
            """INSERT INTO attractions
               (id, name, city, district, latitude, longitude, category,
                description, open_hours, ticket_info, weather_fit, suitable_for,
                source, fetched_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 name=excluded.name, city=excluded.city, district=excluded.district,
                 latitude=excluded.latitude, longitude=excluded.longitude,
                 category=excluded.category, description=excluded.description,
                 open_hours=excluded.open_hours, ticket_info=excluded.ticket_info,
                 weather_fit=excluded.weather_fit, source=excluded.source,
                 fetched_at=excluded.fetched_at""",
            (
                a["AttractionID"], a.get("AttractionName"),
                addr.get("City"), addr.get("Town"),
                lat, lon,
                labels_for(codes), a.get("Description"),
                a.get("ServiceTimeInfo") or None, a.get("FeeInfo") or None,
                classify_weather_fit(codes), None,
                SOURCE_ATTRACTION, fetched_at,
            ),
        )
        n += 1
    conn.commit()
    return n


def load_hotels(conn, fetched_at: str) -> int:
    raw = _download_json(HOTEL_ZIP, "HotelList.json")
    rows = raw["Hotels"]
    n = 0
    for h in rows:
        addr = h.get("PostalAddress") or {}
        lat, lon = _clean_latlon(h.get("PositionLat"), h.get("PositionLon"))
        low, high = h.get("LowestPrice"), h.get("CeilingPrice")
        price_range = f"{low}~{high}" if low or high else None
        conn.execute(
            """INSERT INTO lodgings
               (id, name, city, district, latitude, longitude, license_no,
                description, price_range, source, fetched_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 name=excluded.name, city=excluded.city, district=excluded.district,
                 latitude=excluded.latitude, longitude=excluded.longitude,
                 license_no=excluded.license_no, description=excluded.description,
                 price_range=excluded.price_range, source=excluded.source,
                 fetched_at=excluded.fetched_at""",
            (
                h["HotelID"], h.get("HotelName"),
                addr.get("City"), addr.get("Town"),
                lat, lon,
                h.get("HotelLicenseNumber") or None,
                h.get("Description"), price_range,
                SOURCE_HOTEL, fetched_at,
            ),
        )
        n += 1
    conn.commit()
    return n


def load_restaurants(conn, fetched_at: str) -> int:
    raw = _download_json(RESTAURANT_ZIP, "RestaurantList.json")
    rows = raw["Restaurants"]
    n = 0
    for r in rows:
        addr = r.get("PostalAddress") or {}
        codes = r.get("CuisineClasses") or []
        lat, lon = _clean_latlon(r.get("PositionLat"), r.get("PositionLon"))
        conn.execute(
            """INSERT INTO restaurants
               (id, name, city, district, latitude, longitude, category,
                description, open_hours, source, fetched_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 name=excluded.name, city=excluded.city, district=excluded.district,
                 latitude=excluded.latitude, longitude=excluded.longitude,
                 category=excluded.category, description=excluded.description,
                 open_hours=excluded.open_hours, source=excluded.source,
                 fetched_at=excluded.fetched_at""",
            (
                r["RestaurantID"], r.get("RestaurantName"),
                addr.get("City"), addr.get("Town"),
                lat, lon,
                cuisine_labels_for(codes), r.get("Description"),
                r.get("ServiceTimeInfo") or None,
                SOURCE_RESTAURANT, fetched_at,
            ),
        )
        n += 1
    conn.commit()
    return n


if __name__ == "__main__":
    fetched_at = datetime.now(timezone.utc).isoformat()
    conn = get_conn()
    n_attr = load_attractions(conn, fetched_at)
    n_hotel = load_hotels(conn, fetched_at)
    n_rest = load_restaurants(conn, fetched_at)
    print(f"attractions: {n_attr} rows, lodgings: {n_hotel} rows, restaurants: {n_rest} rows")
    conn.close()
