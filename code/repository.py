"""資料存取層：所有讀 Data/travel.db 的地方都走這裡，system.py/streamlit_app.py 不直接碰 SQL。
新資料源沒有 Rating、沒有餐廳(restaurant)類別（見 ROADMAP.md Phase 0 決策），呼叫端要接受這兩點。

city/district 一律在這裡做「台→臺」正規化再進 SQL，因為資料源存的是官方全名（臺中市，非台中市），
呼叫端（LLM 解析出的文字、使用者輸入）常用簡寫，正規化只做這一處，不必每個呼叫端各自處理。
"""
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

DB_PATH = Path(__file__).resolve().parents[1] / "Data" / "travel.db"

ATTRACTION_COLS = "id, name, city, district, latitude, longitude, category, description, weather_fit"
LODGING_COLS = "id, name, city, district, latitude, longitude, license_no, description, price_range"
RESTAURANT_COLS = "id, name, city, district, latitude, longitude, category, description, open_hours"


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def _query(table: str, cols: str, city: Optional[str], district: Optional[str], top_n: int) -> pd.DataFrame:
    sql = f"SELECT {cols} FROM {table} WHERE 1=1"
    params: list = []
    if city:
        sql += " AND city = ?"
        params.append(city.replace("台", "臺"))
    if district:
        norm = district.replace("台", "臺")
        sql += " AND district LIKE ?"
        params.append(f"%{norm}%")
    sql += " ORDER BY name LIMIT ?"
    params.append(top_n)
    with _connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def query_attractions(city: Optional[str] = None, district: Optional[str] = None, top_n: int = 5) -> pd.DataFrame:
    return _query("attractions", ATTRACTION_COLS, city, district, top_n)


def query_lodgings(city: Optional[str] = None, district: Optional[str] = None, top_n: int = 3) -> pd.DataFrame:
    return _query("lodgings", LODGING_COLS, city, district, top_n)


def query_restaurants(city: Optional[str] = None, district: Optional[str] = None, top_n: int = 5) -> pd.DataFrame:
    return _query("restaurants", RESTAURANT_COLS, city, district, top_n)


def get_attractions_for_area(area: str, top_n: int = 5, city: str = "臺中市") -> list[str]:
    """相容舊介面：回傳名稱清單。city 預設台中僅作備援，呼叫端(system.py)已可傳全台縣市。"""
    return query_attractions(city=city, district=area, top_n=top_n)["name"].tolist()


def get_lodgings_for_area(area: str, top_n: int = 3, city: str = "臺中市") -> list[str]:
    return query_lodgings(city=city, district=area, top_n=top_n)["name"].tolist()


def get_district_coords(city: str, district: str) -> Optional[tuple[float, float]]:
    """行政區中心座標查詢（見 collector/derive_district_coords.py 產生的 district_coords 表）。
    查無該區（例如極少數完全沒景點/住宿/餐飲資料的偏遠鄉鎮）時，退回同縣市其他區的平均座標；
    連縣市都查無資料則回傳 None，由呼叫端決定最終備援值。"""
    city_norm = city.replace("台", "臺") if city else city
    district_norm = district.replace("台", "臺") if district else district
    with _connect() as conn:
        row = conn.execute(
            "SELECT latitude, longitude FROM district_coords WHERE city = ? AND district = ?",
            (city_norm, district_norm),
        ).fetchone()
        if row:
            return row
        row = conn.execute(
            "SELECT AVG(latitude), AVG(longitude) FROM district_coords WHERE city = ?",
            (city_norm,),
        ).fetchone()
        if row and row[0] is not None:
            return row
    return None
