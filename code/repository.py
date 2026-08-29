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

        # LLM 偶爾會把縣市名稱黏在一起回傳（例如把「彰化市」誤判成「彰化縣彰化市」），
        # 導致上面兩個精確查詢都落空、靜靜地退回預設座標（使用者問彰化卻查到台中天氣，
        # 沒有任何錯誤訊息）。這裡多一步：檢查 city_norm 裡有沒有包住某個真實存在的
        # 縣市全名，包到的話用那個乾淨的縣市名重查一次。
        if city_norm:
            known_cities = [r[0] for r in conn.execute("SELECT DISTINCT city FROM district_coords")]
            matches = [c for c in known_cities if c in city_norm]
            if len(matches) == 1:
                return get_district_coords(matches[0], district)
    return None


def resolve_unique_city_for_district(district: str) -> Optional[str]:
    """行政區名稱如果全臺只對應唯一一個縣市，直接回傳那個縣市，不用猜。用在
    `analyze_prompt_with_llm()` 校正 LLM 猜錯縣市的情況（例如 LLM 把「復興區」
    這種只存在於桃園市的行政區猜成臺中市——這種obscure鄉鎮的縣市歸屬本來就是
    查表就能查到的正確答案，不該靠 LLM 的世界知識去猜）。同名跨縣市的行政區
    （中山區/中正區等）回傳 None，交給 LLM 自己從問句上下文判斷，不亂覆寫。"""
    if not district:
        return None
    district_norm = district.replace("台", "臺")
    with _connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT city FROM district_coords WHERE district = ?",
            (district_norm,),
        ).fetchall()
    if len(rows) == 1:
        return rows[0][0]
    return None
