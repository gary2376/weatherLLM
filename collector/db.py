import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "Data" / "travel.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS attractions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    city TEXT,
    district TEXT,
    latitude REAL,
    longitude REAL,
    category TEXT,
    description TEXT,
    open_hours TEXT,
    closed_days TEXT,
    ticket_info TEXT,
    weather_fit TEXT,
    suitable_for TEXT,
    source TEXT,
    fetched_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_attractions_city_district ON attractions(city, district);

CREATE TABLE IF NOT EXISTS restaurants (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    city TEXT,
    district TEXT,
    latitude REAL,
    longitude REAL,
    category TEXT,
    description TEXT,
    open_hours TEXT,
    source TEXT,
    fetched_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_restaurants_city_district ON restaurants(city, district);

CREATE TABLE IF NOT EXISTS lodgings (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    city TEXT,
    district TEXT,
    latitude REAL,
    longitude REAL,
    license_no TEXT,
    description TEXT,
    price_range TEXT,
    source TEXT,
    fetched_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_lodgings_city_district ON lodgings(city, district);

-- UNIQUE(tdx_id, type, city)：審查發現fetch_transit.py原本沒有唯一鍵，重跑
-- 腳本會整批重複插入（不像fetch_tourism.py用AttractionID當PK做upsert）。
-- 三欄一起當唯一鍵，不是只用tdx_id：TDX的parking資料集tdx_id不是全臺唯一，
-- 同一個小數字id會在不同縣市重複出現（實測案例：id=44分別是新竹市「停七停車場」
-- 跟宜蘭縣「蘇澳冷泉停車場」，兩個都是真實不同站點，只用tdx_id當唯一鍵會誤刪）。
CREATE TABLE IF NOT EXISTS transit_stations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_name TEXT,
    city TEXT,
    type TEXT,
    tdx_id TEXT,
    latitude REAL,
    longitude REAL,
    source TEXT,
    fetched_at TEXT,
    UNIQUE(tdx_id, type, city)
);
CREATE INDEX IF NOT EXISTS idx_transit_city_type ON transit_stations(city, type);

CREATE TABLE IF NOT EXISTS attraction_transit (
    entity_type TEXT,   -- 'attraction' or 'lodging'
    entity_id TEXT,
    station_id INTEGER,
    distance_m REAL
);
CREATE INDEX IF NOT EXISTS idx_at_entity ON attraction_transit(entity_type, entity_id);

CREATE TABLE IF NOT EXISTS lodging_nearest_attraction (
    lodging_id TEXT PRIMARY KEY,
    attraction_id TEXT,
    distance_m REAL
);

CREATE TABLE IF NOT EXISTS district_coords (
    city TEXT,
    district TEXT,
    latitude REAL,
    longitude REAL,
    sample_count INTEGER,
    PRIMARY KEY (city, district)
);
"""


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn
