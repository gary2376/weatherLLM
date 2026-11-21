"""幫每個景點/住宿算「同城市內、各類型(bus/bike/parking)最近的一個站」，寫進 attraction_transit，
超過 CUTOFF_M（預設1500公尺）不算有效關聯。跑完後把沒被任何景點/住宿引用到的 transit_stations
刪掉，讓最終 DB 只留有用的站，不留全台十幾萬筆不相關站牌。
用法：python3 collector/link_transit.py
"""
import math
from collections import defaultdict

from db import get_conn

CUTOFF_M = 1500
GRID = 0.01  # ~1.1km 網格，只比對同格+鄰近8格的站，不用整表比對


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def cell(lat, lon):
    return (round(lat / GRID), round(lon / GRID))


def main():
    conn = get_conn()
    conn.execute("DELETE FROM attraction_transit")

    stations = conn.execute(
        "SELECT id, city, type, latitude, longitude FROM transit_stations "
        "WHERE latitude IS NOT NULL AND longitude IS NOT NULL"
    ).fetchall()

    # index: (city, type, cell) -> list[(id, lat, lon)]
    index = defaultdict(list)
    for sid, city, stype, lat, lon in stations:
        index[(city, stype, cell(lat, lon))].append((sid, lat, lon))

    def nearest(city, stype, lat, lon):
        cy, cx = cell(lat, lon)
        best = None
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for sid, slat, slon in index.get((city, stype, (cy + dy, cx + dx)), []):
                    d = haversine_m(lat, lon, slat, slon)
                    if d <= CUTOFF_M and (best is None or d < best[1]):
                        best = (sid, d)
        return best

    types = ["bus", "bike", "parking"]
    n_links = 0
    for table, entity_type in [("attractions", "attraction"), ("lodgings", "lodging")]:
        rows = conn.execute(
            f"SELECT id, city, latitude, longitude FROM {table} "
            f"WHERE latitude IS NOT NULL AND longitude IS NOT NULL"
        ).fetchall()
        for eid, city, lat, lon in rows:
            for stype in types:
                hit = nearest(city, stype, lat, lon)
                if hit:
                    sid, dist = hit
                    conn.execute(
                        "INSERT INTO attraction_transit (entity_type, entity_id, station_id, distance_m) "
                        "VALUES (?,?,?,?)",
                        (entity_type, eid, sid, round(dist, 1)),
                    )
                    n_links += 1
    conn.commit()

    total_stations = conn.execute("SELECT COUNT(*) FROM transit_stations").fetchone()[0]
    if n_links == 0 or total_stations == 0:
        print(f"abort: n_links={n_links}, total_stations={total_stations} — 看起來 join 沒對上，"
              f"不執行刪除，先檢查 city 格式是否一致再重跑")
        conn.close()
        return

    conn.execute(
        "DELETE FROM transit_stations WHERE id NOT IN (SELECT DISTINCT station_id FROM attraction_transit)"
    )
    conn.commit()
    remaining = conn.execute("SELECT COUNT(*) FROM transit_stations").fetchone()[0]
    print(f"attraction_transit rows: {n_links}")
    print(f"transit_stations remaining after prune: {remaining}")
    conn.close()


if __name__ == "__main__":
    main()
