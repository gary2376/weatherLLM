"""幫每筆住宿算「同城市內最近的一個景點」，寫進 lodging_nearest_attraction。
補 Phase 0 schema 那個 `distance_to_nearest_attraction`——之前誤把這個跟 attraction_transit
（景點/住宿到最近交通站）搞混，那張表算的是完全不同的東西，這裡才是真的「住宿到最近景點」。
用法：python3 collector/link_lodging_attraction.py
"""
import math
from collections import defaultdict

from db import get_conn

GRID = 0.01  # ~1.1km 網格，同 link_transit.py 的做法


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
    conn.execute("DELETE FROM lodging_nearest_attraction")

    attractions = conn.execute(
        "SELECT id, city, latitude, longitude FROM attractions "
        "WHERE latitude IS NOT NULL AND longitude IS NOT NULL"
    ).fetchall()

    index = defaultdict(list)
    for aid, city, lat, lon in attractions:
        index[(city, cell(lat, lon))].append((aid, lat, lon))

    def nearest(city, lat, lon):
        cy, cx = cell(lat, lon)
        best = None
        # 先找同城市內最近的，網格範圍逐步擴大，直到找到為止（景點比交通站稀疏，1格內不一定有）
        for radius in range(1, 20):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if max(abs(dy), abs(dx)) != radius and radius != 1:
                        continue
                    for aid, alat, alon in index.get((city, (cy + dy, cx + dx)), []):
                        d = haversine_m(lat, lon, alat, alon)
                        if best is None or d < best[1]:
                            best = (aid, d)
            if best is not None:
                break
        return best

    n = 0
    rows = conn.execute(
        "SELECT id, city, latitude, longitude FROM lodgings "
        "WHERE latitude IS NOT NULL AND longitude IS NOT NULL"
    ).fetchall()
    for lid, city, lat, lon in rows:
        hit = nearest(city, lat, lon)
        if hit:
            aid, dist = hit
            conn.execute(
                "INSERT INTO lodging_nearest_attraction (lodging_id, attraction_id, distance_m) "
                "VALUES (?,?,?)",
                (lid, aid, round(dist, 1)),
            )
            n += 1
    conn.commit()
    print(f"lodging_nearest_attraction rows: {n} / {len(rows)} lodgings")
    conn.close()


if __name__ == "__main__":
    main()
