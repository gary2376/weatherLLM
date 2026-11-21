"""從已收集的景點/住宿/餐飲三張表，算出每個「縣市+行政區」的平均座標，
當作天氣查詢用的行政區中心點——取代 system.py 裡手打的台中30筆 dict。

為什麼不抓 TGOS 鄉鎮市區界線圖資：那個是 shapefile 多邊形，要算中心點得另外裝
shapely/pyshp 處理幾何，我們手上已經有全台上萬筆帶座標的景點/住宿/餐飲資料，
直接拿來平均算中心點，不用多引入一個幾何函式庫依賴。代價：極少數完全沒有
景點/住宿/餐飲資料的偏遠鄉鎮不會有中心點，查不到時交給呼叫端 fallback 到
縣市層級平均（在 system.py 做，不在這裡）。

用法：python3 collector/derive_district_coords.py
"""
from db import get_conn


def main():
    # district_coords 的建表語句在 db.py 的中央 SCHEMA 裡（get_conn() 會確保它存在），
    # 這裡不用再自己重複建一次——之前這張表的 CREATE TABLE 只寫在這支腳本裡，沒放進
    # 中央schema，若有人重建資料庫時漏跑這支腳本，表根本不存在，system.py 查詢會噴
    # `no such table` 而不是優雅退回預設座標。
    conn = get_conn()
    conn.execute("DELETE FROM district_coords")

    conn.execute("""
        INSERT INTO district_coords (city, district, latitude, longitude, sample_count)
        SELECT city, district, AVG(latitude), AVG(longitude), COUNT(*)
        FROM (
            SELECT city, district, latitude, longitude FROM attractions
            WHERE city IS NOT NULL AND district IS NOT NULL AND latitude IS NOT NULL
            UNION ALL
            SELECT city, district, latitude, longitude FROM lodgings
            WHERE city IS NOT NULL AND district IS NOT NULL AND latitude IS NOT NULL
            UNION ALL
            SELECT city, district, latitude, longitude FROM restaurants
            WHERE city IS NOT NULL AND district IS NOT NULL AND latitude IS NOT NULL
        )
        GROUP BY city, district
    """)
    conn.commit()

    n = conn.execute("SELECT COUNT(*) FROM district_coords").fetchone()[0]
    thin = conn.execute("SELECT city, district, sample_count FROM district_coords WHERE sample_count < 3").fetchall()
    print(f"district_coords rows: {n}")
    print(f"districts with fewer than 3 samples (centroid less reliable): {len(thin)}")
    for r in thin[:15]:
        print("  ", r)
    conn.close()


if __name__ == "__main__":
    main()
