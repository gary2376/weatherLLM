"""景點分類代碼表（觀光資料標準 V2.1，AttractionClassEnum，見 media.taiwan.net.tw 標準PDF 第130-131頁）。
weather_fit 分類是我方判斷（依代碼中文名稱字面意義），不是官方欄位，第一版跑完需要人工抽樣核對。
"""

CLASS_LABELS = {
    1: "文化類", 2: "生態類", 3: "文化資產類", 4: "宗教廟宇類", 5: "藝術類",
    6: "商圈商店類", 7: "國家公園類", 8: "國家風景區類", 9: "休閒農業類", 10: "溫泉類",
    11: "自然風景類", 12: "遊憩類", 13: "體育健身類", 14: "觀光工廠類", 15: "都會公園類",
    16: "森林遊樂區類", 17: "平地森林園區類", 18: "國家自然公園類", 19: "公園綠地類",
    20: "觀光遊樂業類", 21: "原住民文化類", 22: "客家文化類", 23: "交通場站類",
    24: "水域環境類", 25: "藝文場館類", 26: "生態場館類", 27: "娛樂場館類", 254: "其他",
}

# 室內/半室內、雨天照樣能去；10(溫泉)雨天濕冷天氣反而合適，歸這邊
INDOOR_OK = {1, 3, 4, 5, 6, 10, 14, 23, 25, 26, 27}
# 以戶外空間為主，下雨會影響體驗
OUTDOOR_SENSITIVE = {2, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22, 24}
# 20(觀光遊樂業) 室內外混合，不硬分，歸兩邊都不算，算 unknown 讓人工判斷


def classify_weather_fit(codes: list[int]) -> str:
    codes = set(codes or [])
    has_indoor = bool(codes & INDOOR_OK)
    has_outdoor = bool(codes & OUTDOOR_SENSITIVE)
    if has_indoor and has_outdoor:
        return "mixed"
    if has_indoor:
        return "indoor_ok"
    if has_outdoor:
        return "outdoor_only"
    return "unknown"


def labels_for(codes: list[int]) -> str:
    return "、".join(CLASS_LABELS.get(c, str(c)) for c in (codes or []))
