import openai
try:
    from openai import OpenAI
    HAS_OPENAI_NEW = True
except Exception:
    OpenAI = None
    HAS_OPENAI_NEW = False
import os
import sqlite3
import json
from datetime import datetime, timezone, timedelta
from typing import Optional
import time

import pandas as pd  # 讀取 Excel 用
import requests
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
import sys

# Standardized prefixes to avoid emoji/encoding issues on some consoles
ERROR_PREFIX = "ERROR:"
WARN_PREFIX = "WARN:"

def _make_safe(msg: str) -> str:
    """Return a console-safe representation of msg.

    If the current stdout encoding supports UTF-8, return the original message so
    human-readable Chinese/emoji are preserved. Otherwise, fallback to escaping
    non-ASCII characters (backslashreplace) to avoid encoding errors in environments
    using latin-1 or other limited encodings.
    """
    try:
        enc = None
        if hasattr(sys, 'stdout') and getattr(sys, 'stdout') is not None:
            enc = getattr(sys.stdout, 'encoding', None)
        if not enc:
            enc = os.environ.get('PYTHONIOENCODING')
        # If the current encoding can encode the message, return it unchanged
        if enc:
            try:
                msg.encode(enc)
                return msg
            except Exception:
                pass
        # fallback: escape non-ascii into \uXXXX so it is safe for limited consoles
        return msg.encode('ascii', 'backslashreplace').decode('ascii')
    except Exception:
        # last resort: explicit \uXXXX replacement
        return ''.join(c if ord(c) < 128 else f'\\u{ord(c):04x}' for c in msg)

# 請勿將 OpenAI API Key 硬編碼在程式碼中。
# 金鑰應由執行環境或呼叫端設定，例如：
#  - 使用環境變數 OPENAI_API_KEY
#  - 或由呼叫的應用程式（例如 streamlit）動態設定 system.openai.api_key
env_openai_key = os.getenv("OPENAI_API_KEY")
if env_openai_key:
    openai.api_key = env_openai_key

# 是否使用本機資料（DB / radar XML / Excel）。預設使用本機資料，但在要上傳到 GitHub
# 或執行於不能依賴本機檔案的環境時，可設定環境變數 USE_LOCAL_DATA=0
USE_LOCAL_DATA = os.getenv("USE_LOCAL_DATA", "1") != "0"

# SQLite 資料庫路徑
DATABASE_NAME = r"E:\python_project\contest\TGIS\DB\taichung_weather.db"

# 自訂區域 ID ↔ 區域名稱 對應表
CUSTOM_ID_TO_NAME_MAP = {
    1: "中區", 2: "北區", 3: "北屯區", 4: "南區", 5: "南屯區",
    6: "后里區", 7: "和平區", 8: "外埔區", 9: "大安區", 10: "大甲區",
    11: "大肚區", 12: "大里區", 13: "大雅區", 14: "太平區", 15: "新社區",
    16: "東勢區", 17: "東區", 18: "梧棲區", 19: "沙鹿區", 20: "清水區",
    21: "潭子區", 22: "烏日區", 23: "石岡區", 24: "神岡區", 25: "西區",
    26: "西屯區", 27: "豐原區", 28: "霧峰區", 29: "龍井區"
}
NAME_TO_CUSTOM_ID_MAP = {v: k for k, v in CUSTOM_ID_TO_NAME_MAP.items()}

# 可擴充的行政區 -> 經緯度映射（若無更精準座標，會使用台中市中心作為預設）
# 注意：部分座標為暫時預設（使用台中市中心），若需要更精準位置建議以官方或地理編碼資料更新。
TAICHUNG_CENTER = (24.1477, 120.6736)
AREA_COORDS: dict[str, tuple[float, float]] = {
    "台中市": TAICHUNG_CENTER,
    "臺中": TAICHUNG_CENTER,
    # 市中心或主要行政區（若有更精準經緯度，可替換下列值）
    "中區": (24.14383, 120.67951),
    "北區": (24.166039, 120.682318),
    "西區": (24.14138, 120.67104),
    "東區": (24.136625, 120.703854),
    "南區": (24.117079, 120.663608),
    "北屯區": (24.182264, 120.686288),
    "西屯區": (24.165303, 120.633655),
    "南屯區": (24.134631, 120.644374),
    "太平區": (24.126472, 120.718523),
    "大里區": (24.099417, 120.67786),
    "霧峰區": (24.061698, 120.700272),
    "豐原區": (24.24219, 120.71846),
    "潭子區": (24.20953, 120.70516),
    "大雅區": (24.229141, 120.64778),
    "大甲區": (24.34892, 120.62239),
    "外埔區": (24.33201, 120.65437),
    "清水區": (24.268576, 120.559767),
    "梧棲區": (24.254924, 120.531626),
    "龍井區": (24.192679, 120.545838),
    "烏日區": (24.104696, 120.623806),
    "大肚區": (24.151083, 120.545439),
    "石岡區": (24.27498, 120.78041),
    "后里區": (24.30491, 120.71071),
    "新社區": (24.23414, 120.8095),
    "東勢區": (24.25861, 120.82777),
    "和平區": (24.17477, 120.88349),
    "神岡區": (24.257826, 120.661511),
    "沙鹿區": (24.233445, 120.566218),
    "大安區": (24.34607, 120.58652),
    # 若有需要，可在此處加入更多別名或替代拼法
}

# ========= 一啟動程式就把 attractions.xlsx 讀進來 =========
# 使用整合後的景點檔案（與 Streamlit app 相同）
EXCEL_PATH = r"E:\python_project\contest\TGIS\Data\location_consolidated_enhanced.xlsx"
try:
    # sheet_name=None 會把每個 sheet 讀成 { "西屯區": DataFrame, "北區": DataFrame, ... }
    ATTRACTIONS_SHEETS: dict[str, pd.DataFrame] = pd.read_excel(
        EXCEL_PATH, sheet_name=None, engine="openpyxl"
    )
except Exception as e:
    # 讀不到 Excel 不應該阻斷整個應用，改為空資料並在日誌中提醒
    print(_make_safe(f"{WARN_PREFIX} 無法讀取 Excel 檔 {EXCEL_PATH}：{e} - 將使用空的景點/住宿清單"))
    ATTRACTIONS_SHEETS = {}
# ======================================================

def analyze_prompt_with_llm(user_prompt: str) -> dict:
    """
    呼叫 LLM 把使用者問題解析成 JSON，包含：
      - type: 'radar' 或 'forecast'
      - area: 行政區名稱 (e.g. '西屯區')
      - date: YYYY-MM-DD (若 type == 'forecast')
      - hour: HH:MM (可選，否則預設用 06:00 或 18:00)
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    system_prompt = (
        f"今天是 {today_str}。\n"
        "你是一個天氣查詢理解助手，請從使用者的問題中抽取出以下資訊並以 JSON 回傳：\n\n"
        "- type: 'radar' 或 'forecast'\n"
        "  - 查詢『現在、即時、等一下、目前、這時候』等屬於 radar\n"
        "  - 查詢『明天、後天、週末、晚上、早上、未來幾天、6月3日』等屬於 forecast\n\n"
        "- area: 台中市的行政區名稱，如 '西屯區', '北區' 等；若提及景點，請推論該景點所屬的行政區。\n"
        "- date: 如果 type 是 forecast，請回傳查詢目標日期（格式：YYYY-MM-DD）\n"
        "- hour: 如有提供時間，回傳格式為 'HH:MM'，否則可省略\n\n"
    "僅回傳 JSON 結果，請勿加入解釋或多餘文字。"
    )

    # 兼容舊版與新版 openai 套件，統一呼叫 helper
    def _create_chat_completion(model, messages, **kwargs):
        if HAS_OPENAI_NEW and OpenAI is not None:
            client = OpenAI()
            return client.chat.completions.create(model=model, messages=messages, **kwargs)
        else:
            # fallback to older openai package interface
            return openai.ChatCompletion.create(model=model, messages=messages, **kwargs)

    def _extract_content(resp):
        # 嘗試多種可用的位置來取得回傳內容
        try:
            return resp.choices[0].message.content
        except Exception:
            try:
                return resp.choices[0].message['content']
            except Exception:
                try:
                    return resp.choices[0].text
                except Exception:
                    raise RuntimeError('無法從 LLM 回傳中擷取內容')

    response = _create_chat_completion(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=150,
        temperature=0.2
    )
    content = _extract_content(response)
    # 如果回傳有 code fence，把 ``` 或 ```json 去掉
    if content.startswith("```"):
        content = content.strip("`").strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()
    return json.loads(content)


def get_attractions_for_area(area: str, top_n: int = 5) -> list[str]:
    """
    回傳該區域 (area) 前 top_n 名 Type == "attraction" 的景點名稱清單。
    如果找不到該區 sheet 或沒有 Type/Name 欄位，回傳空 list。
    """
    # ATTRACTIONS_SHEETS may be a dict of sheets (one-per-district) or a single consolidated sheet.
    df: Optional[pd.DataFrame] = ATTRACTIONS_SHEETS.get(area)
    if df is None:
        # If there's a single consolidated sheet (e.g., 'Sheet1'), use it and filter by source_district
        if isinstance(ATTRACTIONS_SHEETS, dict) and len(ATTRACTIONS_SHEETS) == 1:
            df = list(ATTRACTIONS_SHEETS.values())[0]
        else:
            return []

    if "Type" not in df.columns or "Name" not in df.columns:
        return []

    # 如果 DataFrame 有 source_district 欄位，優先以包含比對 (contains) 來篩選該區域
    df_work = df.copy()
    if 'source_district' in df_work.columns and isinstance(area, str):
        norm_area = area.replace('台', '臺')
        # match either '沙鹿' or '沙鹿區' etc.
        df_work = df_work[df_work['source_district'].astype(str).str.contains(norm_area, na=False, case=False)]

    # 篩出 Type = "attraction"
    df_attraction = df_work[df_work["Type"].astype(str).str.lower() == "attraction"].copy()
    if df_attraction.empty:
        return []

    # 如果有 Rating 欄，就依 Rating 由高到低排序
    if "Rating" in df_attraction.columns:
        try:
            df_attraction = df_attraction.sort_values("Rating", ascending=False)
        except Exception:
            pass

    # 回傳前 top_n 個 Name
    return df_attraction["Name"].head(top_n).astype(str).tolist()


def get_lodgings_for_area(area: str, top_n: int = 3) -> list[str]:
    """
    回傳該區域 (area) 前 top_n 名 Type == "hotel" 的住宿名稱清單。
    如果找不到該區 sheet 或沒有 Type/Name 欄位，回傳空 list。
    """
    df: Optional[pd.DataFrame] = ATTRACTIONS_SHEETS.get(area)
    if df is None:
        return []

    if "Type" not in df.columns or "Name" not in df.columns:
        return []

    # 篩出 Type = "hotel"
    df_hotel = df[df["Type"].astype(str).str.lower() == "hotel"].copy()
    if df_hotel.empty:
        return []

    # 如果有 Rating 欄，就依 Rating 由高到低排序
    if "Rating" in df_hotel.columns:
        try:
            df_hotel = df_hotel.sort_values("Rating", ascending=False)
        except Exception:
            pass

    # 回傳前 top_n 個 Name
    return df_hotel["Name"].head(top_n).astype(str).tolist()


def generate_answer_from_user_prompt_and_data(user_prompt: str, raw_data: str) -> str:
    """
    呼叫 LLM，把「使用者提問」以及拼好的 raw_data (包含天氣、景點、住宿)
    一併傳給 LLM，讓它回覆一段口語化、自然的建議。
    """
    system_prompt = (
        "你是一個整合「天氣＋旅遊／住宿」的建議助手，"
        "根據使用者的問題以及下面提供的「天氣資料」和「景點清單」以及「住宿清單」，"
        "請用繁體中文產生一段自然、口語化、容易理解的回答。\n\n"
        "請同時做到：\n"
        "1. 天氣部分：重點放在『體感描述』與『建議』（例如：是否要帶傘、穿薄外套／短袖、注意防曬／防雨等）。\n"
        "2. 旅遊景點部分：從「景點清單」中，挑幾個適合當天的戶外／室內景點，並說明為何適合（例如：今天太熱，就推薦室內或有遮陽的景點）。\n"
        "3. 住宿部分：推薦「住宿清單」中的幾個優質住宿，並簡單說明為何適合（例如：若是帶小孩，這間民宿環境適合；若想看夜景，這家飯店地點便利）。\n"
        "4. 整體風格要自然、像朋友聊天，不要只羅列列表或純數據。\n"
    )

    user_prompt_combined = (
        f"使用者提問：{user_prompt}\n"
        f"查詢結果如下：\n{raw_data}\n"
        "請根據上述內容，生成一段自然語言的回答。"
    )

    # 使用與 analyze_prompt_with_llm 相同的兼容 helper
    def _create_chat_completion(model, messages, **kwargs):
        if HAS_OPENAI_NEW and OpenAI is not None:
            client = OpenAI()
            return client.chat.completions.create(model=model, messages=messages, **kwargs)
        else:
            return openai.ChatCompletion.create(model=model, messages=messages, **kwargs)

    def _extract_content(resp):
        try:
            return resp.choices[0].message.content
        except Exception:
            try:
                return resp.choices[0].message['content']
            except Exception:
                try:
                    return resp.choices[0].text
                except Exception:
                    raise RuntimeError('無法從 LLM 回傳中擷取內容')

    try:
        response = _create_chat_completion(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt_combined}
            ],
            temperature=0.5,
            max_tokens=1000
        )
        generated_summary = _extract_content(response).strip()
    except Exception as e:
        # 若 LLM 呼叫失敗（無 API key、網路或認證錯誤），回傳友善且 ASCII-safe 的錯誤訊息
        return _make_safe(f"{ERROR_PREFIX} LLM 呼叫失敗：{e}")

    # 清理：若 LLM 回傳的是 JSON 或陣列（例如意外回傳 [] / [null, ...]），嘗試將其轉成可讀文字
    try:
        # 移除 code fence
        if generated_summary.startswith("```"):
            generated_summary = generated_summary.strip("`").strip()
        parsed = None
        try:
            parsed = json.loads(generated_summary)
        except Exception:
            parsed = None

        if parsed is not None:
            def _extract_text_from_json(obj):
                if obj is None:
                    return ""
                if isinstance(obj, str):
                    return obj
                if isinstance(obj, list):
                    parts = [_extract_text_from_json(i) for i in obj]
                    return "\n".join([p for p in parts if p])
                if isinstance(obj, dict):
                    parts = [_extract_text_from_json(v) for v in obj.values()]
                    return "\n".join([p for p in parts if p])
                return str(obj)

            cleaned = _extract_text_from_json(parsed).strip()
            if cleaned:
                generated_summary = cleaned
            else:
                # 無法從 JSON 中抽出可讀文字，改回友善的錯誤提示
                generated_summary = "（OpenAI 回傳的內容為機器格式／空值，請稍候重試或更換查詢）"
    except Exception:
        # 若清理流程有例外，仍回傳原始文本，避免吞掉可能的有效回覆
        pass

    # 最後一層防護：如果回傳是空字串或仍看起來像機器格式（例如只包含 [] 或大量 null），
    # 則直接回退為友善訊息，避免把機器格式傳回到前端
    try:
        def _is_machine_like(s: str) -> bool:
            if s is None:
                return True
            if not isinstance(s, str):
                return True
            ss = s.strip()
            if not ss:
                return True
            low = ss.lower()
            # 明顯純 JSON 結構但內容沒有可讀文字
            if low in ('[]', '[ ]', '{}'):
                return True
            try:
                parsed = json.loads(ss)
                # 如果解析後是 list/dict，檢查是否可以抽出任何非空字串
                def has_useful(o):
                    if o is None:
                        return False
                    if isinstance(o, str):
                        return bool(o.strip())
                    if isinstance(o, (list, tuple)):
                        return any(has_useful(i) for i in o)
                    if isinstance(o, dict):
                        return any(has_useful(v) for v in o.values())
                    return True
                if isinstance(parsed, (list, dict)) and not has_useful(parsed):
                    return True
            except Exception:
                # 若不能解析但包含多個 null/none，視為機器格式
                if low.count('null') >= 1 or low.count('none') >= 1:
                    # 若字串很短且大部分是 null/brackets，視為機器格式
                    cleaned = low.replace('null', '').replace('none', '').replace('[', '').replace(']', '').replace('{', '').replace('}', '').strip()
                    if len(cleaned) < 10:
                        return True
            return False

        if _is_machine_like(generated_summary):
            generated_summary = "（OpenAI 回傳的內容無法解析為可讀文字，請稍候重試或調整查詢。）"
    except Exception:
        # 若檢查失敗，保留原始回覆
        pass

    return generated_summary


def fetch_radar_data(district_id: int, limit: int = 5) -> str:
    """
    查詢指定 district_id 的即時雷達回波資料，取最新 limit 筆，並回傳成文字。
    """
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT timestamp_utc, dbz_value
            FROM realtime_observations
            WHERE custom_district_id = ?
            ORDER BY timestamp_utc DESC
            LIMIT ?
            """,
            (district_id, limit)
        )
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            # 若本機 DB 中無即時雷達資料，改為使用遠端降水量作為替代 (Open-Meteo)
            area_name = CUSTOM_ID_TO_NAME_MAP.get(district_id)
            return fetch_precipitation_open_meteo(area=area_name)

        result = f"📡 雷達回波 - {CUSTOM_ID_TO_NAME_MAP[district_id]}：\n"
        for t, dbz in rows:
            result += f"  - 時間: {t}, dBZ 值: {dbz if dbz is not None else 'N/A'}\n"
        return result

    except Exception as e:
        return _make_safe(f"{ERROR_PREFIX} 雷達資料查詢錯誤：{e}")


def get_latest_radar_summary(radar_dir: str = r"E:\python_project\contest\TGIS\radar") -> str:
    """
    讀取本地 radar/ 目錄，找最新的 XML 檔，回傳簡短摘要（檔名、修改時間、若能解析則回傳根節點資訊）。
    此函式不依賴遠端 API，適合作為本機 RAG 的一部分。
    """
    # 如果環境設定關閉本機資料，或 radar 目錄不存在／找不到檔案，改用遠端即時降水資料作為替代
    try:
        p = Path(radar_dir)
        if p.exists() and USE_LOCAL_DATA:
            xml_files = sorted(p.glob('*.xml'), key=lambda x: x.stat().st_mtime, reverse=True)
            if xml_files:
                newest = xml_files[0]
                mtime = datetime.fromtimestamp(newest.stat().st_mtime)
                summary = f"Radar 檔案: {newest.name} (最後修改: {mtime.strftime('%Y-%m-%d %H:%M:%S')})"
                # 嘗試解析 XML 並抓出少量可讀資訊（如果可能）
                try:
                    tree = ET.parse(newest)
                    root = tree.getroot()
                    attrs = []
                    for k, v in list(root.attrib.items())[:5]:
                        attrs.append(f"{k}={v}")
                    if attrs:
                        summary += " | root_attrs: " + ",".join(attrs)
                except Exception:
                    pass
                return summary

        # fallback: 以 Open-Meteo 的降雨量資料作為代替（不需要 API key）
        return fetch_precipitation_open_meteo()
    except Exception as e:
        return f"（讀取 radar/遠端降水摘要時發生錯誤：{e}）"


def fetch_precipitation_open_meteo(area: Optional[str] = None, lat: Optional[float] = None, lon: Optional[float] = None) -> str:
    """
    使用 Open-Meteo 取得最近數小時的降雨量（作為 radar 的替代），若有 area 可用 AREA_COORDS
    回傳簡短文字摘要。
    """
    try:
        if area and area in AREA_COORDS:
            lat, lon = AREA_COORDS[area]
        if lat is None or lon is None:
            lat, lon = AREA_COORDS.get("台中市", (24.1477, 120.6736))

        # 取得最近 6 小時的 hourly precipitation
        url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&hourly=precipitation&timezone=Asia/Taipei"
        )
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        prec = hourly.get("precipitation", [])
        if not times or not prec:
            return "（無法從 Open-Meteo 取得降水資訊）"

        # 取最後 6 筆（或少於 6 筆）作摘要
        last_n = min(6, len(times))
        summary = "Open-Meteo 降雨量摘要：\n"
        for t, pval in zip(times[-last_n:], prec[-last_n:]):
            summary += f"  - {t}: 降水量 {pval} mm\n"
        return summary
    except Exception as e:
        return f"（取得遠端降水資料時發生錯誤：{e}）"


def fetch_current_weather_open_meteo(lat: float = 24.1477, lon: float = 120.6736, area: Optional[str] = None) -> str:
    """
    使用 Open-Meteo 公共 API 取得當前天氣（free, 無需 API key）。
    回傳簡短文字摘要，便於作為 RAG 的上下文輸入給 LLM。
    """
    try:
        # 如果有提供區域名稱且在 AREA_COORDS 中，使用其座標
        if area and area in AREA_COORDS:
            lat, lon = AREA_COORDS[area]
        url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&current_weather=true&timezone=Asia/Taipei"
        )
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        cw = data.get('current_weather') or {}
        if not cw:
            return "（無法取得 Open-Meteo 的即時天氣）"
        t = cw.get('temperature')
        ws = cw.get('windspeed')
        wc = cw.get('weathercode')
        time_str = cw.get('time')
        summary = f"即時天氣 (Open-Meteo) - 時間: {time_str}, 氣溫: {t}°C, 風速: {ws} m/s, weathercode: {wc}"
        return summary
    except Exception as e:
        return f"（取得即時天氣時發生錯誤：{e}）"


def fetch_forecast_data(
    district_id: int,
    target_date: datetime,
    limit: int = 1,
    target_hour: Optional[str] = None
) -> str:
    """
    查詢指定 district_id 在 target_date (YYYY-MM-DD) 的天氣預報，
    若提供 target_hour (如 "06:00" 或 "18:00")，再依小時過濾，最後回傳文字結果。
    """
    try:
        # 把 target_date 轉成兩個 UTC 邊界（00:00 到隔天 00:00）
        start_utc = datetime(
            target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc
        )
        end_utc = start_utc + timedelta(days=1)

        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()

        sql = """
            SELECT *
            FROM weekly_forecasts
            WHERE custom_district_id = ?
              AND forecast_period_start_utc >= ?
              AND forecast_period_start_utc < ?
        """
        params = [
            district_id,
            start_utc.strftime('%Y-%m-%d %H:%M:%S'),
            end_utc.strftime('%Y-%m-%d %H:%M:%S')
        ]

        if target_hour:
            sql += " AND strftime('%H:%M', forecast_period_start_utc) = ?"
            params.append(target_hour)

        sql += " ORDER BY forecast_period_start_utc ASC LIMIT ?"
        params.append(limit)

        cursor.execute(sql, tuple(params))
        rows = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        conn.close()

        if not rows:
            # 若本機 DB 中沒有該區的預報資料，直接使用 Open-Meteo 作為後備（不管 DB 檔案是否存在）
            try:
                area_name = CUSTOM_ID_TO_NAME_MAP.get(district_id)
                lat, lon = AREA_COORDS.get(area_name, AREA_COORDS.get("台中市"))
                return fetch_forecast_open_meteo(lat, lon, target_date, target_hour)
            except Exception:
                return _make_safe(
                    f"{WARN_PREFIX} 沒有查到 {CUSTOM_ID_TO_NAME_MAP[district_id]} {target_hour or ''} 的 "
                    f"{target_date.strftime('%m月%d日')} 預報資料"
                )

        result = f"🌤️ {CUSTOM_ID_TO_NAME_MAP[district_id]} {target_hour or ''} 天氣預報（{target_date.strftime('%m月%d日')}）：\n"
        for row in rows:
            result += "=============================\n"
            for col, val in zip(column_names, row):
                result += f"{col}: {val}\n"
        return result

    except Exception as e:
        return _make_safe(f"{ERROR_PREFIX} 預報資料查詢錯誤：{e}")


def fetch_forecast_open_meteo(lat: float, lon: float, target_date: datetime, target_hour: Optional[str] = None) -> str:
    """
    使用 Open-Meteo 取得指定日期 (target_date) 與時段 (target_hour) 的預報，回傳簡短文字。
    - lat, lon: 座標
    - target_date: datetime with the date to query (UTC tz aware expected)
    - target_hour: "06:00" or "18:00" 樣式，若 None 則回傳該日 summary
    """
    try:
        # Open-Meteo 接口: 取得當日 hourly 與 daily 資料
        date_str = target_date.strftime("%Y-%m-%d")
        url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&hourly=temperature_2m,precipitation,weathercode&windspeed_unit=ms&timezone=Asia/Taipei"
            f"&start_date={date_str}&end_date={date_str}"
        )
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        hourly = data.get('hourly', {})
        times = hourly.get('time', [])
        temps = hourly.get('temperature_2m', [])
        prec = hourly.get('precipitation', [])
        codes = hourly.get('weathercode', [])

        if not times:
            return f"（Open-Meteo 未回傳 {date_str} 的預報資料）"

        result = f"🌤️ Open-Meteo 天氣預報（{date_str}）:\n"
        if target_hour:
            # 嘗試匹配目標小時
            # target_hour 例如 '06:00'，我們比對時分
            matched = False
            for t, tmp, p, c in zip(times, temps, prec, codes):
                if t.endswith(target_hour):
                    result += f"  - {t}: 氣溫 {tmp}°C, 降水 {p} mm, weathercode {c}\n"
                    matched = True
                    break
            if not matched:
                # 找不到精確小時時，取該日早上/晚間代表值
                result += "  （找不到精確時段資料，請改以全天摘要為主）\n"
        # 提供簡短全天摘要（取最大/最小/總降水）
        try:
            temps_f = [float(x) for x in temps]
            prec_f = [float(x) for x in prec]
            result += f"  當日最高 {max(temps_f):.1f}°C, 最低 {min(temps_f):.1f}°C, 總降水 {sum(prec_f):.1f} mm\n"
        except Exception:
            pass
        return result
    except Exception as e:
        return f"（使用 Open-Meteo 取得預報時發生錯誤：{e}）"


def get_weather_data(user_prompt: str, area_override: Optional[str] = None) -> str:
    """
    主要入口：根據使用者提問，執行以下步驟：
    1. analyze_prompt_with_llm 解析出 type, area, date, hour
    2. 依 type 決定呼叫 fetch_radar_data 或 fetch_forecast_data
    3. 如果查到天氣資料，繼續抓該區域的景點與住宿清單
    4. 把「天氣＋景點＋住宿」組成 raw_data，呼叫 generate_answer_from_user_prompt_and_data
    5. 回傳 LLM 的最終建議 (不包含 system.py 自己的 CCTV 列表)
    """
    try:
        info = analyze_prompt_with_llm(user_prompt)
    except Exception as e: # 更通用的 Exception 捕捉
        return _make_safe(f"{ERROR_PREFIX} Prompt 分析階段出錯：{e}") # 返回錯誤信息

    query_type = info.get("type")
    area = info.get("area") # area 是中文區域名稱, e.g., "西屯區"
    # 如果呼叫端提供覆寫的行政區，使用它
    if area_override:
        area = area_override

    if not area:
        return _make_safe(f"{ERROR_PREFIX} 無法辨識您想查詢的區域。") # 更友好的提示

    district_id = NAME_TO_CUSTOM_ID_MAP.get(area)
    if not district_id:
        return _make_safe(f"{ERROR_PREFIX} 系統中找不到區域『{area}』的對應 ID，請確認是否為台中市的行政區。") # 更友好的提示

    raw_data = "" # 初始化 raw_data
    if query_type == "radar":
        raw_data = fetch_radar_data(district_id)
    elif query_type == "forecast":
        date_str = info.get("date")
        hour = info.get("hour", "06:00") # LLM 給的 hour

        if not date_str:
            return _make_safe(f"{ERROR_PREFIX} 預報查詢需要有效日期。") # 更友好的提示

        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return _make_safe(f"{ERROR_PREFIX} 日期格式錯誤 ({date_str})，應為 YYYY-MM-DD。") # 更友好的提示

        # 如果 LLM 回傳的 hour 不是 06:00 或 18:00，做合理的時段轉換
        if hour and isinstance(hour, str) and hour not in ["06:00", "18:00"]: # 檢查 hour 是否有效
            try:
                hour_parts = hour.split(":")
                hour_int = int(hour_parts[0])
                if 0 <= hour_int <= 2:
                    target_date = target_date - timedelta(days=1)
                    hour = "18:00"
                elif 3 <= hour_int < 17:
                    hour = "06:00"
                else:
                    hour = "18:00"
            except Exception: # 若轉換失敗，使用預設
                hour = "06:00"
        elif not hour: # 如果 hour 是 None 或空字串
             hour = "06:00"


        raw_data = fetch_forecast_data(
            district_id,
            target_date=target_date,
            target_hour=hour
        )
    else:
        return _make_safe(f"{ERROR_PREFIX} 無法辨識查詢類型 (應為 'radar' 或 'forecast')。LLM 回應 type: {query_type or '未提供'}") # 更友好的提示

    # 3. 如果 fetch 天氣時有錯誤，直接回傳
    if raw_data.startswith((WARN_PREFIX, ERROR_PREFIX)):
        return raw_data

    # 3.5 取得即時外部資料 (RAG)：local radar summary 與 Open-Meteo 即時天氣
    # 這些資料會併入 raw_data 作為 LLM 的上下文，提升回覆的即時性
    try:
        radar_summary = get_latest_radar_summary()
        raw_data = f"[RAG - 本機雷達摘要]\n{radar_summary}\n\n" + raw_data
    except Exception:
        pass
    try:
        # 優先以 area 取得即時天氣座標（若 area 在 AREA_COORDS 中則使用對應經緯度）
        current_weather_summary = fetch_current_weather_open_meteo(area=area)
        raw_data = f"[RAG - 即時天氣]\n{current_weather_summary}\n\n" + raw_data
    except Exception:
        pass

    # 4. 抓取景點與住宿清單，並拼接到 raw_data
    attractions_list = get_attractions_for_area(area, top_n=5)
    if attractions_list:
        raw_data += "\n\n🏞️ 此區推薦景點：\n"
        for idx, name in enumerate(attractions_list, start=1):
            raw_data += f"  {idx}. {name}\n" # 修正縮排
    else:
        raw_data += f"\n\n🏞️ 抱歉，目前沒有「{area}」的景點資料可供推薦。\n"

    lodgings_list = get_lodgings_for_area(area, top_n=3)
    if lodgings_list:
        raw_data += "\n\n🏨 此區推薦住宿：\n"
        for idx, name in enumerate(lodgings_list, start=1):
            raw_data += f"  {idx}. {name}\n" # 修正縮排
    else:
        raw_data += f"\n\n🏨 抱歉，目前沒有「{area}」的住宿資料可供推薦。\n"

    # 5. 一併把「天氣 + 景點 + 住宿」送給 LLM 生成最終回覆
    start_time = time.time()
    generated_summary = generate_answer_from_user_prompt_and_data(user_prompt, raw_data)
    elapsed = time.time() - start_time
    
    # 這裡只回傳 LLM 的建議，不包含 system.py 自身的 CCTV 列表
    llm_response_with_header = f"回應建議（回應耗時 {elapsed:.2f} 秒）：\n{generated_summary}"
    # 返回給呼叫端時，使用 _make_safe 以避免在某些終端或日誌系統發生編碼錯誤
    return _make_safe(llm_response_with_header)

if __name__ == "__main__":
    print(_make_safe("台中天氣小幫手，輸入你想查詢的內容吧（輸入 exit 離開）\n"))
    while True:
        user_input = input("👉 請輸入問題：")
        if user_input.lower().strip() == "exit":
            print("👋 再見！")
            break
        print(get_weather_data(user_input))
        print("-" * 60)
