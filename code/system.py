import os
from pathlib import Path
from dotenv import load_dotenv

# .env 沒被載入過的話 os.getenv(...) 永遠讀不到值——collector/ 底下每支腳本都有呼叫
# 這個，但 system.py 之前沒有，等於 .env 這條路徑對主程式從沒真的生效過，只有靠
# Streamlit 側欄手動貼 key 才動得了。load_dotenv() 放在檔案最前面，比 openai/Langfuse
# 的 client 初始化更早跑，兩邊才都讀得到 .env 裡的 key。明確指定路徑（跟專案裡
# `DB_PATH`/`CHROMA_PATH` 同樣用 `Path(__file__)` 定位），不依賴呼叫端的當前工作
# 目錄——不然從專案外的目錄執行時，預設的 CWD-relative 搜尋會找不到 .env。
# 找不到 .env 檔不會報錯，已經是環境變數（例如部署環境用 st.secrets 橋接過來）的話
# 這行也不會覆蓋掉。
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

import openai
try:
    from openai import OpenAI
    HAS_OPENAI_NEW = True
except Exception:
    OpenAI = None
    HAS_OPENAI_NEW = False

# Langfuse 版的 OpenAI client 是完全相容的 drop-in 替換：換了 import 就自動幫每次
# chat.completions.create() 呼叫記錄 prompt/response/latency/token，不用手動包一層
# span。沒設定 LANGFUSE_PUBLIC_KEY/SECRET_KEY 或套件沒裝時優雅退回原生
# openai.OpenAI，不影響核心天氣查詢功能——Langfuse 純粹是可觀測性，不是天氣查詢的
# 必要依賴。
LANGFUSE_TRACING = False
if HAS_OPENAI_NEW and os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
    try:
        from langfuse.openai import OpenAI as OpenAI  # noqa: F811
        LANGFUSE_TRACING = True
    except Exception:
        pass

if LANGFUSE_TRACING:
    from langfuse import observe
else:
    def observe(*_args, **_kwargs):
        """Langfuse沒設定時的no-op替身，讓`@observe(...)`不管有沒有裝Langfuse都能用。"""
        def _decorator(fn):
            return fn
        if _args and callable(_args[0]):
            return _args[0]
        return _decorator

import json
from datetime import datetime, timezone, timedelta
from typing import Optional
import time

import requests
import glob
import xml.etree.ElementTree as ET
import sys

from repository import get_attractions_for_area, get_lodgings_for_area, get_district_coords

# RAG（Data/chroma/ 向量索引 + Data/bm25_index.pkl）是選配：索引沒建過、或缺
# chromadb/fastembed/rank_bm25 依賴時，不該讓整個 system.py 連天氣查詢都掛掉——
# import 失敗就退回 repository 的純名稱清單。優先用 hybrid_search（BM25+語意
# RRF融合），eval/ 實測 Hit Rate@5 比純語意檢索高（91.67% vs 83.33%，見
# ROADMAP.md Phase 1）；只有 hybrid_search 依賴（rank_bm25/jieba/bm25索引檔）
# 缺了才退回純語意的 rag.py，兩者介面(search)相同不用改呼叫端。
try:
    import hybrid_search as rag_module
    RAG_AVAILABLE = True
except Exception:
    try:
        import rag as rag_module
        RAG_AVAILABLE = True
    except Exception:
        rag_module = None
        RAG_AVAILABLE = False

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

# （已移除本機 SQLite DB 支援，改以 Open‑Meteo 作為唯一資料來源）

# 縣市＋行政區座標改查 Data/travel.db 的 district_coords 表（見
# collector/derive_district_coords.py），不再手打台中30區座標——舊版只涵蓋台中，
# 現在資料全台皆有，靠一份手打 dict 沒辦法跟著全台區維護。
DEFAULT_CITY = "臺中市"
DEFAULT_COORDS = (24.1477, 120.6736)  # 台中市中心，找不到任何座標時的最後備援


def _resolve_coords(area: Optional[str], city: str) -> tuple:
    if area:
        coords = get_district_coords(city, area)
        if coords:
            return coords
    return DEFAULT_COORDS


# 兼容舊版與新版 openai 套件的共用 helper——原本 analyze_prompt_with_llm() 跟
# generate_answer_from_user_prompt_and_data() 各自 nested 定義一份完全一樣的
# code，搬到 module 層級後兩邊共用同一份，改一次兩邊都生效。
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


def _is_machine_like(s: str) -> bool:
    """判斷 LLM 回傳是不是機器格式（純 JSON/空值），不是可讀文字。
    搬到 module 層級才能直接 import 測試——原本是
    generate_answer_from_user_prompt_and_data() 內部的 nested function，沒辦法
    單獨測。"""
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


@observe(name="analyze_prompt_with_llm")
def analyze_prompt_with_llm(user_prompt: str) -> dict:
    """
    呼叫 LLM 把使用者問題解析成 JSON，包含：
      - type: 'radar' 或 'forecast'
      - city: 縣市名稱 (e.g. '臺北市')，未提及時預設台中市
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
        "- city: 台灣縣市全名，如 '臺北市', '高雄市' 等（簡稱如台北、台中請轉成官方全名，"
        "'台'一律寫成'臺'）；使用者沒提到縣市時，預設 '臺中市'。\n"
        "- area: 該縣市的行政區名稱，如 '西屯區', '北區' 等；若提及景點，請推論該景點所屬的縣市與行政區。\n"
        "- date: 如果 type 是 forecast，請回傳查詢目標日期（格式：YYYY-MM-DD）\n"
        "- hour: 如有提供時間，回傳格式為 'HH:MM'，否則可省略\n\n"
    "僅回傳 JSON 結果，請勿加入解釋或多餘文字。"
    )

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


@observe(name="generate_answer_from_user_prompt_and_data")
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
        if _is_machine_like(generated_summary):
            generated_summary = "（OpenAI 回傳的內容無法解析為可讀文字，請稍候重試或調整查詢。）"
    except Exception:
        # 若檢查失敗，保留原始回覆
        pass

    return generated_summary


def fetch_radar_data(area: Optional[str], city: str = DEFAULT_CITY, limit: int = 5) -> str:
    """
    查詢指定縣市/行政區的即時雷達回波資料，取最新 limit 筆，並回傳成文字。
    """
    # 移除本機 DB 存取：直接使用 Open‑Meteo 的降水摘要作為雷達/降雨資料來源
    try:
        header = f"📡 使用 Open‑Meteo 取得 {city}{area or ''} 的近期降水摘要（不使用本機 DB）：\n"
        body = fetch_precipitation_open_meteo(area=area, city=city)
        return header + body
    except Exception as e:
        return _make_safe(f"{ERROR_PREFIX} 雷達資料查詢錯誤（Open‑Meteo）：{e}")


def get_latest_radar_summary(radar_dir: str = r"E:\python_project\contest\TGIS\radar") -> str:
    """
    讀取本地 radar/ 目錄，找最新的 XML 檔，回傳簡短摘要（檔名、修改時間、若能解析則回傳根節點資訊）。
    此函式不依賴遠端 API，適合作為本機 RAG 的一部分。
    """
    # 如果環境設定關閉本機資料，或 radar 目錄不存在／找不到檔案，改用遠端即時降水資料作為替代
    try:
        p = Path(radar_dir)
        if p.exists():
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


def fetch_precipitation_open_meteo(area: Optional[str] = None, city: str = DEFAULT_CITY, lat: Optional[float] = None, lon: Optional[float] = None) -> str:
    """
    使用 Open-Meteo 取得最近數小時的降雨量（作為 radar 的替代），若有 area 會查
    district_coords 表取得座標，回傳簡短文字摘要。
    """
    try:
        if lat is None or lon is None:
            lat, lon = _resolve_coords(area, city)

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


def fetch_current_weather_open_meteo(lat: float = 24.1477, lon: float = 120.6736, area: Optional[str] = None, city: str = DEFAULT_CITY) -> str:
    """
    使用 Open-Meteo 公共 API 取得當前天氣（free, 無需 API key）。
    回傳簡短文字摘要，便於作為 RAG 的上下文輸入給 LLM。
    """
    try:
        # 如果有提供區域名稱，查 district_coords 表取得座標
        if area:
            lat, lon = _resolve_coords(area, city)
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
    area: Optional[str],
    city: str,
    target_date: datetime,
    limit: int = 1,
    target_hour: Optional[str] = None
) -> str:
    """
    查詢指定縣市/行政區在 target_date (YYYY-MM-DD) 的天氣預報，
    若提供 target_hour (如 "06:00" 或 "18:00")，再依小時過濾，最後回傳文字結果。
    """
    try:
        # 直接使用 Open‑Meteo 取得該日預報（不再依賴本機 DB）
        lat, lon = _resolve_coords(area, city)
        return fetch_forecast_open_meteo(lat, lon, target_date, target_hour)
    except Exception as e:
        return _make_safe(f"{ERROR_PREFIX} 預報資料查詢錯誤（Open‑Meteo）：{e}")


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


# 使用者問句裡出現這些字眼，代表想要「不受天氣影響」的景點——只對 attraction 有意義，
# weather_fit 這個 metadata 欄位只有 attractions 才有（見 collector/attraction_classes.py）。
# 用問句本身判斷，而不是抓即時天氣的 weathercode：使用者常問的是假設性的「如果下雨」
# 或未來預報情境，不是當下天氣，問句字面比weathercode更直接對應使用者真正想要的東西。
RAIN_OR_INDOOR_KEYWORDS = ("下雨", "雨天", "會下雨", "陰雨", "室內", "室内")
INDOOR_WEATHER_FITS = ["indoor_ok", "mixed"]


def _wants_indoor(user_prompt: str) -> bool:
    return any(kw in user_prompt for kw in RAIN_OR_INDOOR_KEYWORDS)


def _score_rag_fallback(fell_back: bool) -> None:
    """幫Phase 5可觀測性補的信號：RAG查詢是不是退回了repository純名稱清單。
    fallback比例突然升高代表RAG層本身出問題（索引壞掉、依賴缺失等），比單純LLM
    延遲更早能看出RAG層健康度。Langfuse沒設定時直接跳過，不影響天氣查詢本身。"""
    if not LANGFUSE_TRACING:
        return
    try:
        from langfuse import get_client
        get_client().score_current_span(
            name="rag_fallback", value=1.0 if fell_back else 0.0, data_type="BOOLEAN"
        )
    except Exception:
        pass


@observe(name="rag_search")
def _get_places_for_area(user_prompt: str, area: str, city: str, place_type: str, top_n: int) -> list:
    """優先用 rag.search() 做語意檢索（依使用者原始問句排序，內容含真正的
    description，不只是名稱），向量索引沒建、依賴沒裝、或該區在索引裡查無資料
    （例如住宿description是樣板句被篩掉、沒進索引）時，退回 repository 的
    純 metadata 名稱清單，至少保證查得到東西。

    景點類查詢若問句提到下雨/室內，帶上 weather_fit 硬 filter（indoor_ok/mixed）
    ——室內外這種硬分類之前實測過純語意檢索猜不準，必須靠 metadata 過濾，不能只讓
    LLM 自己從 description 文字猜。"""
    if RAG_AVAILABLE:
        try:
            weather_fit = INDOOR_WEATHER_FITS if (place_type == "attraction" and _wants_indoor(user_prompt)) else None
            hits = rag_module.search(
                user_prompt, city=city, district=area, type_=place_type, weather_fit=weather_fit, top_k=top_n
            )
            if hits:
                _score_rag_fallback(False)
                return [h["text"] for h in hits]
        except Exception:
            pass
    _score_rag_fallback(True)
    if place_type == "attraction":
        return get_attractions_for_area(area, top_n=top_n, city=city)
    return get_lodgings_for_area(area, top_n=top_n, city=city)


@observe(name="get_weather_data")
def get_weather_data(user_prompt: str, area_override: Optional[str] = None, city_override: Optional[str] = None) -> str:
    """
    主要入口：根據使用者提問，執行以下步驟：
    1. analyze_prompt_with_llm 解析出 type, city, area, date, hour
    2. 依 type 決定呼叫 fetch_radar_data 或 fetch_forecast_data
    3. 如果查到天氣資料，繼續抓該縣市/行政區的景點與住宿清單
    4. 把「天氣＋景點＋住宿」組成 raw_data，呼叫 generate_answer_from_user_prompt_and_data
    5. 回傳 LLM 的最終建議 (不包含 system.py 自己的 CCTV 列表)
    """
    try:
        info = analyze_prompt_with_llm(user_prompt)
    except Exception as e: # 更通用的 Exception 捕捉
        return _make_safe(f"{ERROR_PREFIX} Prompt 分析階段出錯：{e}") # 返回錯誤信息

    query_type = info.get("type")
    area = info.get("area") # area 是中文區域名稱, e.g., "西屯區"
    city = info.get("city") or DEFAULT_CITY
    # 如果呼叫端提供覆寫的縣市/行政區，使用它
    if area_override:
        area = area_override
    if city_override:
        city = city_override

    if not area:
        return _make_safe(f"{ERROR_PREFIX} 無法辨識您想查詢的區域。") # 更友好的提示

    raw_data = "" # 初始化 raw_data
    if query_type == "radar":
        raw_data = fetch_radar_data(area, city)
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
            area,
            city,
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
        # 優先以 area+city 查 district_coords 取得即時天氣座標
        current_weather_summary = fetch_current_weather_open_meteo(area=area, city=city)
        raw_data = f"[RAG - 即時天氣]\n{current_weather_summary}\n\n" + raw_data
    except Exception:
        pass

    # 4. 抓取景點與住宿清單（優先語意檢索，含真正的description），並拼接到 raw_data
    attractions_list = _get_places_for_area(user_prompt, area, city, "attraction", top_n=5)
    if attractions_list:
        raw_data += "\n\n🏞️ 此區推薦景點：\n"
        for idx, name in enumerate(attractions_list, start=1):
            raw_data += f"  {idx}. {name}\n" # 修正縮排
    else:
        raw_data += f"\n\n🏞️ 抱歉，目前沒有「{area}」的景點資料可供推薦。\n"

    lodgings_list = _get_places_for_area(user_prompt, area, city, "lodging", top_n=3)
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
