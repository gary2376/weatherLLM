import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
import os
import importlib
import json
import time

import repository
import orchestrator
import bootstrap_data

SECRETS_BRIDGE_KEYS = ("OPENAI_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")


def _bridge_secrets_to_env(secrets_source, environ: dict) -> None:
    """系統邊界（見 ROADMAP.md Phase 0）：Streamlit Community Cloud 部署時金鑰要放
    在後台的Secrets管理介面，不是`.env`檔案（部署環境本來就沒有這個檔案，也不該把
    金鑰打包進repo/image）。用`setdefault`橋接：本機開發`.env`已經有值的話維持
    原樣不覆蓋，部署環境`.env`不存在時才從`secrets_source`（正式呼叫時是`st.secrets`）
    補進`environ`，下游`system.py`的`os.getenv(...)`呼叫完全不用改，也不用整套換成
    新的config系統。純函式（`secrets_source`/`environ`都當參數傳入，不直接碰
    `st.secrets`/`os.environ`），方便直接測試不用真的起Streamlit runtime；呼叫端
    （下面）負責接住`st.secrets`可能丟出的例外並傳真正的`os.environ`進來。"""
    for key in SECRETS_BRIDGE_KEYS:
        environ.setdefault(key, secrets_source.get(key, ""))


# 這段要放在`import system`（下面）之前——system.py在import當下就會檢查
# LANGFUSE_PUBLIC_KEY/SECRET_KEY決定要不要啟用Langfuse tracing。本機開發沒有
# 設定任何`.streamlit/secrets.toml`是正常情況，`st.secrets`此時可能丟例外
# （視Streamlit版本而定），接住例外而不是讓整個app掛掉。
try:
    _bridge_secrets_to_env(st.secrets, os.environ)
except Exception:
    pass

# 全域顯示前的 sanitizer，確保傳給 Streamlit 的一定是可顯示的字串
def sanitize_for_display(content) -> str:
    try:
        if content is None:
            return "（無內容）"
        if isinstance(content, str):
            if not content.strip():
                return "（無內容）"
            return content
        # 其他型別：嘗試 JSON 化，失敗就轉成 str()
        try:
            return json.dumps(content, ensure_ascii=False)
        except Exception:
            return str(content)
    except Exception:
        return "（訊息內容無法顯示）"

# ---- 新增：嘗試導入 system.py 中的函式（import整個module，方便存取get_weather_data） ----
SYSTEM_PY_AVAILABLE = False
# 預先定義一個替代函式，以防導入失敗
def get_system_weather_data_fallback(prompt_text: str) -> str:
    return "錯誤：進階建議系統模組 (system.py) 載入失敗或執行時發生錯誤，或尚未設定您的 OpenAI API Key，部分建議可能無法提供。"

get_system_weather_data = get_system_weather_data_fallback
system_module = None
# 強制使用遠端 Open-Meteo 作為天氣資料來源（不依賴本機 DB/XML）
os.environ['USE_LOCAL_DATA'] = os.getenv('USE_LOCAL_DATA', '0')  # 預設為 '0'，即使用遠端
try:
    # 匯入整個 module，才能拿到 system.get_weather_data；不是為了動態設定
    # system.openai.api_key——那個寫法已經因為跨session洩漏風險被拿掉了，
    # 見main()裡effective_api_key的說明，key是當參數傳，不是寫模組屬性
    system_module = importlib.import_module('system')
    if hasattr(system_module, 'get_weather_data'):
        get_system_weather_data = system_module.get_weather_data
        SYSTEM_PY_AVAILABLE = True
except Exception:
    system_module = None
# ---- END 新增 ----

# 系統邊界（見 ROADMAP.md Phase 4）：擋掉單一瀏覽器 session 短時間內狂刷，避免
# 有人把 RAG/embedding 運算或 Open-Meteo/OpenAI 呼叫當免費算力濫用。用
# session 而不是 IP：Streamlit 公開 API 拿不到可靠的 visitor IP，對個人 side
# project 的免費方案資源保護（擋單一使用者狂刷，不是防範分散式濫用）已經夠用。
RATE_LIMIT_MAX_REQUESTS = 10
RATE_LIMIT_WINDOW_SECONDS = 60


def _rate_limit_check(timestamps: List[float], now: float) -> Tuple[bool, List[float]]:
    """純函式，不碰 st.session_state，方便直接測試；呼叫端負責讀寫
    session_state。回傳 (是否允許這次請求, 更新後的 timestamps)。"""
    recent = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW_SECONDS]
    if len(recent) >= RATE_LIMIT_MAX_REQUESTS:
        return False, recent
    recent.append(now)
    return True, recent


def load_css():
    """載入自訂 CSS 樣式"""
    css_file = Path(__file__).parent / "style.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .main .block-container { padding-top: 2rem; max-width: 100%; }
        .main h1 { color: #1f4e79; text-align: center; font-weight: 700; border-bottom: 3px solid #4a90e2; padding-bottom: 1rem; }
        .stButton > button { background-color: #4a90e2; color: white; border-radius: 5px; border: none; padding: 0.5rem 1rem; font-weight: 500; }
        .stButton > button:hover { background-color: #357abd; }
        </style>
        """, unsafe_allow_html=True)

TAICHUNG_CENTER = (24.1477, 120.6736)


def create_attraction_map(attractions_df: Optional[pd.DataFrame] = None) -> folium.Map:
    m = folium.Map(location=list(TAICHUNG_CENTER), zoom_start=11, tiles='OpenStreetMap')
    if attractions_df is not None and not attractions_df.empty:
        for idx, row in attractions_df.iterrows():
            try: lat, lng = float(row['latitude']), float(row['longitude'])
            except (ValueError, TypeError): continue
            category, price_range = row.get('category'), row.get('price_range')
            place_label = category if pd.notna(category) else (price_range if pd.notna(price_range) else '景點')
            popup_content_attraction = f"""<div style="width: 250px;"><h4>{row['name']} ({place_label})</h4><p><b>地區:</b> {row.get('city', '')}{row.get('district', 'N/A')}</p><p><b>座標:</b> {lat:.6f}, {lng:.6f}</p></div>"""
            folium.Marker(location=[lat, lng], popup=folium.Popup(popup_content_attraction, max_width=270),
                          tooltip=f"{row['name']} ({place_label})",
                          icon=folium.Icon(color='green', icon='map-marker', prefix='fa')).add_to(m)
    return m

class AttractionManager:
    """景點/住宿/餐廳改讀 Data/travel.db（見 repository.py），不再預先整包載入記憶體，
    每次查詢才下 SQL 撈需要的那一小塊。新資料源沒有 Rating（見 ROADMAP.md Phase 0）；
    餐廳類別後來有補上（觀光署餐飲資料集），下面 `restaurant` 分支會查 `repository.query_restaurants()`。"""

    def get_attractions(self, city: str = "臺中市", district: Optional[str] = None,
                         attraction_type: Optional[str] = None, num_attractions: int = 5) -> pd.DataFrame:
        type_map = {"景點": "attraction", "住宿": "hotel", "飯店": "hotel", "餐廳": "restaurant"}
        kind = type_map.get(attraction_type, attraction_type)
        if kind == "hotel":
            data = repository.query_lodgings(city=city, district=district, top_n=num_attractions)
        elif kind == "restaurant":
            data = repository.query_restaurants(city=city, district=district, top_n=num_attractions)
        else:
            data = repository.query_attractions(city=city, district=district, top_n=num_attractions)
        if data.empty:
            return data
        return data.dropna(subset=['latitude', 'longitude'])

def _short_district_form(name: str) -> str:
    return name.replace("區", "").replace("市", "").replace("鄉", "").replace("鎮", "")


def _build_city_maps() -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, str], Dict[str, Dict[str, str]]]:
    """從 repository.get_all_city_districts() 動態產生 WeatherChatbot 關鍵字比對用的
    三份對照表，涵蓋 district_coords 表裡全部22縣市（原本手key只涵蓋台中/台北/
    新竹/台南/高雄5個，其餘17個縣市問了都會落到「找不到」或誤判成台中）。

    縣市簡稱去掉「市/縣」尾字後有兩組會撞名（新竹市/新竹縣都簡稱「新竹」，
    嘉義市/嘉義縣都簡稱「嘉義」），這裡把撞名的兩個官方縣市併進同一個別名
    分組、鍵盤搜尋清單合併，預設對到「市」那個（跟原本新竹的處理習慣一致：
    單講「新竹」查新竹市），縣專屬的行政區名稱另外進 DISTRICT_CITY_OVERRIDE
    校正——沿用原本「先猜市，行政區名稱能分辨縣市時才校正」的策略，不用
    整套重寫比對邏輯。呼叫時機要在Data/已經下載完成之後（WeatherChatbot
    建構時），不能放在module import當下就跑，那時候Data/可能還沒被
    bootstrap_data補齊。

    第四份`city_suffix_variants`（{別名: {"市"/"縣": 官方全名}}）只收錄真的
    存在的那個字尾——像「台中」只有「市」這個key，不會因為程式碼寫死
    `alias+"縣"`去比對訊息，就讓使用者打錯字的「台中縣」（2010年就併入
    臺中市了，現實中不存在）被誤導去查一個不存在的縣市名稱。"""
    raw = repository.get_all_city_districts()

    alias_groups: Dict[str, List[str]] = {}
    for full_name in raw:
        alias = full_name[:-1].replace("臺", "台")
        alias_groups.setdefault(alias, []).append(full_name)

    city_districts_map: Dict[str, List[str]] = {}
    city_name_map: Dict[str, str] = {}
    district_override: Dict[str, str] = {}
    city_suffix_variants: Dict[str, Dict[str, str]] = {}

    for alias in sorted(alias_groups, key=lambda a: a != "台中"):  # 台中固定排最前，維持既有的預設fallback行為
        variants = sorted(alias_groups[alias], key=lambda full: full[-1] != "市")  # 市優先當預設
        default_full = variants[0]
        city_name_map[alias] = default_full
        city_suffix_variants[alias] = {full[-1]: full for full in variants}
        default_districts = set(raw[default_full])

        dist_list = [alias, alias + "市", alias + "縣", default_full[:-1]]
        for full_name in variants:
            dist_list.extend(raw[full_name])
            if full_name != default_full:
                for district in raw[full_name]:
                    if district in default_districts:
                        continue  # 目前資料沒有同名跨市/縣的行政區，防禦性跳過避免誤蓋
                    district_override[district] = full_name
                    short = _short_district_form(district)
                    if short != district:
                        district_override[short] = full_name
        city_districts_map[alias] = dist_list

    return city_districts_map, city_name_map, district_override, city_suffix_variants


class WeatherChatbot:
    def __init__(self, attraction_manager: AttractionManager):
        self.attraction_manager = attraction_manager
        self.weather_phenomenon_keywords = {'下雨': '降雨', '雨': '降雨', '晴': '晴朗', '陰': '陰天', '風': '強風', '霧': '霧氣', '能見度': '能見度'}
        self.general_weather_query_keywords = ['天氣', '氣象']
        self.attraction_query_keywords = ['景點', '好玩', '去處', '旅遊', '玩樂', '觀光', '參觀', '逛逛', '玩']
        self.restaurant_query_keywords = ['餐廳', '美食', '吃飯', '小吃', '好吃']
        self.hotel_query_keywords = ['住宿', '飯店', '旅館', '民宿', '住哪']
        self.city_districts_map, self.CITY_NAME_MAP, self.DISTRICT_CITY_OVERRIDE, self.CITY_SUFFIX_VARIANTS = _build_city_maps()

    def process_message(self, user_message: str, override_district: Optional[str] = None) -> Tuple[str, Optional[pd.DataFrame], Optional[str]]: # Added query_type_detected
        """
        處理使用者訊息。
        - override_district: 若側欄有選取行政區，會傳入此參數以覆寫訊息中自動偵測的區域
        返回: (文字回應, 建議景點DataFrame, 偵測到的查詢類型)
        """
        return self._generate_response(user_message, override_district)

    def _generate_response(self, message: str, override_district: Optional[str] = None) -> Tuple[str, Optional[pd.DataFrame], Optional[str]]: # Added query_type_detected
        message_lower = message.lower()
        recommended_attractions_df: Optional[pd.DataFrame] = None
        city_districts_map = self.city_districts_map
        query_type, target_region, target_district = None, None, None

        # 審查發現的真bug：像「大安區」（台中/台北都有）、「北區」「東區」「南區」
        # （台中/台南都有）這些同名跨縣市的行政區，原本不管訊息裡有沒有明確提到
        # 縣市，一律照city_districts_map的字典insertion順序（台中最先）找到誰就
        # 用誰——導致「台北大安區」被誤判成台中的大安區、「台南北區」被誤判成
        # 台中的北區。修法：先掃一次訊息裡有沒有明確提到某個縣市別名，有的話
        # 該縣市的行政區清單優先比對，比對不到才退回原本的字典順序。
        mentioned_city_key = None
        for city_k, dist_list in city_districts_map.items():
            for alias in dist_list:
                is_alias = alias in (city_k, city_k + "市", city_k + "縣", city_k.replace("台", "臺"))
                if is_alias and alias in message:
                    mentioned_city_key = city_k
                    break
            if mentioned_city_key:
                break
        search_order = list(city_districts_map.keys())
        if mentioned_city_key:
            search_order = [mentioned_city_key] + [k for k in search_order if k != mentioned_city_key]

        for city_k in search_order:
            dist_list = city_districts_map[city_k]
            for name_part in dist_list:
                is_city_alias = name_part in (city_k, city_k + "市", city_k + "縣", city_k.replace("台", "臺"))
                if is_city_alias: continue
                short_form = name_part.replace("區", "").replace("市", "").replace("鄉", "").replace("鎮", "")
                if (name_part in message) or (len(short_form) >= 2 and short_form in message):
                    target_district = short_form if len(short_form) >= 2 and short_form in message else name_part
                    target_region = city_k; break
            if target_district: break
        if not target_district:
            for city_k, dist_list in city_districts_map.items():
                for city_alias in dist_list:
                    is_city_alias_k = (city_alias == city_k or city_alias == city_k + "市" or city_alias == city_alias.replace("台", "臺"))
                    if is_city_alias_k and city_alias in message: target_region = city_k; break
                if target_region and not target_district: break
        if not target_region and not target_district and any(k in message for k in ["全台", "台灣", "臺灣"]): target_region = "全台"

        is_general_weather_q = any(gkw in message for gkw in self.general_weather_query_keywords)
        # 使用者常常一次問兩種（「南投的景點以及附近的住宿」）——原本一次只認一種
        # 類型（景點優先），問兩種只會回第一種，看起來像沒聽懂另一半的問題。這裡
        # 收集訊息裡有出現的全部類型，下面一次查完全部、合併進同一則回覆。
        matched_types = [
            t for t, kws in (
                ("attraction", self.attraction_query_keywords),
                ("restaurant", self.restaurant_query_keywords),
                ("hotel", self.hotel_query_keywords),
            ) if any(kw in message for kw in kws)
        ]
        if matched_types: query_type = matched_types[0]
        elif is_general_weather_q: query_type = "weather"

        weather_type = None
        if query_type == "weather" or not query_type:
            for phenom_kw, weather_phenom_val in self.weather_phenomenon_keywords.items():
                if phenom_kw in message_lower: weather_type, query_type = weather_phenom_val, "weather"; break
        
        query_type_detected = query_type # Store the detected query type to return

        response_parts = []
        # 優先使用外部覆寫的行政區（例如側欄選擇），然後才使用 LLM 偵測出來的 district/region
        search_area = override_district if override_district else (target_district if target_district else target_region)

        if matched_types:
            place_map = {"attraction": "景點", "restaurant": "餐廳", "hotel": "住宿地點"}
            if not search_area:
                search_area = "台中"
                wanted = "、".join(place_map[t] for t in matched_types)
                response_parts = [f"您似乎沒有指定地區，我將為您搜尋'{search_area}'的{wanted}。"]
            explicit_variant = next(
                (full for suffix, full in self.CITY_SUFFIX_VARIANTS.get(target_region, {}).items()
                 if (target_region + suffix) in message),
                None,
            )
            resolved_city = (
                explicit_variant
                or self.DISTRICT_CITY_OVERRIDE.get(target_district)
                or self.CITY_NAME_MAP.get(target_region, "臺中市")
            )
            result_frames = []
            for search_type in matched_types:
                df = self.attraction_manager.get_attractions(
                    city=resolved_city, district=(target_district or None), attraction_type=search_type)
                if df is not None and not df.empty:
                    legend_text = " (地圖上以 <font color='green'>綠色圖釘</font> 標記)"
                    if search_type == "attraction": response_parts.append(f"建議關注的旅遊景點：{legend_text}")
                    else: response_parts.append(f"\n為您找到以下位於'{search_area}'的推薦{place_map[search_type]}:{legend_text}")
                    for i, (_, row) in enumerate(df.iterrows(), 1):
                        category = row.get('category')
                        price_range = row.get('price_range')
                        detail = category if pd.notna(category) else (price_range if pd.notna(price_range) else 'N/A')
                        response_parts.append(f"{i}. **{row['name']}** ({detail}, 行政區: {row.get('city','')}{row.get('district','N/A')})")
                    result_frames.append(df)
                else:
                    response_parts.append(f"抱歉，在'{search_area if search_area else (target_region if target_region else '指定區域')}'找不到符合條件的{place_map[search_type]}。")
            if result_frames:
                recommended_attractions_df = pd.concat(result_frames, ignore_index=True, sort=False)
        elif query_type_detected == "weather":
            if target_region and weather_type:
                response_parts.append(f"'{target_region}'地區 '{weather_type}' 天氣觀察。") # Streamlit bot's own weather text
            elif target_region:
                response_parts.append(f"'{target_region}'地區天氣觀察。") # Streamlit bot's own weather text
            elif weather_type:
                response_parts.append(f"全台'{weather_type}'觀察。") # Streamlit bot's own weather text
            else: response_parts.append("（關於天氣，請參考OpenAI的詳細建議）") # Placeholder if only general "weather"
        else: # Fallback for query_type_detected is None
            response_parts.append("我能為您查詢天氣、景點、餐廳或住宿。請試著問我更具體的問題。")

        return "\n".join(response_parts), recommended_attractions_df, query_type_detected # Return detected type

def initialize_session_state():
    if 'attraction_manager' not in st.session_state:
        st.session_state.attraction_manager = AttractionManager()
    if 'chatbot' not in st.session_state: st.session_state.chatbot = WeatherChatbot(st.session_state.attraction_manager)
    if 'recommended_attractions_on_map' not in st.session_state: st.session_state.recommended_attractions_on_map = None
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = [{"role": "assistant", "content": "您好！我是您的天氣與景點助手。請問您想了解哪個地區的天氣、景點、餐廳或住宿資訊？"}]
    else:
        # 如果 session 已有舊的 chat_messages，清理其內容避免遺留的機器格式被渲染
        try:
            cleaned = []
            for m in st.session_state.chat_messages:
                if isinstance(m, dict):
                    cleaned.append({
                        "role": m.get("role", "assistant"),
                        "content": sanitize_for_display(m.get("content"))
                    })
                else:
                    # 非 dict 的情況也要轉成合理的訊息
                    cleaned.append({"role": "assistant", "content": sanitize_for_display(m)})
            st.session_state.chat_messages = cleaned
        except Exception:
            # 若清理失敗，保留原本內容但不 crash
            pass


def main():

    st.set_page_config(page_title="個人化氣象決策系統", page_icon="🌤️", layout="wide", initial_sidebar_state="expanded")
    load_css()

    # 系統邊界（見 ROADMAP.md「資料版本化」項目）：`Data/`整個被`.gitignore`排除
    # 不進git版控（travel.db/Chroma索引/BM25索引加起來將近200MB，是可重新產生的
    # 衍生資料不是原始碼）。全新部署（例如Streamlit Community Cloud剛clone完）
    # 第一次啟動時`Data/`是空的，這裡從GitHub Release下載資料包補齊——`Data/`
    # 已經存在的話（本機開發、或部署後的第二次以後啟動）`ensure_data_present()`
    # 內部會直接跳過，不重複下載。放在`initialize_session_state()`之前執行，
    # 因為那邊會建立`AttractionManager`讀`Data/travel.db`。
    _data_already_exists = bootstrap_data.data_exists()
    print(f"[bootstrap_data] main(): data_exists()={_data_already_exists}", flush=True)
    if not _data_already_exists:
        with st.spinner("首次啟動，正在下載天氣/景點資料庫（約90MB，僅需一次）..."):
            bootstrap_data.ensure_data_present()

    # 側欄：僅保留 OpenAI API Key 輸入（其餘選項已移除以簡化介面）
    with st.sidebar:
        st.header("系統設定")
        st.markdown("使用進階服務需輸入您的 API Key（不會儲存在 repo 或伺服器）。")
        user_api_key_input = st.text_input("OpenAI API Key", type="password", key="openai_api_key_input")
        st.caption("此金鑰僅存在於本 session，請勿公開。")
        if user_api_key_input:
            # 只存進session_state（per-session），不寫進os.environ/openai.api_key
            # 這種process全域狀態——Streamlit常常多個使用者共用同一個process，
            # 寫全域變數等於A使用者的key可能被B使用者的請求誤用（見ROADMAP.md
            # 稽核紀錄）。真正呼叫OpenAI時把key當參數往下傳，見main()裡
            # get_system_weather_data的呼叫。
            st.session_state['user_api_key'] = user_api_key_input
            st.success("OpenAI API Key 已設定（僅在此 session 有效）")
        else:
            # 若 session 中已存在舊金鑰，保持不動；否則顯示提示
            if not st.session_state.get('user_api_key'):
                st.info("若要使用進階天氣建議，請在此輸入您的 OpenAI API Key。")

    # 在側欄讀取完使用者可能的 data_dir 覆寫後再初始化 session
    initialize_session_state()

    st.title("🌤️ 天氣輔助旅遊系統")
    st.markdown("---")
    col1, col2 = st.columns([3, 2])

    with col2:
        st.header("💬 AI助理")
        chat_container = st.container(height=500)
        with chat_container:
            for msg_item in st.session_state.chat_messages:
                # 安全化顯示：避免把 list/dict/None 等機器格式原樣傳給 st.markdown，導致瀏覽器顯示 JSON/NULL 結構
                content = msg_item.get("content") if isinstance(msg_item, dict) else None
                try:
                    if content is None:
                        display_text = "（無內容）"
                    elif isinstance(content, str):
                        if not content.strip():
                            display_text = "（無內容）"
                        else:
                            display_text = content
                    else:
                        # 其他型別（list/dict/tuple 等），嘗試以 JSON 字串顯示；若失敗則用 str()
                        try:
                            display_text = json.dumps(content, ensure_ascii=False)
                        except Exception:
                            display_text = str(content)
                except Exception:
                    display_text = "（訊息內容無法顯示）"

                with st.chat_message(msg_item.get("role", "assistant")):
                    # 不允許 unsafe html，避免意外渲染
                    st.markdown(display_text, unsafe_allow_html=False)
        
        user_input = st.chat_input("請問您想了解什麼天氣或景點資訊？")

        if user_input:
            allowed, updated_timestamps = _rate_limit_check(
                st.session_state.get('_request_timestamps', []), time.time()
            )
            st.session_state['_request_timestamps'] = updated_timestamps

            st.session_state.chat_messages.append({"role": "user", "content": user_input})

            if not allowed:
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": f"請求太頻繁，請稍候再試（{RATE_LIMIT_WINDOW_SECONDS} 秒內最多 {RATE_LIMIT_MAX_REQUESTS} 次），避免佔用免費方案資源。",
                })
                st.rerun()

            sel_dist = st.session_state.get('selected_district_final')

            # 是否要走 OpenAI 進階建議路徑：跟 orchestrator.supervisor_node 內部
            # 用的是同一份 chatbot 關鍵字表，這裡先問一次純粹是為了決定要不要跳
            # st.warning 提示使用者缺 API Key（UI 層決定），不是重複算一次要花
            # 錢的東西——supervisor_node 本身不打任何外部 API。
            should_call_advanced_preview = orchestrator.supervisor_node(
                {"user_input": user_input, "chatbot": st.session_state.chatbot}
            )["should_call_advanced"]

            # 只讀不寫：session自己貼的key優先，否則退回owner在環境變數設的
            # （本機開發/owner自付費部署才會有這條，公開BYOK部署下通常是空的）。
            # 不寫回os.environ/openai.api_key這種process全域狀態——見側欄那段
            # 的說明，避免多使用者共用process時互相洩漏key。
            effective_api_key = st.session_state.get('user_api_key') or os.getenv('OPENAI_API_KEY')
            api_key_available = bool(effective_api_key) and should_call_advanced_preview and SYSTEM_PY_AVAILABLE

            if should_call_advanced_preview and SYSTEM_PY_AVAILABLE and not api_key_available:
                st.warning("要使用進階天氣建議功能，請先在左側輸入您的 OpenAI API Key（private）。")

            # 多輪對話 Structured State（見 ROADMAP.md Phase 2 SCOPE 設計草案）：
            # 上一輪解析出的 info（city/area/date/hour）存在 session_state，這輪
            # 當 previous_info 傳進去，讓「那附近有推薦餐廳嗎」這種沒有重講地點
            # 的問句能延續上一輪的地點。這裡只負責讀寫 session_state 這個
            # Streamlit-only 動作，實際的繼承/覆寫判斷邏輯在
            # weather_graph.py::_make_intent_node()，不碰 session_state。
            def _call_system_weather(prompt, override):
                info_out: dict = {}
                answer = get_system_weather_data(
                    prompt, override, api_key=effective_api_key,
                    previous_info=st.session_state.get('last_weather_info'),
                    info_out=info_out,
                )
                if info_out:
                    st.session_state['last_weather_info'] = info_out
                return answer

            # LangGraph supervisor + 兩個 worker（WeatherChatbot 輕量路徑／system.py
            # OpenAI 進階路徑）+ 組合節點，取代原本這裡一大段手動 if/else 拼接。
            # 見 ROADMAP.md Phase 2、code/orchestrator.py。key當參數傳進closure，
            # 不靠全域狀態——見上面effective_api_key的說明。
            with st.spinner("正在處理您的請求..."):
                result = orchestrator.run_orchestrator(
                    user_input=user_input,
                    chatbot=st.session_state.chatbot,
                    system_weather_fn=_call_system_weather,
                    override_district=sel_dist,
                    api_key_available=api_key_available,
                )

            st.session_state.recommended_attractions_on_map = result.get("sl_attr_df")

            final_display_text = result.get("final_display_text") or \
                "抱歉，目前無法處理您的請求，請再試一次或調整您的問題。"

            st.session_state.chat_messages.append({"role": "assistant", "content": final_display_text})
            st.rerun()

    with col1:
        st.header("🗺️ 地圖")
        with st.expander("地圖控制選項", expanded=True):
            if st.button("清除推薦景點"):
                st.session_state.recommended_attractions_on_map = None; st.rerun()

        attr_map = st.session_state.recommended_attractions_on_map
        current_map_obj = create_attraction_map(attractions_df=attr_map)
        st_folium(current_map_obj, width="100%", height=550, returned_objects=["last_clicked"])


if __name__ == "__main__":
    main()