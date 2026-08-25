import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
import numpy as np
import pygeohash as pgh
import os
import importlib
import json

import repository

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

# ---- 新增：嘗試導入 system.py 中的函式（改為 import module，方便動態設定 openai.api_key） ----
SYSTEM_PY_AVAILABLE = False
# 預先定義一個替代函式，以防導入失敗
def get_system_weather_data_fallback(prompt_text: str) -> str:
    return "錯誤：進階建議系統模組 (system.py) 載入失敗或執行時發生錯誤，或尚未設定您的 OpenAI API Key，部分建議可能無法提供。"

get_system_weather_data = get_system_weather_data_fallback
system_module = None
# 強制使用遠端 Open-Meteo 作為天氣資料來源（不依賴本機 DB/XML）
os.environ['USE_LOCAL_DATA'] = os.getenv('USE_LOCAL_DATA', '0')  # 預設為 '0'，即使用遠端
try:
    # 匯入整個 module，稍後可動態設定 system.openai.api_key
    system_module = importlib.import_module('system')
    if hasattr(system_module, 'get_weather_data'):
        get_system_weather_data = system_module.get_weather_data
        SYSTEM_PY_AVAILABLE = True
except Exception:
    system_module = None
# ---- END 新增 ----


# 請確認您已安裝 pygeohash: pip install pygeohash

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

class CCTVManager:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.cctv_data: pd.DataFrame = self._load_cctv_data()

    def _load_cctv_data(self) -> pd.DataFrame:
        default_cols = ['name', 'lat', 'long', 'link', 'geohash']
        try:
            data = pd.read_csv(self.csv_path)
            data = data.dropna(subset=['name', 'lat', 'long', 'link'])
            geohash_precision_storage = 7 
            data['geohash'] = data.apply(
                lambda row: pgh.encode(float(row['lat']), float(row['long']), precision=geohash_precision_storage)
                if pd.notnull(row['lat']) and pd.notnull(row['long']) else None,
                axis=1
            )
            data = data.dropna(subset=['geohash'])
            return data
        except Exception as e:
            st.error(f"載入 CCTV 資料時發生錯誤: {e}")
            return pd.DataFrame(columns=default_cols)

    def get_taichung_center(self) -> Tuple[float, float]:
        return 24.1477, 120.6736

    def create_map(self,
                   cctv_data_to_plot: Optional[pd.DataFrame] = None,
                   highlighted_cctv_names: Optional[List[str]] = None,
                   attractions_df: Optional[pd.DataFrame] = None) -> folium.Map:
        center_lat, center_long = self.get_taichung_center()
        m = folium.Map(location=[center_lat, center_long], zoom_start=11, tiles='OpenStreetMap')
        if cctv_data_to_plot is not None and not cctv_data_to_plot.empty:
            for idx, row in cctv_data_to_plot.iterrows():
                is_highlighted = highlighted_cctv_names and row['name'] in highlighted_cctv_names
                color = 'red' if is_highlighted else 'blue'
                icon_name = 'video-camera' if is_highlighted else 'camera'
                popup_content_cctv = f"""<div style="width: 300px;"><h4>{row['name']} (CCTV)</h4><p><b>座標:</b> {row['lat']:.6f}, {row['long']:.6f}</p><p><b>即時影像:</b></p><img src="{row['link']}" width="280" style="border: 1px solid #ccc;" onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjgwIiBoZWlnaHQ9IjE1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjgwIiBoZWlnaHQ9IjE1MCIgZmlsbD0iI2Y0ZjRmNCIvPjx0ZXh0IHg9IjE0MCIgeT0iNzUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPuW9seePoueEoeazlTwvdGV4dD48L3N2Zz4=';" /><br><br><a href="{row['link']}" target="_blank" style="color: #0066cc;">開啟完整影像</a></div>"""
                folium.Marker(location=[row['lat'], row['long']], popup=folium.Popup(popup_content_cctv, max_width=320), tooltip=row['name'],
                              icon=folium.Icon(color=color, icon=icon_name, prefix='fa', icon_color='white')).add_to(m)
        if attractions_df is not None and not attractions_df.empty:
            for idx, row in attractions_df.iterrows():
                try: lat, lng = float(row['latitude']), float(row['longitude'])
                except (ValueError, TypeError): continue
                place_label = row.get('category') or row.get('price_range') or '景點'
                popup_content_attraction = f"""<div style="width: 250px;"><h4>{row['name']} ({place_label})</h4><p><b>地區:</b> {row.get('city', '')}{row.get('district', 'N/A')}</p><p><b>座標:</b> {lat:.6f}, {lng:.6f}</p></div>"""
                folium.Marker(location=[lat, lng], popup=folium.Popup(popup_content_attraction, max_width=270),
                              tooltip=f"{row['name']} ({place_label})",
                              icon=folium.Icon(color='green', icon='map-marker', prefix='fa')).add_to(m)
        return m

    def get_nearby_cctvs(self, lat: float, lng: float, radius_km: float = 1.0) -> List[Dict]:
        def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
            R, rlat1, rlon1, rlat2, rlon2 = 6371, np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2) # Renamed for clarity
            dlat, dlon = rlat2 - rlat1, rlon2 - rlon1
            a = np.sin(dlat/2)**2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon/2)**2
            return R * 2 * np.arcsin(np.sqrt(a))
        nearby_cctvs = []
        if self.cctv_data.empty: return nearby_cctvs
        for idx, row in self.cctv_data.iterrows():
            try:
                cctv_lat, cctv_long = float(row['lat']), float(row['long'])
                distance = haversine_distance(lat, lng, cctv_lat, cctv_long)
                if distance <= radius_km:
                    nearby_cctvs.append({'name': row['name'], 'distance': distance, 'lat': cctv_lat, 'long': cctv_long, 'link': row['link']})
            except (ValueError, TypeError): continue
        return sorted(nearby_cctvs, key=lambda x: x['distance'])

class AttractionManager:
    """景點/住宿改讀 Data/travel.db（見 repository.py），不再預先整包載入記憶體，
    每次查詢才下 SQL 撈需要的那一小塊。新資料源沒有 Rating、沒有餐廳類別（見 ROADMAP.md Phase 0）。"""

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
        data['geohash'] = data.apply(
            lambda row: pgh.encode(row['latitude'], row['longitude'], precision=7)
            if pd.notnull(row['latitude']) and pd.notnull(row['longitude']) else None, axis=1)
        return data.dropna(subset=['geohash'])

class WeatherChatbot:
    def __init__(self, cctv_manager: CCTVManager, attraction_manager: AttractionManager):
        self.cctv_manager = cctv_manager
        self.attraction_manager = attraction_manager
        self.weather_phenomenon_keywords = {'下雨': '降雨', '雨': '降雨', '晴': '晴朗', '陰': '陰天', '風': '強風', '霧': '霧氣', '能見度': '能見度'}
        self.general_weather_query_keywords = ['天氣', '氣象']
        self.attraction_query_keywords = ['景點', '好玩', '去處', '旅遊', '玩樂', '觀光', '參觀', '逛逛', '玩']
        self.restaurant_query_keywords = ['餐廳', '美食', '吃飯', '小吃', '好吃']
        self.hotel_query_keywords = ['住宿', '飯店', '旅館', '民宿', '住哪']

    CITY_NAME_MAP = {"台中": "臺中市", "台北": "臺北市", "新竹": "新竹市", "台南": "臺南市", "高雄": "高雄市"}

    def process_message(self, user_message: str, override_district: Optional[str] = None) -> Tuple[str, List[str], Optional[pd.DataFrame], Optional[str]]: # Added query_type_detected
        """
        處理使用者訊息。
        - override_district: 若側欄有選取行政區，會傳入此參數以覆寫訊息中自動偵測的區域
        返回: (文字回應, 建議CCTV列表, 建議景點DataFrame, 偵測到的查詢類型)
        """
        return self._generate_response(user_message, override_district)

    def _generate_response(self, message: str, override_district: Optional[str] = None) -> Tuple[str, List[str], Optional[pd.DataFrame], Optional[str]]: # Added query_type_detected
        message_lower = message.lower()
        recommended_cctvs: List[str] = []
        recommended_attractions_df: Optional[pd.DataFrame] = None
        city_districts_map = {'台中': ['中區', '北區', '西區', '南區', '東區', '北屯區', '西屯區', '南屯區', '太平區', '大里區', '霧峰區', '烏日區', '豐原區', '后里區', '石岡區', '東勢區', '和平區', '新社區', '潭子區', '大雅區', '神岡區', '大肚區', '沙鹿區', '龍井區', '梧棲區', '清水區', '大甲區', '外埔區', '大安區', '台中市', '臺中'], '台北': ['台北', '士林', '內湖', '大安', '信義', '中山', '中正', '萬華', '文山', '北投', '大同', '南港', '臺北'], '新竹': ['新竹', '竹北', '湖口', '竹東'], '台南': ['台南', '安南', '北區', '中西區', '東區', '南區', '永康', '安平', '新營', '臺南'], '高雄': ['高雄', '鳳山', '左營', '三民', '苓雅', '前鎮', '楠梓', '鼓山']}
        query_type, target_region, target_district = None, None, None

        for city_k, dist_list in city_districts_map.items():
            for name_part in dist_list:
                is_city_alias = (name_part == city_k or name_part == city_k + "市" or name_part == city_k.replace("台", "臺"))
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
        if any(kw in message for kw in self.attraction_query_keywords): query_type = "attraction"
        elif any(kw in message for kw in self.restaurant_query_keywords): query_type = "restaurant"
        elif any(kw in message for kw in self.hotel_query_keywords): query_type = "hotel"
        elif is_general_weather_q: query_type = "weather"
        
        weather_type = None
        if query_type == "weather" or not query_type:
            for phenom_kw, weather_phenom_val in self.weather_phenomenon_keywords.items():
                if phenom_kw in message_lower: weather_type, query_type = weather_phenom_val, "weather"; break
        
        query_type_detected = query_type # Store the detected query type to return

        response_parts = []
        # 優先使用外部覆寫的行政區（例如側欄選擇），然後才使用 LLM 偵測出來的 district/region
        search_area = override_district if override_district else (target_district if target_district else target_region)

        if query_type_detected in ["attraction", "restaurant", "hotel"]: # Use detected query type
            place_map = {"attraction": "景點", "restaurant": "餐廳", "hotel": "住宿地點"}
            search_type = query_type_detected
            if not search_area:
                search_area = "台中"
                response_parts = [f"您似乎沒有指定地區，我將為您搜尋'{search_area}'的{place_map[search_type]}。"]
            resolved_city = self.CITY_NAME_MAP.get(target_region, "臺中市")
            recommended_attractions_df = self.attraction_manager.get_attractions(
                city=resolved_city, district=(target_district or None), attraction_type=search_type)
            if recommended_attractions_df is not None and not recommended_attractions_df.empty:
                legend_text = " (地圖上以 <font color='green'>綠色圖釘</font> 標記)"
                if search_type == "attraction": response_parts.append(f"建議關注的旅遊景點：{legend_text}")
                else: response_parts.append(f"\n為您找到以下位於'{search_area}'的推薦{place_map[search_type]}:{legend_text}")
                for i, (_, row) in enumerate(recommended_attractions_df.iterrows(), 1):
                    detail = row.get('category') or row.get('price_range') or 'N/A'
                    response_parts.append(f"{i}. **{row['name']}** ({detail}, 行政區: {row.get('city','')}{row.get('district','N/A')})")
                cctvs_near_attractions_names = []
                geohash_search_precision = 6 
                all_cctvs_df = self.cctv_manager.cctv_data
                if 'geohash' in all_cctvs_df.columns and not all_cctvs_df.empty and 'geohash' in recommended_attractions_df.columns and not recommended_attractions_df.empty:
                    target_geohashes_for_search = set()
                    for _, attr_row in recommended_attractions_df.iterrows():
                        attr_geohash_full = attr_row.get('geohash')
                        if attr_geohash_full and isinstance(attr_geohash_full, str) and len(attr_geohash_full) >= geohash_search_precision:
                            attr_geohash_prefix = attr_geohash_full[:geohash_search_precision]
                            target_geohashes_for_search.add(attr_geohash_prefix)
                            try: target_geohashes_for_search.update(pgh.neighbors(attr_geohash_prefix))
                            except Exception: pass 
                    unique_cctv_names_found = set()
                    for _, cctv_row in all_cctvs_df.iterrows():
                        cctv_geohash_full = cctv_row.get('geohash')
                        if cctv_geohash_full and isinstance(cctv_geohash_full, str) and len(cctv_geohash_full) >= geohash_search_precision:
                            cctv_geohash_prefix = cctv_geohash_full[:geohash_search_precision]
                            if cctv_geohash_prefix in target_geohashes_for_search: unique_cctv_names_found.add(cctv_row['name'])
                    cctvs_near_attractions_names = list(unique_cctv_names_found)
                if cctvs_near_attractions_names:
                    recommended_cctvs = cctvs_near_attractions_names[:5] 
                    if recommended_cctvs: response_parts.append(f"\n在您關注的景點附近找到以下監視器，方便您確認即時路況：")
                elif target_region == "台中": 
                    taichung_cctvs = self._get_region_cctvs(target_region)
                    if taichung_cctvs:
                        current_cctv_set = set(recommended_cctvs); [recommended_cctvs.append(c) for c in taichung_cctvs if c not in current_cctv_set]
                        recommended_cctvs = recommended_cctvs[:5]
                        if recommended_cctvs: response_parts.append(f"\n同時，您可以參考'{search_area if search_area else target_region}'區域的即時影像。")
            else: 
                response_parts.append(f"抱歉，在'{search_area if search_area else (target_region if target_region else '指定區域')}'找不到符合條件的{place_map[search_type]}。")
                if query_type_detected == "attraction" and (target_region == "台中" or (search_area and "台中" in search_area)):
                    taichung_cctvs = self._get_region_cctvs("台中")
                    if taichung_cctvs: recommended_cctvs = taichung_cctvs[:5]
        elif query_type_detected == "weather":
            c_cctv_list_for_weather = []
            if target_region and weather_type:
                response_parts.append(f"'{target_region}'地區 '{weather_type}' 天氣觀察。") # Streamlit bot's own weather text
                c_cctv_list_for_weather = self._get_region_cctvs(target_region)
            elif target_region:
                response_parts.append(f"'{target_region}'地區天氣觀察。") # Streamlit bot's own weather text
                c_cctv_list_for_weather = self._get_region_cctvs(target_region)
            elif weather_type:
                response_parts.append(f"全台'{weather_type}'觀察。") # Streamlit bot's own weather text
                c_cctv_list_for_weather = self._get_weather_related_cctvs(weather_type)
            else: response_parts.append("（關於天氣，請參考OpenAI的詳細建議）") # Placeholder if only general "weather"
            if c_cctv_list_for_weather:
                current_cctv_set = set(recommended_cctvs); [recommended_cctvs.append(c) for c in c_cctv_list_for_weather if c not in current_cctv_set]
        else: # Fallback for query_type_detected is None
            response_parts.append("我能為您查詢天氣、景點、餐廳或住宿。請試著問我更具體的問題。")
            default_cctvs = self._get_region_cctvs("台中")
            if default_cctvs: current_cctv_set = set(recommended_cctvs); [recommended_cctvs.append(c) for c in default_cctvs if c not in current_cctv_set]
        
        if recommended_cctvs: recommended_cctvs = list(set(recommended_cctvs))[:5] 
        return "\n".join(response_parts), recommended_cctvs, recommended_attractions_df, query_type_detected # Return detected type

    def _get_region_cctvs(self, region: str) -> List[str]:
        if self.cctv_manager.cctv_data.empty: return []
        norm_reg, alt_reg = region.replace("台", "臺"), region.replace("臺", "台")
        rel_cctvs = [row['name'] for _, row in self.cctv_manager.cctv_data.iterrows() if any(r in str(row['name']) for r in [norm_reg, alt_reg, region])]
        if not rel_cctvs and len(self.cctv_manager.cctv_data) > 0: return self.cctv_manager.cctv_data['name'].sample(min(5, len(self.cctv_manager.cctv_data))).tolist()
        return list(set(rel_cctvs))[:10]

    def _get_weather_related_cctvs(self, weather_type: str) -> List[str]:
        if self.cctv_manager.cctv_data.empty: return []
        rel_cctvs = []
        if weather_type in ['降雨', '霧氣', '強風']:
            kw_chk = ['國道', '省道', '橋', '交流道', '快速道路', '港', '機場', '車站', '高鐵']
            rel_cctvs = [row['name'] for _, row in self.cctv_manager.cctv_data.iterrows() if any(kw in str(row['name']) for kw in kw_chk)]
            if not rel_cctvs and len(self.cctv_manager.cctv_data) > 0: return self.cctv_manager.cctv_data['name'].sample(min(8, len(self.cctv_manager.cctv_data))).tolist()
        elif len(self.cctv_manager.cctv_data) > 0: return self.cctv_manager.cctv_data['name'].sample(min(8, len(self.cctv_manager.cctv_data))).tolist()
        return list(set(rel_cctvs))[:8]

def initialize_session_state():
    # 計算 Data 目錄：使用 repo 相對路徑 Data/
    try:
        project_root = Path(__file__).resolve().parents[1]
    except Exception:
        project_root = Path.cwd()
    data_dir = project_root / 'Data'

    cctv_path = str(data_dir / 'cctv_enhanced.csv')

    if 'cctv_manager' not in st.session_state:
        st.session_state.cctv_manager = CCTVManager(str(cctv_path))
    if 'attraction_manager' not in st.session_state:
        st.session_state.attraction_manager = AttractionManager()
    if 'chatbot' not in st.session_state: st.session_state.chatbot = WeatherChatbot(st.session_state.cctv_manager, st.session_state.attraction_manager)
    if 'highlighted_cctvs' not in st.session_state: st.session_state.highlighted_cctvs = []
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
    if 'show_all_cctvs_checkbox_value' not in st.session_state: st.session_state.show_all_cctvs_checkbox_value = False


def main():
    # 宣告我們會在此函式內修改這些 module-level 變數，避免 UnboundLocalError
    global system_module, get_system_weather_data, SYSTEM_PY_AVAILABLE

    st.set_page_config(page_title="個人化氣象決策系統", page_icon="🌤️", layout="wide", initial_sidebar_state="expanded")
    load_css()

    # 側欄：僅保留 OpenAI API Key 輸入（其餘選項已移除以簡化介面）
    with st.sidebar:
        st.header("系統設定")
        st.markdown("使用進階服務需輸入您的 API Key（不會儲存在 repo 或伺服器）。")
        user_api_key_input = st.text_input("OpenAI API Key", type="password", key="openai_api_key_input")
        st.caption("此金鑰僅存在於本 session，請勿公開。")
        if user_api_key_input:
            # 儲存在 session_state 以便同一 session 使用
            st.session_state['user_api_key'] = user_api_key_input
            try:
                # 若 system.py 已被成功匯入，動態設定其 openai.api_key
                if system_module is not None and hasattr(system_module, 'openai'):
                    system_module.openai.api_key = user_api_key_input
                # 也放到環境變數（方便其他套件以 env 取得）
                os.environ['OPENAI_API_KEY'] = user_api_key_input
                st.success("OpenAI API Key 已設定（僅在此 session 有效）")
            except Exception:
                st.error("設定 API Key 時發生錯誤，進階功能可能無法使用。")
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
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            
            system_py_main_response = ""
            streamlit_chatbot_text_part = "" 
            
            # 1. Streamlit 內部機器人處理，獲取地圖資料、CCTV列表、初步文字和查詢類型
            with st.spinner("正在分析您的請求..."):
                # 不再使用側欄覆寫，改為讓使用者在 prompt 中直接提及行政區
                sl_resp_text, sl_cctv_names, sl_attr_df, sl_query_type = \
                    st.session_state.chatbot.process_message(user_input)
            
            st.session_state.highlighted_cctvs = sl_cctv_names
            st.session_state.recommended_attractions_on_map = sl_attr_df

            # --- MODIFIED: 判斷是否為天氣相關查詢以呼叫 system.py ---
            should_call_system_py = False
            temp_marker = None
            # 檢查是否包含一般天氣關鍵字 (如 "天氣", "氣象")
            if hasattr(st.session_state.chatbot, 'general_weather_query_keywords') and \
               any(gkw in user_input for gkw in st.session_state.chatbot.general_weather_query_keywords):
                should_call_system_py = True
            
            # 如果還不是，再檢查是否包含特定天氣現象關鍵字 (如 "下雨", "晴朗")
            if not should_call_system_py and \
               hasattr(st.session_state.chatbot, 'weather_phenomenon_keywords') and \
               any(pkw in user_input.lower() for pkw in st.session_state.chatbot.weather_phenomenon_keywords.keys()): # 注意這裡用 .keys()
                should_call_system_py = True
            # --- END MODIFIED ---

            # Helper: 偵測回傳是否看起來像機器格式 (JSON / list / null 等)
            def _is_machine_format(content):
                try:
                    if content is None:
                        return True
                    if not isinstance(content, str):
                        return True
                    s = content.strip()
                    if not s:
                        return True
                    low = s.lower()
                    # 明顯的 JSON 結構或包含 null/none
                    if low.startswith('[') or low.startswith('{'):
                        if 'null' in low or 'none' in low or ',' in s:
                            return True
                        return True
                    # 有時候回傳會是 Python-list/string repr
                    if s.startswith('[') and ']' in s:
                        return True
                    # 含有多個逗號且沒有中文，可能是 machine repr
                    if s.count(',') >= 2 and all(ord(ch) < 128 for ch in s if ch.isalpha()):
                        return True
                except Exception:
                    return True
                return False

            def _sanitize_system_response(content):
                # 對 system.py 回傳做友善處理，避免把 raw JSON/NULL 列表直接顯示給使用者。
                fallback = "抱歉，進階建議系統暫時無法提供可讀回覆，請稍後或調整查詢內容。"
                try:
                    if content is None:
                        return fallback
                    if not isinstance(content, str):
                        # 嘗試轉換簡單型別
                        try:
                            import json
                            return json.dumps(content, ensure_ascii=False)
                        except Exception:
                            return fallback
                    # 如果看起來像 machine format，嘗試解析 JSON 並抽取可讀文字
                    if _is_machine_format(content):
                        try:
                            import json
                            parsed = json.loads(content)
                            if isinstance(parsed, list):
                                for item in parsed:
                                    if isinstance(item, str) and item.strip():
                                        return item
                                return fallback
                            if isinstance(parsed, dict):
                                for k in ('text', 'content', 'message'):
                                    if k in parsed and isinstance(parsed[k], str):
                                        return parsed[k]
                                return json.dumps(parsed, ensure_ascii=False)
                        except Exception:
                            # 簡單清理一些常見 tokens
                            s = content.replace('[', '').replace(']', '').replace('null', '').replace('None', '').strip()
                            if s:
                                return s
                            return fallback
                    return content
                except Exception:
                    return fallback

            # 2. 如果是天氣相關查詢且 system.py 可用，則在已設定 API Key 的情況下呼叫 system.py
            if should_call_system_py and SYSTEM_PY_AVAILABLE:
                # 確認是否有 API Key：優先使用 system_module.openai.api_key -> session_state -> 環境變數
                api_key_available = False
                try:
                    if system_module is not None and hasattr(system_module, 'openai') and getattr(system_module.openai, 'api_key', None):
                        api_key_available = True
                    elif st.session_state.get('user_api_key'):
                        if system_module is not None and hasattr(system_module, 'openai'):
                            system_module.openai.api_key = st.session_state.get('user_api_key')
                        os.environ['OPENAI_API_KEY'] = st.session_state.get('user_api_key')
                        api_key_available = True
                    elif os.getenv('OPENAI_API_KEY'):
                        if system_module is not None and hasattr(system_module, 'openai'):
                            system_module.openai.api_key = os.getenv('OPENAI_API_KEY')
                        api_key_available = True
                except Exception:
                    api_key_available = False

                if not api_key_available:
                    st.warning("要使用進階天氣建議功能，請先在左側輸入您的 OpenAI API Key（private）。")
                else:
                    # 如果使用者在側欄選了行政區，將其作為覆寫參數傳給 system.get_weather_data
                    sel_dist = st.session_state.get('selected_district_final')
                    # 在 UI 先插入佔位訊息，避免在等待期間或失敗時顯示 raw machine output
                    temp_marker = "（系統）正在向 OpenAI 查詢天氣與綜合建議，請稍候..."
                    st.session_state.chat_messages.append({"role": "assistant", "content": temp_marker})
                    with st.spinner("正在向 OpenAI 查詢天氣與綜合建議..."):
                        try:
                            # 傳遞 user_input 與可選的區域覆寫給 system.py
                            system_py_main_response = get_system_weather_data(user_input, sel_dist)
                        except Exception as e:
                            st.error(f"呼叫 system.py 時發生錯誤: {e}")
                            system_py_main_response = "抱歉，OpenAI 進階建議系統暫時無法服務。"
                    # 對 system 回傳做清理與友善化處理，避免直接顯示 JSON/NULL
                    try:
                        system_py_main_response = _sanitize_system_response(system_py_main_response)
                    except Exception:
                        system_py_main_response = "抱歉，進階建議系統暫時無法提供可讀回覆。"
            
            # 3. 組合最終顯示的聊天回應
            final_response_elements = []

            if system_py_main_response: # 如果 system.py 有回應，將其作為主要內容
                final_response_elements.append(system_py_main_response)
            
            # 如果 system.py 未被呼叫或呼叫失敗，且 Streamlit 內部機器人有關於景點/餐廳/住宿的回應，則使用它
            # sl_query_type 來自 Streamlit 內部機器人的判斷
            if not system_py_main_response or "錯誤：" in system_py_main_response or "無法導入" in system_py_main_response:
                if sl_query_type in ["attraction", "restaurant", "hotel"]:
                    # sanitize sl_resp_text before using
                    safe_sl_resp = _sanitize_system_response(sl_resp_text) if sl_resp_text is not None else None
                    if safe_sl_resp and ("建議關注的旅遊景點：" in safe_sl_resp or "為您找到以下位於" in safe_sl_resp):
                        # 避免重複添加，檢查 final_response_elements 是否已包含類似內容 (這裡簡化，直接添加)
                        if not any("建議關注的旅遊景點：" in str(part) for part in final_response_elements) and \
                           not any("為您找到以下位於" in str(part) for part in final_response_elements):
                           final_response_elements.append(safe_sl_resp)
                else:
                    # 非景點類型或 fallback，確保 sl_resp_text 是可讀字串再加入
                    safe_sl_resp = _sanitize_system_response(sl_resp_text) if sl_resp_text is not None else None
                    if safe_sl_resp and not final_response_elements:
                        final_response_elements.append(safe_sl_resp)


            if not final_response_elements: # 最終的備援訊息
                 final_response_elements.append("抱歉，目前無法處理您的請求，請再試一次或調整您的問題。")

            # 4. 附加 Streamlit 這邊生成的 CCTV 列表和圖例 (總是執行，除非 sl_cctv_names 為空)
            if sl_cctv_names:
                cctv_legend_text = "\n\n**🎯 建議關注的監視器位置：** (地圖上紅色圖釘標記)"
                if st.session_state.get('show_all_cctvs_checkbox_value', False):
                    cctv_legend_text += " 其他監視器以藍色相機標記。"
                final_response_elements.append(cctv_legend_text) 
                
                cctv_list_for_display = []
                for i, cctv_name in enumerate(sl_cctv_names[:5], 1):
                    cctv_list_for_display.append(f"{i}. {cctv_name}")
                if len(sl_cctv_names) > 5:
                    cctv_list_for_display.append(f"...以及其他 {len(sl_cctv_names) - 5} 個位置")
                final_response_elements.append("\n".join(cctv_list_for_display))

            final_display_text = "\n".join(filter(None, final_response_elements)).strip() # 移除開頭可能多餘的換行

            # 如果最終顯示文字僅為空字串或只有空白，改為友善的備援訊息，避免在 UI 顯示空白訊息
            if not final_display_text or not final_display_text.strip():
                final_display_text = "抱歉，進階建議系統暫時無法提供內容，請稍後或調整查詢。"

            # 如果之前插入了 temp_marker，嘗試以最終回覆取代該佔位訊息，避免重複訊息
            try:
                if temp_marker:
                    replaced = False
                    for i in range(len(st.session_state.chat_messages) - 1, -1, -1):
                        if st.session_state.chat_messages[i].get("role") == "assistant" and st.session_state.chat_messages[i].get("content") == temp_marker:
                            st.session_state.chat_messages[i]["content"] = final_display_text
                            replaced = True
                            break
                    if not replaced:
                        st.session_state.chat_messages.append({"role": "assistant", "content": final_display_text})
                else:
                    st.session_state.chat_messages.append({"role": "assistant", "content": final_display_text})
            except Exception:
                # 如果對 session_state 操作失敗，退回到直接 append 的保險作法
                st.session_state.chat_messages.append({"role": "assistant", "content": final_display_text})

            st.rerun()

    with col1: # 地圖顯示部分 (與前一版本相同)
        st.header("🗺️ 地圖")
        with st.expander("地圖控制選項", expanded=True):
            st.session_state.show_all_cctvs_checkbox_value = st.checkbox(
                "在地圖上顯示所有CCTV", value=st.session_state.show_all_cctvs_checkbox_value, key="show_all_cctvs_widget")
            if st.button("清除地圖重點標記與推薦景點"):
                st.session_state.highlighted_cctvs, st.session_state.recommended_attractions_on_map = [], None; st.rerun()
        
        cctv_df_map: pd.DataFrame
        if st.session_state.show_all_cctvs_checkbox_value: 
            cctv_df_map = st.session_state.cctv_manager.cctv_data
        else:
            if st.session_state.highlighted_cctvs and not st.session_state.cctv_manager.cctv_data.empty :
                cctv_df_map = st.session_state.cctv_manager.cctv_data[st.session_state.cctv_manager.cctv_data['name'].isin(st.session_state.highlighted_cctvs)]
            else: 
                cctv_df_map = pd.DataFrame(columns=(st.session_state.cctv_manager.cctv_data.columns if not st.session_state.cctv_manager.cctv_data.empty else ['name', 'lat', 'long', 'link', 'geohash']))
        
        attr_map = st.session_state.recommended_attractions_on_map
        current_map_obj = st.session_state.cctv_manager.create_map(
            cctv_data_to_plot=cctv_df_map, highlighted_cctv_names=st.session_state.highlighted_cctvs, attractions_df=attr_map)
        
        map_display_data = st_folium(current_map_obj, width="100%", height=550, returned_objects=["last_clicked"])
        if map_display_data and map_display_data["last_clicked"]:
            clk_lat, clk_lng = map_display_data["last_clicked"]["lat"], map_display_data["last_clicked"]["lng"]
            near_cctvs = st.session_state.cctv_manager.get_nearby_cctvs(clk_lat, clk_lng, radius_km=0.5)
            if near_cctvs:
                closest = near_cctvs[0]
                st.info(f"📍 您點擊位置附近 ({closest['distance']:.2f} km) 的監視器: **{closest['name']}**")
                try: st.image(closest['link'], caption=f"即時影像 - {closest['name']}", width=350)
                except Exception: st.warning("無法載入即時影像，請檢查網路連線或監視器狀態。")
            else: st.info("您點擊的位置附近 0.5km 內沒有找到CCTV。")


if __name__ == "__main__":
    main()