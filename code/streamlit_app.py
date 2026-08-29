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
import time

import repository
import orchestrator

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

    # 「新竹」這個分組底下混了新竹市跟新竹縣的地名（竹北/湖口/竹東其實都在新竹縣，
    # 不是新竹市）——CITY_NAME_MAP是整組對應同一個縣市，會把這幾個地名的天氣
    # 查到新竹市去。其餘分組（台中/台北/台南/高雄）list裡的地名都真的屬於同一個
    # 縣市，沒有這種混雜問題，不用比照處理。
    DISTRICT_CITY_OVERRIDE = {"竹北": "新竹縣", "湖口": "新竹縣", "竹東": "新竹縣"}

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

        # 審查發現的真bug：像「大安區」（台中/台北都有）、「北區」「東區」「南區」
        # （台中/台南都有）這些同名跨縣市的行政區，原本不管訊息裡有沒有明確提到
        # 縣市，一律照city_districts_map的字典insertion順序（台中最先）找到誰就
        # 用誰——導致「台北大安區」被誤判成台中的大安區、「台南北區」被誤判成
        # 台中的北區。修法：先掃一次訊息裡有沒有明確提到某個縣市別名，有的話
        # 該縣市的行政區清單優先比對，比對不到才退回原本的字典順序。
        mentioned_city_key = None
        for city_k, dist_list in city_districts_map.items():
            for alias in dist_list:
                is_alias = (alias == city_k or alias == city_k + "市" or alias == city_k.replace("台", "臺"))
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
            resolved_city = self.DISTRICT_CITY_OVERRIDE.get(target_district) or self.CITY_NAME_MAP.get(target_region, "臺中市")
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

    st.set_page_config(page_title="個人化氣象決策系統", page_icon="🌤️", layout="wide", initial_sidebar_state="expanded")
    load_css()

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

            # LangGraph supervisor + 兩個 worker（WeatherChatbot 輕量路徑／system.py
            # OpenAI 進階路徑）+ 組合節點，取代原本這裡一大段手動 if/else 拼接。
            # 見 ROADMAP.md Phase 2、code/orchestrator.py。key當參數傳進closure，
            # 不靠全域狀態——見上面effective_api_key的說明。
            with st.spinner("正在處理您的請求..."):
                result = orchestrator.run_orchestrator(
                    user_input=user_input,
                    chatbot=st.session_state.chatbot,
                    system_weather_fn=lambda prompt, override: get_system_weather_data(
                        prompt, override, api_key=effective_api_key
                    ),
                    override_district=sel_dist,
                    api_key_available=api_key_available,
                    show_all_cctvs=st.session_state.get('show_all_cctvs_checkbox_value', False),
                )

            st.session_state.highlighted_cctvs = result.get("sl_cctv_names") or []
            st.session_state.recommended_attractions_on_map = result.get("sl_attr_df")

            final_display_text = result.get("final_display_text") or \
                "抱歉，目前無法處理您的請求，請再試一次或調整您的問題。"

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