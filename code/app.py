"""Streamlit Web UI for the weather-aware Taiwan travel agent."""

import asyncio
import sqlite3
import sys
import uuid
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st
from langgraph.checkpoint.memory import MemorySaver

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collector.db import DB_PATH  # noqa: E402

from agent import WeatherAgent  # noqa: E402


st.set_page_config(
    page_title="台灣旅遊天氣助理",
    page_icon="🌤️",
    layout="wide",
)

# ponytail: 第三輪重新設計時查了artifact-design skill，發現前兩輪(漸層卡片+
# 玻璃感側邊欄)落在它點名的「AI通用設計套路」裡(Inter字體、rounded-lg到處用、
# 淺色漸層hero)。改參考真實網站抓到的具體設計token(Windy.com/Linear.app/
# Kayak)定調成「氣象觀測儀面板」概念——小而一致的圓角(非到處rounded-lg)、
# 數字用等寬字體呈現「讀數」感(氣象站真的是這樣顯示數據的，不是裝飾)、陰影
# 克制只留給真正要突出的元素(參考Linear幾乎零陰影的節制)。
# 色票：主要accent沿用地圖「住宿」橘(#eb6834)+「景點」藍(#2a78d6)+「餐廳」
# 綠(#1baf7a)三色並存(不是單一「孤色」，因為地圖本來就需要三色共存，這點
# 刻意跟「近黑+單一孤色」的AI套路做出區隔)。
# 第四輪(隔天)使用者要求改回淺色但不要白色——因為是token-based架構，只改
# :root色票值，結構完全不動；IBM Plex Sans/Mono取代Inter——中文不受影響(CJK自動fallback回系統字體)，
# 差別只在拉丁字母/數字的字體質感，IBM Plex Mono用在所有數據讀數上。
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    /* 淺色系但不是白色——canvas用帶一點灰綠調的石板色，跟地圖「接近黑」
       的deck.gl底圖剛好形成對比，而不是另一片純白蓋過去 */
    --wl-ink: #eef1f0;
    --wl-panel: #e3e8e6;
    --wl-panel-raised: #f5f7f6;
    --wl-hair: rgba(27, 36, 32, 0.12);
    --wl-text: #1b2420;
    --wl-muted: #5b6b62;
    --wl-ember: #eb6834;
    --wl-ember-glow: rgba(235, 104, 52, 0.28);
    --wl-sky: #2a78d6;
    --wl-sky-soft: rgba(42, 120, 214, 0.10);
    --wl-leaf: #0f9566;
    --wl-leaf-soft: rgba(15, 149, 102, 0.10);
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', -apple-system, 'PingFang TC', 'Microsoft JhengHei', sans-serif;
}

/* 所有數字(金額/筆數/氣溫)走等寬字體——氣象儀器讀數的質感，tabular-nums
   讓直排數字對得齊，不是裝飾 */
[data-testid="stMetricValue"],
[data-testid="stChatInput"] textarea::placeholder {
    font-family: 'IBM Plex Mono', 'SFMono-Regular', monospace;
    font-variant-numeric: tabular-nums;
}

.stApp {
    background: var(--wl-ink);
}
body, .stApp, [data-testid="stAppViewContainer"] {
    color: var(--wl-text);
}

.block-container {
    max-width: 1440px;
    margin: 0 auto;
    padding-top: 2.5rem !important;
}

/* ---------- Hero：左對齊的儀表板標頭，不是置中漸層卡片 ---------- */
.wl-hero {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 1.5rem;
    padding-bottom: 1.25rem;
    margin-bottom: 1.75rem;
    border-bottom: 1px solid var(--wl-hair);
}
.wl-hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--wl-ember);
}
.wl-hero-title {
    font-size: 1.9rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: var(--wl-text);
    line-height: 1.2;
    margin: 0.25rem 0 0.3rem;
}
.wl-hero-subtitle {
    font-size: 0.88rem;
    color: var(--wl-muted);
    max-width: 34rem;
}
.wl-hero-status {
    flex: none;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem;
    color: var(--wl-muted);
    text-align: right;
    line-height: 1.6;
}
.wl-hero-status b {
    color: var(--wl-leaf);
    font-weight: 600;
}

h2, h3 {
    font-weight: 600 !important;
    letter-spacing: -0.005em;
    color: var(--wl-text) !important;
}
p, span, label, div {
    color: var(--wl-text);
}

/* ---------- 側邊欄：跟主面板同一塊深色儀表板，用hairline分隔不是浮動卡片 ---------- */
[data-testid="stSidebar"] {
    background: var(--wl-panel);
    border-right: 1px solid var(--wl-hair);
}
[data-testid="stSidebarHeader"] {
    min-height: 0 !important;
    height: auto !important;
    padding: 0.75rem 0 0 0 !important;
}
[data-testid="stLogoSpacer"] {
    display: none !important;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--wl-muted) !important;
}
[data-testid="stSidebar"] hr {
    border-color: var(--wl-hair);
}

/* ---------- Tabs：底線式分段，寬度貼合內容，不是全寬膠囊 ---------- */
[data-baseweb="tab-list"] {
    gap: 1.5rem;
    border-bottom: 1px solid var(--wl-hair);
}
[data-baseweb="tab"] {
    font-weight: 500;
    color: var(--wl-muted);
    padding: 0.5rem 0.1rem !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: var(--wl-ember) !important;
}
[data-baseweb="tab-highlight"] {
    background-color: var(--wl-ember) !important;
    height: 2px !important;
}

/* ---------- stat tile：小圓角、hairline邊框，陰影只用在ember強調色上 ---------- */
[data-testid="stMetric"] {
    background: var(--wl-panel-raised);
    border: 1px solid var(--wl-hair) !important;
    border-radius: 10px !important;
    padding: 0.9rem 1rem !important;
}
/* white-space:nowrap+overflow:visible(不是normal)——之前用normal讓長數字
   在很窄的欄位下一個字一行疊起來(2026-10-06，3欄stat tile在窄螢幕發現)，
   改成不換行、允許溢出，數字寧可極端窄時視覺上超出卡片邊界，也不要疊成
   直的看不懂或被裁切看不到。 */
[data-testid="stMetricValue"] {
    font-weight: 600 !important;
    color: var(--wl-text);
    font-size: 1.5rem !important;
    white-space: nowrap !important;
    overflow: visible !important;
    line-height: 1.25 !important;
}
/* 窄螢幕下地圖頁籤的3欄stat tile/類別勾選列可用寬度太窄，連single nowrap
   行都會被壓到近乎0px寬而整個視覺跑掉(2026-10-06實測過，光靠white-space
   調整救不了)。只對這兩個keyed container生效(.st-key-...)，不動到主欄位
   (對話/地圖並排)的5:7比例分配——直接用`:has()`選中父層`stHorizontalBlock`
   更保險，避免class只掛在外層div、沒影響到實際排版的flex容器。 */
.st-key-map_metrics_row [data-testid="stHorizontalBlock"],
.st-key-map_category_row [data-testid="stHorizontalBlock"] {
    flex-wrap: wrap !important;
}
.st-key-map_metrics_row [data-testid="stColumn"],
.st-key-map_category_row [data-testid="stColumn"] {
    min-width: 150px !important;
    flex: 1 1 150px !important;
}

[data-testid="stMetricLabel"] {
    color: var(--wl-muted) !important;
    font-weight: 500 !important;
    font-size: 0.76rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stMetricDelta"] {
    font-family: 'IBM Plex Mono', monospace;
}

/* ---------- 對話泡泡：小圓角、助理訊息用左側ember細線標示「這是系統回應」 ---------- */
[data-testid="stChatMessage"] {
    border-radius: 10px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.6rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: var(--wl-sky-soft);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: var(--wl-panel-raised);
    border: 1px solid var(--wl-hair);
    border-left: 2px solid var(--wl-ember);
}

/* ---------- 輸入框 ---------- */
[data-testid="stChatInput"] {
    background: var(--wl-panel-raised) !important;
    border: 1px solid var(--wl-hair) !important;
    border-radius: 10px !important;
}
[data-testid="stChatInput"] textarea {
    color: var(--wl-text) !important;
}
[data-testid="stTextInput"] input {
    background: var(--wl-panel-raised) !important;
    border-radius: 8px !important;
    border-color: var(--wl-hair) !important;
    color: var(--wl-text) !important;
}
.stButton button, [data-testid="stChatInput"] button {
    border-radius: 8px !important;
}

/* ---------- info/alert 提示框：左側色條代替整塊底色，克制一點 ---------- */
[data-testid="stAlertContainer"] {
    background: var(--wl-panel-raised) !important;
    border: 1px solid var(--wl-hair) !important;
    border-left: 2px solid var(--wl-sky) !important;
    border-radius: 8px !important;
}
[data-testid="stAlertContainer"] p {
    color: var(--wl-text) !important;
}

/* ---------- 地圖panel ---------- */
[data-testid="stDeckGlJsonChart"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--wl-hair);
}

/* checkbox/multiselect文字顏色跟著深色主題走 */
[data-testid="stWidgetLabel"] p {
    color: var(--wl-text) !important;
}
</style>
"""


CATEGORY_CONFIG = {
    "景點": {
        "table": "attractions",
        "light": "#2a78d6",
        "dark": "#3987e5",
    },
    "住宿": {
        "table": "lodgings",
        "light": "#eb6834",
        "dark": "#d95926",
    },
    "餐廳": {
        "table": "restaurants",
        "light": "#1baf7a",
        "dark": "#199e70",
    },
}

DOC_TYPE_TO_LABEL = {
    "attraction": "景點",
    "lodging": "住宿",
    "restaurant": "餐廳",
}

# 地圖上「這是推薦地點」的badge固定用品牌ember色——不管哪個類別被標示，
# 推薦狀態都用同一個顏色語言表達；類別本身仍由內層圓點的fill color呈現
# (2026-10-07改版，詳見render_map_tab()的說明)。
HIGHLIGHT_BADGE_COLOR = "#eb6834"


@st.cache_data(show_spinner=False)
def load_map_data() -> dict[str, pd.DataFrame]:
    """Read the three map datasets once per Streamlit data-cache lifetime."""
    datasets = {}
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        for label, config in CATEGORY_CONFIG.items():
            table = config["table"]
            datasets[label] = pd.read_sql_query(
                f"""
                SELECT name, city, district, latitude, longitude
                FROM {table}
                WHERE latitude IS NOT NULL AND longitude IS NOT NULL
                """,
                conn,
            ).fillna({"city": "", "district": ""})
    return datasets


@st.cache_data(show_spinner=False)
def load_highlighted_points(doc_ids: tuple[str, ...]) -> pd.DataFrame:
    """Resolve RAG document ids to map points without sharing mutable request state."""
    rows = []
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        for doc_id in doc_ids:
            if ":" not in doc_id:
                continue
            doc_type, original_id = doc_id.split(":", 1)
            label = DOC_TYPE_TO_LABEL.get(doc_type)
            if not label or not original_id:
                continue
            table = CATEGORY_CONFIG[label]["table"]
            row = conn.execute(
                f"""
                SELECT name, city, district, latitude, longitude
                FROM {table}
                WHERE id = ? AND latitude IS NOT NULL AND longitude IS NOT NULL
                """,
                (original_id,),
            ).fetchone()
            if row:
                rows.append((*row, label, doc_id))
    return pd.DataFrame(
        rows,
        columns=["name", "city", "district", "latitude", "longitude", "type_label", "doc_id"],
    ).fillna({"city": "", "district": ""})


def theme_palette() -> dict[str, str]:
    """Use Streamlit's active theme when exposed; app defaults to light (config.toml base="light")."""
    # Streamlit 1.64 exposes st.context.theme.type. The fallback intentionally
    # stays simple for compatibility with older versions.
    try:
        theme_type = st.context.theme.type
    except (AttributeError, RuntimeError):
        theme_type = "light"
    variant = "dark" if theme_type == "dark" else "light"
    return {label: config[variant] for label, config in CATEGORY_CONFIG.items()}


def hex_to_rgba(hex_color: str, alpha: int = 190) -> list[int]:
    value = hex_color.lstrip("#")
    return [int(value[index : index + 2], 16) for index in (0, 2, 4)] + [alpha]


async def ask_agent(api_key: str, user_text: str) -> tuple[str, float, list[str]]:
    # Streamlit reruns the script for every message. Reusing MemorySaver keeps
    # conversation state while a fresh MCP subprocess is opened per turn. If
    # that startup cost becomes material, the upgrade path is a persistent
    # background event loop/thread that owns one long-lived WeatherAgent.
    async with WeatherAgent(
        openai_api_key=api_key,
        checkpointer=st.session_state.checkpointer,
    ) as agent:
        reply, doc_ids = await agent.ask(st.session_state.thread_id, user_text)
        return reply, agent.cost_so_far(st.session_state.thread_id), doc_ids


def initialize_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "checkpointer" not in st.session_state:
        st.session_state.checkpointer = MemorySaver()
    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.0
    if "last_turn_cost" not in st.session_state:
        st.session_state.last_turn_cost = 0.0
    if "last_matched_doc_ids" not in st.session_state:
        st.session_state.last_matched_doc_ids = []


CHAT_PANEL_HEIGHT = 560


def render_chat_tab(api_key: str, key_is_valid: bool) -> None:
    st.subheader("旅遊與天氣對話")
    st.caption("可詢問台灣景點、住宿、餐廳與即時天氣資訊。對話只保留在目前瀏覽器工作階段。")

    # 使用者要求「像聊天室一樣」——訊息區固定高度、內部自己捲動，輸入框釘在
    # 固定位置，不要讓整頁隨對話筆數無限往下延伸(不然地圖欄位會被拋在後面，
    # 失去並排對齊的意義)。st.container(height=...)搭配st.chat_input兩者都
    # 放在同一個column、但input寫在container外面，這是Streamlit官方「inline
    # chat」的寫法，input就會貼齊在這個固定高度訊息框的正下方，不會被推到
    # 整個瀏覽器頁面最底部。
    messages_box = st.container(height=CHAT_PANEL_HEIGHT)
    with messages_box:
        if not st.session_state.messages:
            st.info(
                "試著問：「日月潭有哪些景點？」、「台南有適合住宿又方便吃美食的地方嗎？」"
                "或「台北現在天氣如何？」"
            )
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    user_text = st.chat_input(
        "例如：南投日月潭有什麼景點？",
        disabled=not key_is_valid,
        key="travel_chat_input",
    )
    if not user_text:
        return

    st.session_state.messages.append({"role": "user", "content": user_text})
    with messages_box:
        with st.chat_message("user"):
            st.markdown(user_text)

        with st.chat_message("assistant"):
            with st.spinner("正在查詢旅遊與天氣資料…"):
                cost_before_turn = st.session_state.total_cost
                st.session_state.last_turn_cost = 0.0
                try:
                    reply, cost, doc_ids = asyncio.run(ask_agent(api_key, user_text))
                except Exception:  # noqa: BLE001 — never expose service internals or secrets in the UI
                    reply = "系統暫時無法完成查詢，請稍後再試。"
                    st.session_state.last_matched_doc_ids = []
                    st.error(reply)
                else:
                    st.markdown(reply)
                    st.session_state.total_cost = cost
                    st.session_state.last_turn_cost = cost - cost_before_turn
                    st.session_state.last_matched_doc_ids = doc_ids

    st.session_state.messages.append({"role": "assistant", "content": reply})


def render_map_tab() -> None:
    st.subheader("旅遊地圖")
    st.caption("瀏覽資料庫中的景點、住宿與餐廳；最近對話命中的地點會以橘色光暈標示。")

    datasets = load_map_data()
    highlighted = load_highlighted_points(tuple(st.session_state.last_matched_doc_ids))
    # key="map_metrics_row"讓這個container有穩定的`.st-key-map_metrics_row`
    # CSS class，CUSTOM_CSS裡窄螢幕換行的規則只套用在這個class下面，不會
    # 影響到主欄位(對話/地圖並排)的5:7比例分配。
    metrics_container = st.container(key="map_metrics_row")
    cities = sorted(
        {
            city
            for frame in datasets.values()
            for city in frame["city"].unique().tolist()
            if city
        }
    )
    selected_cities = st.multiselect(
        "城市",
        options=cities,
        placeholder="全部城市",
        help="未選擇時顯示全台資料。",
    )

    st.markdown("**類別**")
    with st.container(key="map_category_row"):
        columns = st.columns(3)
        enabled = {}
        for column, label in zip(columns, CATEGORY_CONFIG):
            with column:
                enabled[label] = st.checkbox(label, value=True, key=f"map_category_{label}")

    palette = theme_palette()
    visible_labels = [label for label, is_enabled in enabled.items() if is_enabled]

    visible_highlighted = highlighted[highlighted["type_label"].isin(visible_labels)]
    if selected_cities:
        visible_highlighted = visible_highlighted[
            visible_highlighted["city"].isin(selected_cities)
        ]
    if st.session_state.last_matched_doc_ids:
        st.caption(
            f"📍 已在地圖上標示 {len(visible_highlighted)} 個與最近對話相關的地點"
        )

    filtered_frames = {}
    for label, frame in datasets.items():
        if not enabled[label]:
            filtered_frames[label] = frame.iloc[0:0]
        elif selected_cities:
            filtered_frames[label] = frame[frame["city"].isin(selected_cities)]
        else:
            filtered_frames[label] = frame

    with metrics_container:
        metric_columns = st.columns(3)
        for column, (label, metric_label) in zip(
            metric_columns,
            (
                ("景點", "景點數量"),
                ("住宿", "住宿數量"),
                ("餐廳", "餐廳數量"),
            ),
        ):
            with column:
                st.metric(metric_label, f"{len(filtered_frames[label]):,}", border=True)

    if visible_labels:
        legend = "&nbsp;&nbsp;&nbsp;".join(
            (
                f'<span style="display:inline-flex;align-items:center;gap:0.35rem">'
                f'<span aria-hidden="true" style="width:0.75rem;height:0.75rem;'
                f'border-radius:50%;background:{palette[label]};display:inline-block"></span>'
                f"{label}</span>"
            )
            for label in visible_labels
        )
        st.markdown(legend, unsafe_allow_html=True)

    layers = []
    visible_frames = []
    for label in visible_labels:
        frame = filtered_frames[label]
        if frame.empty:
            continue
        frame = frame.assign(type_label=label)
        visible_frames.append(frame)
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=frame,
                id=f"{CATEGORY_CONFIG[label]['table']}-layer",
                get_position="[longitude, latitude]",
                get_fill_color=hex_to_rgba(palette[label]),
                get_radius=80,
                radius_min_pixels=2,
                radius_max_pixels=8,
                pickable=True,
                auto_highlight=True,
            )
        )

    if not visible_highlighted.empty:
        visible_highlighted = visible_highlighted.assign(
            fill_color=visible_highlighted["type_label"].map(
                lambda label: hex_to_rgba(palette[label])
            )
        )
        # 原本只是把同色的點放大+白色外框，在淺色底圖(positron)上白框幾乎
        # 看不見，跟一般點位太像(2026-10-07使用者回報)。改成兩層疊加的
        # 「badge」畫法：底下一層ember品牌色的半透明光暈(用尺寸與透明度
        # 區分類別點，不依賴outline在特定底圖上的對比度)，上面疊一般大小+
        # ember外框的同類別色點——類別色還是在，但「這是推薦的」這件事改用
        # 品牌色光暈獨立表達，不管底圖深淺都看得到。
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=visible_highlighted,
                id="recommended-halo-layer",
                get_position="[longitude, latitude]",
                get_fill_color=[*hex_to_rgba(HIGHLIGHT_BADGE_COLOR)[:3], 70],
                get_radius=520,
                radius_min_pixels=16,
                radius_max_pixels=34,
                stroked=False,
                pickable=False,
            )
        )
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=visible_highlighted,
                id="recommended-highlight-layer",
                get_position="[longitude, latitude]",
                get_fill_color="fill_color",
                get_line_color=[*hex_to_rgba(HIGHLIGHT_BADGE_COLOR)[:3], 255],
                get_radius=180,
                radius_min_pixels=7,
                radius_max_pixels=18,
                line_width_min_pixels=3,
                stroked=True,
                pickable=True,
                auto_highlight=True,
            )
        )

    if not layers:
        st.info("目前的篩選條件沒有可顯示的地點。")
        return

    all_visible = (
        pd.concat(visible_frames, ignore_index=True)
        if visible_frames
        else visible_highlighted.copy()
    )
    filtered_to_cities = bool(selected_cities)
    view_points = visible_highlighted if not visible_highlighted.empty else all_visible
    view_state = pdk.ViewState(
        latitude=float(view_points["latitude"].mean()),
        longitude=float(view_points["longitude"].mean()),
        zoom=10.5 if not visible_highlighted.empty else (8 if filtered_to_cities else 6.5),
        pitch=0,
    )
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=pdk.map_styles.LIGHT,
        tooltip={
            "html": "<b>{name}</b><br/>{city} {district}<br/>{type_label}",
            "style": {"backgroundColor": "#1b2420", "color": "white"},
        },
    )
    st.pydeck_chart(deck, height=520)
    st.caption(f"目前顯示 {len(all_visible):,} 個地點。")


def main() -> None:
    initialize_session()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="wl-hero">
            <div>
                <p class="wl-hero-eyebrow">TAIWAN TRAVEL · WEATHER STATION</p>
                <p class="wl-hero-title">台灣旅遊天氣助理</p>
                <p class="wl-hero-subtitle">結合景點／住宿／餐廳資料庫與中央氣象署即時觀測，規劃一趟不被天氣打亂的行程</p>
            </div>
            <div class="wl-hero-status">
                資料來源 · 觀光署 + 中央氣象署<br/>
                狀態 · <b>● ONLINE</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.header("帳號設定")
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        key="openai_api_key",
        help="只保存在目前瀏覽器工作階段，不會寫入磁碟或傳送到 Langfuse。",
    )
    key_is_valid = bool(api_key) and api_key.startswith("sk-")
    if not api_key:
        st.sidebar.info("請輸入自己的 OpenAI API Key 才能開始對話。")
    elif not key_is_valid:
        st.sidebar.error("API Key 格式不正確，必須以 sk- 開頭。")

    st.sidebar.divider()
    st.sidebar.header("使用量")
    cost_placeholder = st.sidebar.empty()

    # 使用者要求對話跟地圖同一頁並排顯示，方便邊看文字邊對照地圖位置，
    # 不要再切頁籤——地圖欄位較寬，因為地圖本身需要的視覺空間比文字對話多。
    col_chat, col_map = st.columns([5, 7], gap="large")
    with col_chat:
        render_chat_tab(api_key, key_is_valid)
    with col_map:
        render_map_tab()

    cost_placeholder.metric(
        label="本次對話花費",
        value=f"US${st.session_state.total_cost:.4f}",
        delta=(
            f"+US${st.session_state.last_turn_cost:.4f}"
            if st.session_state.messages
            else None
        ),
        delta_color="inverse",
        border=True,
    )


if __name__ == "__main__":
    main()
