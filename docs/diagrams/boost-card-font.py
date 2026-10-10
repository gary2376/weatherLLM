#!/usr/bin/env python3
"""archify沒有schema欄位可調流程圖節點文字的字體大小(見archify/#282,
upstream未解，節點文字受desktop-reader viewBox自動縮放影響)。每次跑
`archify deliver`重新產生HTML後，用這支腳本直接調大流程圖節點
(label/sublabel/tag)內嵌的SVG font-size屬性。

下方info卡片維持archify預設大小(不套用override)——原本試過放大，
後來確認使用者要放大的是流程圖節點本身，不是下面的文字卡片。

deliver會整份重新產生檔案，這個override不是一次性的，每次重新
deliver都要重跑本腳本。

用法：
  python3 boost-card-font.py <html_path>
  (預設 <html_path> = agent-workflow.html，跟這支腳本同資料夾)
"""
import re
import sys
from pathlib import Path

MARKER = '<!-- wl-node-font-boost -->'

# 2026-10-08第三輪：使用者反應「不同流程間的字」(edge label，例如節點
# 連線上的「測試通過」「送出查詢+Key」)一樣小到模糊——這些字先前完全沒被
# 這支腳本碰過(舊版regex只鎖node label/sublabel/tag三種data-attribute，
# edge label/lane標題/legend都是沒有data-*屬性的裸<text font-size="...">，
# 漏網)。使用者直接要求「字體全部放大兩倍」，所以改成不分類別、對SVG裡
# 所有<text font-size="...">統一乘2倍，比針對不同文字類型各自調倍率更
# 簡單也不會再漏掉新的文字類型。
GLOBAL_TEXT_SCALE = 2.8
TEXT_FONT_SIZE_PATTERN = re.compile(r'(<text\b[^>]*?font-size=")([0-9.]+)(")')

# 2026-10-08第四輪：用Playwright實際量測getBBox()後發現，放大font-size後
# node label(粗體那行)的文字上緣統一超出框頂約7.8px——baseline的y座標是
# archify deliver時依「原始小字體」的ascent算好的，字放大後ascent跟著變高，
# 但y沒有跟著往下補，導致字的上緣頂出框外。水平方向經量測其實沒問題
# (是視覺上看起來貼邊但其實有margin)，只有這個垂直方向的系統性小溢出，
# 用肉眼看截圖沒抓到，是寫量測腳本直接比對text跟rect的getBBox()才抓到的。
# 把label往下推10px修完頂部溢出後，量測又發現label跟下面的sublabel(context)
# 重疊21.8px(label變高吃掉兩行間原本只有17px的小間距)——sublabel/tag
# 也要跟著往下推，兩個問題(頂部溢出、行間重疊)要一起修，只動一個會製造
# 另一個。
# 2026-10-08第六輪：卡片高度調到140後，使用者要求label+sublabel這兩行要
# 在卡片裡垂直置中(不要貼在上緣)。量過字型的ascent/height比例(≈1.034/
# 1.292倍font-size，CJK字體量出來的經驗值)，但卡片下緣還有一行tag文字
# (已驗證(...)那種)原本就固定貼在box底部附近——如果把label+sublabel用
# 整個box高度置中，會跟tag黏在一起甚至重疊。改成只在「box頂部到tag預留
# 區之前」這段可用空間(扣掉tag+安全距離，算出來約box高度的70%)裡置中，
# tag維持原本archify算好的位置不動，兩者才不會打架。
LABEL_Y_SHIFT = 22
CONTEXT_Y_SHIFT = 43
LABEL_Y_PATTERN = re.compile(r'(<text data-node-label=""[^>]*?\sy=")([0-9.]+)(")')
CONTEXT_Y_PATTERN = re.compile(r'(<text data-detail="context"[^>]*?\sy=")([0-9.]+)(")')
FINE_Y_PATTERN = re.compile(r'(<text data-detail="fine"[^>]*?\sy=")([0-9.]+)(")')

# 2026-10-08第五輪：字體拉到2.8倍後，量測發現5個edge label(都是相鄰node間
# 短距離連線上的字，本來就沒什麼縫可用)跟旁邊的node框重疊——這幾個縫隙
# 窄到放不下2.8倍的字。第一次嘗試單獨把這幾個字縮小，使用者反對——字縮
# 到快看不見，寧可移位不要縮小。改成維持跟其他字同樣的2.8倍，改用垂直
# 位移(往上或往下，視哪個方向有空白)把它們推出node框的範圍。每個label
# 的位移方向/量是量測原始重疊量後手動決定的(見下方TIGHT_EDGE_LABEL_SHIFT)。
TIGHT_EDGE_LABEL_SHIFT = {
    "測試通過": -80,
    "超標": -73,
    "記錄步驟+中斷": -80,
    "整理成答案": 26,
    "呼叫工具前檢查": 28,
}

# 2026-10-08第六輪：使用者要求「虛線上的描述文字」(edge label，連線中段
# 那些字，例如「送出查詢+Key」)再加大。這些字跟其他文字一樣已經在
# GLOBAL_TEXT_SCALE吃過2.8倍，但base font-size(8px)比node label(11px)
# 小，放大後還是相對不明顯。這裡另外抓一層單獨加碼：edge label在markup
# 上的特徵是「沒有font-weight屬性(node label/lane/phase header都有)、
# 也沒有data-detail屬性(node的context/fine都有)」，用這個組合排除法鎖定，
# 從目前2.8倍再乘1.3倍到3.64倍。
EDGE_LABEL_EXTRA_SCALE = 1.3

# 2026-10-08第七輪：使用者反應「threading.Lock 併發保護」這個group框的
# 標題字太小——它跟lane/phase header一樣有font-weight="600"，所以沒被
# 上面的EDGE_LABEL_PATTERN排除法抓到(那組特地排除font-weight屬性，避免
# 誤改lane/phase header)。這裡用精確文字比對單獨加碼，不影響lane/phase
# header(目前28px/22.4px已經夠大，使用者沒抱怨那兩個)。
GROUP_LABEL_EXTRA_SCALE = 1.4
GROUP_LABEL_TEXT = "🔒 threading.Lock 併發保護"
EDGE_LABEL_PATTERN = re.compile(
    r'(<text(?![^>]*data-detail)(?![^>]*data-node-label)(?![^>]*font-weight)[^>]*?font-size=")([0-9.]+)(")'
)

MUTED_CONTRAST_CSS = """
<style>
[data-theme="light"] .t-muted { fill: #334155 !important; }
[data-theme="dark"] .t-muted, :root:not([data-theme]) .t-muted { fill: #cbd5e1 !important; }
</style>
"""


# Legend區塊的項目間距是runtime JS(layoutLegendEntry等)依實際量到的文字
# 寬度動態算的，不是寫死的座標——單純把font-size屬性加倍，量測基準被打亂，
# 項目會互相疊字(實測過，見對話紀錄)。放大範圍排除Legend，只動流程圖本體
# (node/edge/lane/phase)文字。
LEGEND_MARKER = "<!-- Legend -->"


# 2026-10-08第七輪：使用者發現「threading.Lock 併發保護」這個security
# group框上緣留了181px空白(label下面到RAG Tool上緣)，底部反而貼齊
# Weather Tool——archify的group frame沒有schema欄位可調padding，這段
# 空白是排版引擎自己算的固定上邊距，不會因為调yOffset而變。直接後製改寫
# 這個frame的rect座標：底部維持不動(已經貼齊，不能再往下)，頂部拉到貼近
# RAG Tool上緣(留10px呼吸空間)，label的y跟著同步往下移，維持它跟frame
# 頂部原本的相對位置(~2px)，才不會讓label懸空脫離框。
GROUP_FRAME_ID = "group-0"
GROUP_ANCHOR_NODE_ID = "rag_tool"
GROUP_TOP_PADDING = 10
GROUP_LABEL_TO_FRAME_GAP = 2
GROUP_FRAME_PATTERN = re.compile(
    r'(<rect[^>]*data-composition-frame-id="' + GROUP_FRAME_ID + r'"[^>]*?x="[0-9.]+"\s+y=")([0-9.]+)("\s+width="[0-9.]+"\s+height=")([0-9.]+)(")'
)
GROUP_LABEL_Y_PATTERN = re.compile(r'(<text x="[0-9.]+"\s+y=")([0-9.]+)("[^>]*>' + re.escape(GROUP_LABEL_TEXT) + r')')

# 2026-10-08第八輪：group frame的邊框跟edge一樣沒有schema欄位可調
# stroke-width，原本固定1px太細、虛線(4,4)間隔也小，放大整張圖後邊框
# 幾乎看不出來框了哪個範圍。一併加粗邊框+放大虛線間距讓輪廓更清楚。
GROUP_FRAME_STROKE_WIDTH = 2.5
GROUP_FRAME_STROKE_PATTERN = re.compile(
    r'(<rect[^>]*data-composition-frame-id="' + GROUP_FRAME_ID + r'"[^>]*?class="c-security-group" stroke-width=")([0-9.]+)(")'
)
GROUP_FRAME_DASH_CSS = """
<style>
.c-security-group { stroke-dasharray: 8,6 !important; }
</style>
"""


def fix_group_frame_padding(text: str) -> str:
    anchor_idx = text.find(f'data-node-id="{GROUP_ANCHOR_NODE_ID}"')
    if anchor_idx == -1:
        return text
    chunk = text[anchor_idx:anchor_idx + 600]
    rect_match = re.search(r'<rect x="[0-9.]+" y="([0-9.]+)" width="380" height="[0-9.]+"', chunk)
    frame_match = GROUP_FRAME_PATTERN.search(text)
    if not rect_match or not frame_match:
        return text
    anchor_top = float(rect_match.group(1))
    old_frame_top = float(frame_match.group(2))
    old_frame_height = float(frame_match.group(4))
    frame_bottom = old_frame_top + old_frame_height
    new_frame_top = anchor_top - GROUP_TOP_PADDING
    new_frame_height = frame_bottom - new_frame_top

    def repl_frame(m):
        return f"{m.group(1)}{round(new_frame_top, 1)}{m.group(3)}{round(new_frame_height, 1)}{m.group(5)}"
    text = GROUP_FRAME_PATTERN.sub(repl_frame, text)

    def repl_label_y(m):
        return f"{m.group(1)}{round(new_frame_top - GROUP_LABEL_TO_FRAME_GAP, 1)}{m.group(3)}"
    text = GROUP_LABEL_Y_PATTERN.sub(repl_label_y, text)

    def repl_stroke(m):
        return f"{m.group(1)}{GROUP_FRAME_STROKE_WIDTH}{m.group(3)}"
    text = GROUP_FRAME_STROKE_PATTERN.sub(repl_stroke, text)
    return text


def boost_node_text(text: str) -> str:
    def repl(m):
        new_size = round(float(m.group(2)) * GLOBAL_TEXT_SCALE, 1)
        return f"{m.group(1)}{new_size}{m.group(3)}"

    def shift_label_y(m):
        new_y = round(float(m.group(2)) + LABEL_Y_SHIFT, 1)
        return f"{m.group(1)}{new_y}{m.group(3)}"

    def shift_context_y(m):
        new_y = round(float(m.group(2)) + CONTEXT_Y_SHIFT, 1)
        return f"{m.group(1)}{new_y}{m.group(3)}"

    def repl_edge(m):
        new_size = round(float(m.group(2)) * EDGE_LABEL_EXTRA_SCALE, 1)
        return f"{m.group(1)}{new_size}{m.group(3)}"

    legend_idx = text.index(LEGEND_MARKER) if LEGEND_MARKER in text else len(text)
    head, tail = text[:legend_idx], text[legend_idx:]
    head = TEXT_FONT_SIZE_PATTERN.sub(repl, head)
    head = EDGE_LABEL_PATTERN.sub(repl_edge, head)

    group_label_pattern = re.compile(r'(<text\b[^>]*?font-size=")([0-9.]+)("[^>]*>)' + re.escape(GROUP_LABEL_TEXT) + r'(</text>)')
    def repl_group(m):
        new_size = round(float(m.group(2)) * GROUP_LABEL_EXTRA_SCALE, 1)
        return f"{m.group(1)}{new_size}{m.group(3)}{GROUP_LABEL_TEXT}{m.group(4)}"
    head = group_label_pattern.sub(repl_group, head)

    head = LABEL_Y_PATTERN.sub(shift_label_y, head)
    head = CONTEXT_Y_PATTERN.sub(shift_context_y, head)

    for label, shift in TIGHT_EDGE_LABEL_SHIFT.items():
        pattern = re.compile(r'(<text\b[^>]*?\sy=")([0-9.]+)("[^>]*>)' + re.escape(label) + r'(</text>)')
        def repl(m, shift=shift, label=label):
            new_y = round(float(m.group(2)) + shift, 1)
            return f"{m.group(1)}{new_y}{m.group(3)}{label}{m.group(4)}"
        head = pattern.sub(repl, head)
    return head + tail


def main():
    html_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "agent-workflow.html"
    text = html_path.read_text(encoding="utf-8")

    if MARKER in text:
        print(f"skip: override已存在於 {html_path}（先跑deliver重新產生乾淨檔案再套用）")
        return

    idx = text.index("</head>")
    text = text[:idx] + MARKER + "\n" + MUTED_CONTRAST_CSS + GROUP_FRAME_DASH_CSS + text[idx:]
    text = boost_node_text(text)
    text = fix_group_frame_padding(text)

    html_path.write_text(text, encoding="utf-8")
    print(f"done: 節點文字放大已套用到 {html_path}")


if __name__ == "__main__":
    main()
