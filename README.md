# 個人化氣象決策系統 (TGIS)

這個專案是一個基於 Streamlit 的互動式天氣與景點查詢系統（包含 CCTV 地圖與 RAG-LM 整合）。

Live demo
---------

- 線上已部署（Streamlit Community Cloud）：[https://weatherllm-93kwaqk8z5mtxsmvwfmrpn.streamlit.app/](https://weatherllm-93kwaqk8z5mtxsmvwfmrpn.streamlit.app/)

- 你也可以在 README 中加入一個 badge，快速讓訪客點擊啟動 demo，例如：

	[![Live demo](https://img.shields.io/badge/Streamlit-Live-brightgreen?logo=streamlit)](https://weatherllm-93kwaqk8z5mtxsmvwfmrpn.streamlit.app/)

如何把這個連結呈現在 GitHub 頁面
- 在 repository 的 README（本檔）加入 Live demo 區塊（已加入）。
- 可在 GitHub Repo 頁面右側 "About" 區域的 Website 欄位填入該 URL，會顯示為可點擊的網站連結。
- 如果你使用 GitHub Pages，也可以把該連結放在 repo 的首頁說明或 Wiki 中。


## 目標
- 在本地或雲端執行 Streamlit 應用
- 使用 Open‑Meteo 取得即時天氣（預設）
- 可選地使用 OpenAI 做進階 RAG 型建議（使用者須在側欄輸入自己的 API Key）

## 已加入的檔案
- `code/streamlit_app.py` - 主應用程式
- `code/system.py` - LLM 與資料整合邏輯（請勿將金鑰硬編碼）

## 快速開始 (Windows / PowerShell)
1. 建議建立虛擬環境並安裝相依：

```powershell
cd E:\python_project\contest\TGIS
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. 啟動 Streamlit 應用：

```powershell
streamlit run code\streamlit_app.py
```

3. 若要使用 OpenAI 進階建議：
- 在應用側欄輸入您的 OpenAI API Key（此金鑰不會被儲存在 repo）
- 或先在系統環境中設定 `OPENAI_API_KEY`（或在 PowerShell session 設定 ` $env:OPENAI_API_KEY = "sk-..."`）

## 推送至 GitHub
請參考底下「將專案推上 GitHub」章節取得步驟。

## 注意事項（安全）
- 請勿將任何 API Key 或憑證提交到 Git。若不慎提交，請盡快移除並重新產生金鑰。
- 大型資料或敏感資料請放到外部儲存（例如 S3 / private storage）或使用 `.gitignore` 排除。

## 將專案推上 GitHub（PowerShell 範例）
若尚未初始化 git：
```powershell
cd E:\python_project\contest\TGIS
git init
git add .
git commit -m "Initial commit: TGIS Streamlit app"
```
在 GitHub 上建立一個新的 repository（假設名稱為 `TGIS`），然後在本機加入遠端並推送：

```powershell
# 用您 GitHub repo 的 clone URL 替換 <REMOTE-URL>
git remote add origin https://github.com/<your-username>/TGIS.git
git branch -M main
git push -u origin main
```

如果您安裝了 GitHub CLI，也可以直接用：
```powershell
gh repo create <your-username>/TGIS --public --source=. --remote=origin --push
```

## (選用) 在 Streamlit Community Cloud 部署
1. 將程式碼推至 GitHub
2. 在 https://share.streamlit.io/ 登入並連結您的 GitHub 帳號
3. 新增一個 app，選擇剛建立的 repo 與 `code/streamlit_app.py` 作為入口

## CI / 檢查
本 repo 包含一個基本的 GitHub Actions workflow（`.github/workflows/ci.yml`）來檢查 Python 語法，視需要可擴充為 lint/test。

## 後續建議
- 若需要更穩定的 LLM 回應格式，建議在 `system.py` 中實作結構化回應格式 (JSON schema) 並加入 retries
- 若計畫長期部署，考慮建立 Dockerfile 以便一致性部署

如要我直接幫您把這個專案推上 GitHub（例如幫您建立 README/.gitignore、或產生 GitHub Actions workflow），我可以幫忙產生檔案與指令（但無法代替您在本機或 GitHub 上按下授權/建立按鈕）。
