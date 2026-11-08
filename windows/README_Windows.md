# Windows 本機部署與登入後自動啟動

此方式不要求電腦 24 小時開機。Windows 使用者登入後會啟動 Docker Desktop（若尚未啟動），等待 Docker Engine 就緒，再執行 Docker Compose。登出、關機或睡眠期間，Worker 不會持續監控市場。

## 前置需求

1. Windows 10/11 64 位元，已啟用 WSL 2。
2. 安裝並啟動 Docker Desktop：https://www.docker.com/products/docker-desktop/
3. 將本專案完整下載或 clone 到固定位置；不要在設定排程後任意移動資料夾。
4. 確認 docker --version 及 docker compose version 可正常執行。

## 第一次設定

在專案根目錄開啟 PowerShell：

~~~powershell
Copy-Item .env.example .env
notepad .env
~~~

先確認以下設定。預設禁止自動模擬買進：

~~~dotenv
PAPER_WATCHLIST=2330:上市,6488:上櫃
PAPER_AUTO_ENTRIES=false
PAPER_EXIT_PROFILE=atr_dynamic
PAPER_ATR_MULTIPLE=2.0
PAPER_WORKER_INTERVAL_SECONDS=30
~~~

觀察清單只是格式範例，不是投資建議。持倉標的必須留在觀察清單，Worker 才能管理其報價與退出。不要把 .env 或 API Token 上傳到 GitHub。

## 安裝登入後自動啟動

從專案根目錄執行：

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\windows\install-autostart.ps1
~~~

這會建立 Windows 工作排程器工作「TW Stock Paper Trader」，觸發時機是目前 Windows 使用者登入。腳本會在 Docker Engine 未就緒時嘗試啟動 Docker Desktop，最多等待 5 分鐘，再執行 docker compose up -d --build。不需要系統管理員權限。

每次 Windows 登入後，可在瀏覽器開啟 http://localhost:8501。也可以手動啟動：

~~~powershell
.\windows\start-paper-trader.ps1
~~~

## 檢查與停止

~~~powershell
docker compose ps
docker compose logs --tail 100 paper-worker
~~~

停止服務但保留交易資料：

~~~powershell
.\windows\stop-paper-trader.ps1
~~~

移除登入自動啟動排程（不會停止容器，也不會刪除資料）：

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\windows\uninstall-autostart.ps1
~~~

若要完整停止容器但保留資料，可在專案根目錄執行 docker compose down。不要執行 docker compose down -v，除非確定要刪除持久化交易資料。

啟動紀錄：%LOCALAPPDATA%\TwStockPaperTrader\startup.log。

## 重要限制

- 這是「登入後啟動」，不是 Windows 開機但尚未登入時就啟動；Docker Desktop 的使用者工作階段需要登入。
- 電腦關機、睡眠、休眠、網路中斷或 Docker Engine 停止時，Worker 不會監控行情，也不會執行停損。重新登入後才恢復。
- Worker 只做模擬交易，沒有券商 Adapter，不會送出真實委託。
- PAPER_AUTO_ENTRIES=false 時不會自動建立模擬買進委託。確認觀察清單、行情與帳本後，若要開啟自動模擬買進，才手動將 .env 改為 true，再執行 docker compose up -d。
- Yahoo Finance 1 分鐘報價可能延遲；Worker 會依 60 秒新鮮度門檻拒絕過期報價。這不等同交易所認證即時行情。
- 第一次建置需要下載 Docker 映像與 Python 套件，必須有網路連線。--build 只在需要時重建映像；之後登入仍會執行 Compose。
- Docker Desktop 在個人用途通常可免費使用；商業使用須確認 Docker 授權條款。
