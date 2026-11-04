# 台股模擬交易研究系統

部署：前往 https://share.streamlit.io/，以 GitHub 登入，選擇此 repository，Main file path 設為 `app.py`，按 Deploy。

功能：台股觀察清單、1/2/3/5 年歷史日線、EMA/RSI/MACD/OBV/ATR、技術規則評分、歷史回測、CSV 匯出。

限制：行情來自 Yahoo Finance，可能延遲或缺漏，非交易所認證即時行情。法人、融資融券及分點籌碼尚未接入。評分是規則引擎，不是已訓練 AI。回測未完整處理除權息、停牌、漲跌停、流動性及下市股偏誤。無持久化模擬帳本或推播。不連接券商、不會送出真實委託。
