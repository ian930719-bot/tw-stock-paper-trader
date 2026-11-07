# 台股 AI 模擬交易研究系統

本專案是研究與教學用途的台股模擬交易工具，不連接券商、不送出真實委託，也不構成投資建議。

## 功能

- FinMind TaiwanStockInfo 全市場現行上市／上櫃證券主檔，依同一代號最新市場別記錄處理轉板；支援市場／產業／名稱／代號搜尋與 CSV 匯出
- 從完整主檔選取觀察清單，單次最多載入 30 檔歷史日線以控制行情來源限流
- 隨機森林五個交易日方向分類；提供時間序列測試平衡準確率與多數類基準比較
- 回測風險指標（年化波動率、Sharpe、Sortino、Calmar、最大回撤）及交易品質統計
- 三種技術策略代理：Minervini 趨勢模板、O’Neil CAN SLIM 技術代理、Livermore 順勢突破
- 多股票共享現金組合回測，含持倉上限、ATR 停損、風險預算與估算交易成本
- 訓練／驗證／樣本外期間切分，依共同交易日期切分並保留指標暖機資料
- 最新基本面快照，以及巴菲特／彼得林區風格的簡化條件檢查
- 行情缺漏、資料日期間隔與非正價格品質檢查
- 模擬帳本 CSV 匯出，以及 JSON 備份／還原
- GitHub Actions 語法檢查與單元測試

## 執行

```bash
pip install -r requirements.txt
streamlit run app.py
```

若專案尚未提供 requirements.txt，可安裝：

```bash
pip install streamlit yfinance pandas numpy plotly requests scikit-learn
```

## 回測方法與限制

- 技術訊號在交易日收盤後形成，最早於下一交易日開盤執行。
- 多股票組合共用現金，不允許融資或放空；依 20 日動能排序候選標的，受持倉數與可用資金限制。
- 成本為估算值：手續費 0.1425%、一般股票賣出證交稅 0.3%、滑價 0.1%。實際券商折扣、最低手續費、ETF 稅率與市場衝擊可能不同。
- 停損以日線 OHLC 近似；同一根 K 棒內若同時觸及停損與其他條件，無法重建真實成交順序。
- 歷史資料來自 Yahoo Finance，可能延遲、修訂、缺漏，並非交易所認證資料。
- 基本面頁面是最新快照，不是 point-in-time 歷史財報。巴菲特／林區篩選僅為目前快照的簡化條件，不可當作歷史回測或完整複製其投資方法。
- 現行上市／上櫃主檔由 FinMind TaiwanStockInfo 提供，依最新 date 判斷轉板後市場別；不含興櫃、權證與歷史下市證券，也不等於歷史成分股資料庫。若 API 失敗，程式會明確警告並顯示內建備援清單，不會冒充完整資料庫。
- 主檔在應用程式快取 6 小時；歷史日線仍由 Yahoo Finance 依選取標的逐檔取得，並非預先下載全市場歷史價格。未處理完整歷史成分股、下市股、除權息再投資、停牌、漲跌停排隊及完整流動性衝擊。
- 樣本外區段應保留作最終檢驗，不得依其結果反覆調參。樣本外表現不保證未來獲利。
- JSON 備份可跨裝置手動還原，但目前沒有雲端自動同步；匯入會覆蓋當前帳本。

## 自動測試

```bash
python -m py_compile app.py research_engine.py market_data.py analytics.py stock_universe.py tests/test_research_engine.py tests/test_market_data.py tests/test_analytics.py tests/test_stock_universe.py
python -m unittest discover -s tests -v
```
