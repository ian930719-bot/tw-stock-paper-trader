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
- 滾動式樣本外測試：不重疊測試窗口、窗口前資料暖機、測試期交易邊界鎖定及跨窗口報酬鏈結；AI 候選訊號預先計算重用、精確訓練資料指紋快取與窗口進度顯示
- 盤勢條件績效：依基準股票收盤價相對 200 日 EMA 與 EMA 方向分組，觀察多頭／空頭／盤整期間的樣本外日報酬
- 參數敏感度矩陣：預先指定風險預算與出場規則組合，檢查績效對參數變動的敏感程度，不自動挑選最佳參數
- 最新基本面快照，以及巴菲特／彼得林區風格的簡化條件檢查
- 行情缺漏、資料日期間隔與非正價格品質檢查
- 模擬帳本 CSV 匯出，以及 JSON 備份／還原
- 台股普通股票／ETF 價格升降單位檢查；支援整股及 1–999 股盤中零股模擬委託，委託類型會保存於 SQLite 並在委託清單顯示
- 模擬委託採當日有效，跨日未成交委託自動到期
- 觸價模擬成交以委託限價保守計價，不假設能取得更佳價格
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

## Walk-forward 效能與驗證

- AI 自適應 Walk-forward 預先計算 30 組候選訊號，外層測試窗口重用已計算訊號。
- 候選選擇快取鍵包含訓練資料完整內容指紋，以及本金、持倉上限、風險預算、成本、出場規則與暖機長度；資料或設定改變即重新評估，且快取容量有限。
- 網頁顯示已完成的樣本外窗口與日期範圍。快取只重用相同訓練切片的內部選擇結果，不使用外層測試資料挑選策略。

## 回測方法與限制

- 技術訊號在交易日收盤後形成，最早於下一交易日開盤執行。
- 多股票組合共用現金，不允許融資或放空；依 20 日動能排序候選標的，受持倉數與可用資金限制。
- 成本為估算值：手續費 0.1425%、一般股票賣出證交稅 0.3%、滑價 0.1%。實際券商折扣、最低手續費、ETF 稅率與市場衝擊可能不同。
- 出場規則研究比較包含 ATR × 1.5／2／2.5／3、O’Neil 8% 初始停損參考，以及 Minervini 前 10 根已完成日 K 最低價的結構失效代理。各規則共用帳戶風險預算，但股數依各自初始停損距離計算；仍受可用現金及持倉上限約束。
- ATR 移動停損以當日完成的高點、收盤價及 ATR 更新，下一交易日才生效；進場日也納入當日高水位。若隔日跳空跌破停損，回測以開盤價估算成交，再扣滑價。
- O’Neil 代理在收盤價達進場價 +20% 時標記獲利檢視，採 3 ATR 寬幅追蹤；沒有硬性 20% 停利，也尚未完整實作 20–25% 獲利區、強勢領先股資格與 8 週持有規則。
- Minervini 代理在達 2R 後將停損推至成本價，達 3R 後啟動 2.5 ATR 追蹤；R 以該筆交易的初始停損距離計算。10 日結構低點只是透明代理，並非完整 SEPA 型態辨識。
- 所有停損均以日線 OHLC 模擬；若同一根 K 棒內先後順序不明，不能推知實際盤中路徑或保證成交。回測成交是模型假設，不代表實際可成交價格。
- 停損以日線 OHLC 近似；同一根 K 棒內若同時觸及停損與其他條件，無法重建真實成交順序。
- 歷史成交與回測 OHLCV 使用 FinMind TaiwanStockPrice 原始成交價，失敗時才使用 Yahoo Finance auto_adjust=False 的原始 OHLCV 備援；不使用 TaiwanStockPriceAdj，因其還原係數會以最新交易日為基準回溯調整過去價格，可能把未來除權息資訊帶入歷史回測。舊版還原價快取會在資料庫遷移後標記為 unknown 並排除，需重新下載原始價格。
- 回測會載入 FinMind 公司行動事件：現金股利於除息日建立應收、公告發放日入帳；股票股利於除權日調整股數；分割、減資及變更面額依官方事件前後參考價比例調整股數及每股成本／停損。零股權利以參考價或當日開盤價估算現金結算。若任一標的公司行動 API 失敗，介面會明確警告，不能把該次結果視為完整含息績效。公司行動事件資料可能修訂；回測只在事件生效日處理，不把事件提前套用到更早日期。技術指標會在每個公司行動生效日，使用當時已生效的官方除權息前後價格比率，重新調整該日以前的 OHLCV 歷史序列，並只覆寫生效日及之後的指標值；因此不會讓未來除權息改寫事件發生前的訊號。缺少官方比率時，才以現金股利及股數變動作保守推算。
- 獨立 Paper Worker 會定期取得目前持倉的公司行動事件，於生效日以同日事件合併處理，調整模擬持股／每股成本／移動停損；官方除權息參考價比率只套用一次，避免現金股利與股票股利同日重複調整。現金股利於發放日入帳；未成交委託未取消、公司行動資料無法驗證或偵測到漏處理的過去事件時，Worker 會停止撮合與新委託，不猜測權利歸屬。
- 模擬委託依一般股票或 ETF 類別驗證價格升降單位；整股須為 1,000 股整數倍，盤中零股為 1–999 股並使用獨立委託時段。ETF 類別依現行證券主檔名稱／產業欄位推定，無法辨識時採一般股票規則；尚未完整實作每日漲跌幅上下限、初次上市無漲跌幅例外、盤後零股交易、ETF 各類別稅率、處置股／分盤交易及完整交易日／颱風休市日曆。未有官方當日參考價時，不會假裝已驗證漲跌停價。
- 盤中觸價只使用 Yahoo Finance 1 分鐘資料；觸及限價不代表排隊成交。為避免把觸價價位當成可取得的更佳價格，模擬成交以委託限價入帳，但仍無法模擬委託簿深度、成交量分配、部分成交與排隊順位。
- 一般交易模擬支援整張（1,000 股整數倍）及盤中零股（1–999 股）分開委託；盤中零股委託時間為 09:00–13:30，自 09:10 起每 5 秒集合競價，未成交委託不延續至盤後。此系統只驗證盤中零股數量及委託時段，仍以報價觸價模擬成交，沒有重建每 5 秒集合競價、價格／時間優先、瞬間價格穩定措施或實際委託簿；盤後零股 13:40–14:30 委託及 14:30 單次撮合尚未實作。
- 交易日判斷已在模擬委託介面連接 TWSE 官方年度休市日曆，並在無法驗證時停止介面上的委託與撮合操作；但目前尚未整合 TPEx 獨立公告與臨時休市通知，上市／上櫃市場仍須以各交易所當日公告為準。
- 基本面頁面是最新快照，不是 point-in-time 歷史財報。巴菲特／林區篩選僅為目前快照的簡化條件，不可當作歷史回測或完整複製其投資方法。
- 現行上市／上櫃主檔由 FinMind TaiwanStockInfo 提供，依最新 date 判斷轉板後市場別；不含興櫃、權證與歷史下市證券，也不等於歷史成分股資料庫。若 API 失敗，程式會明確警告並顯示內建備援清單，不會冒充完整資料庫。
- 主檔在應用程式快取 6 小時；歷史日線仍由 FinMind 優先、Yahoo Finance 備援依選取標的逐檔取得，並非預先下載全市場歷史價格。未處理完整歷史成分股、下市股、除權息再投資、停牌、漲跌停排隊及完整流動性衝擊。
- 滾動式樣本外測試每個窗口都獨立重設本金與持倉；跨窗口鏈結曲線只串接各窗口報酬，不代表連續持有的真實帳戶。測試窗口不重疊，窗口前資料只用於指標暖機，禁止在測試起點之前建立新倉。
- 盤勢分組使用使用者選取的基準股票作市場代理（若有選取 0050 則預設 0050），以收盤價相對 200 日 EMA、以及 EMA 相對 20 日前的方向分類。這不是加權指數或正式景氣循環判定；結果只描述該代理標的下的條件報酬。
- 參數敏感度矩陣預設測試 0.25%、0.5%、1% 單筆風險預算，搭配 ATR × 2、ATR × 1.5、ATR × 3 三種出場規則。矩陣用於檢查穩健性，不是參數最佳化；不得依同一批最終樣本外結果反覆挑選參數。
- 樣本外區段應保留作最終檢驗，不得依其結果反覆調參。樣本外表現不保證未來獲利。
- JSON 備份可跨裝置手動還原，但目前沒有雲端自動同步；匯入會覆蓋當前帳本。
- 獨立 Paper Worker 的 FinMind 權杖由執行環境的 `FINMIND_TOKEN` 環境變數提供；若未設定，僅能使用 FinMind 允許的免權限資料集，無權限資料集抓取失敗時會 fail-closed，不會把空資料當作沒有公司行動。

## 官方交易日曆

模擬委託介面依市場別核對官方交易日曆：上市使用 TWSE 官方 JSON 日曆，上櫃使用 TPEx 官方「開休市日期表」頁面。兩者均快取 6 小時；官方資料無法取得、格式錯誤或缺少目標年度時，該市場的委託及撮合採 fail-closed 停止。TPEx 頁面由官方網站 HTML 表格解析，若網站改版或雲端來源遭拒絕，會停止上櫃委託而不猜測日期。臨時颱風停市等公告可能未即時反映，仍須核對當日交易所公告。

資料來源：TWSE OpenAPI `https://openapi.twse.com.tw/v1/holidaySchedule/holidaySchedule`；TWSE 日曆 `https://www.twse.com.tw/zh/trading/holiday.html`；TPEx 官方開休市日期表 `https://www.tpex.org.tw/web/bulletin/trading_date/trading_date.php?l=zh-tw`。

## 官方漲跌停價驗證

模擬委託引擎依證券市場別讀取交易所官方每日資料，不以昨收乘以 1.1 推算。上市使用 TWSE TWT84U 的本日漲停價、本日開盤競價基準、本日跌停價；上櫃使用 TPEx 官方每日行情所提供的次日參考價、次日漲停價及次日跌停價欄位。價格資料快取 60 秒；證券代號、日期或價格欄位無法核對時，拒絕委託。交易所提供無漲跌幅限制的特殊標記時，依官方標記處理；無法辨識的特殊商品不自行推算，採 fail-closed。首五日、外國成分 ETF、槓反商品及其他特殊類別仍需持續擴充分類與官方資料驗證，不能視為已完整涵蓋所有商品規則。

## 資料庫持久化設定

可設定 `MARKET_DB_PATH`（歷史行情快取）與 `EXECUTION_DB_PATH`（模擬委託及帳本）指向持久化磁碟上的 SQLite 檔案。若未設定，預設使用系統暫存目錄。Streamlit Community Cloud 的本機檔案不保證跨重啟／重新部署保存；正式長期使用應連接外部持久化資料庫或定期下載備份。

## 自動測試

```bash
python -m py_compile app.py research_engine.py walk_forward.py market_data.py analytics.py stock_universe.py historical_store.py execution_engine.py paper_account.py stop_monitor.py strategy_exits.py taiwan_market_rules.py taiwan_trading_calendar.py market_trading_calendar.py taiwan_price_limits.py
python -m unittest discover -s tests -v
```

## Portfolio risk and multi-strategy research
- Strategy deterioration diagnostics compare recent 63-session OOS Sharpe/drawdown with the full supplied OOS sample. Alerts are research-only and never automatically disable strategies.


- Five distinct daily-OHLCV strategy families are available, including long-term trend structure, volume breakout, momentum continuation, trend-filtered mean reversion, and volatility-squeeze breakout. The adaptive search evaluates 30 predeclared candidates spanning breakout, multifactor, mean-reversion, and squeeze-breakout families. Signals are evaluated at the close and simulated no earlier than the next session. New entries in both historical simulation and the paper worker are capped at 1% of prior-known 20-session average daily dollar turnover; current-session volume is never used to size an opening fill.
- Current cross-sectional risk snapshots use common-date close-to-close returns, covariance shrinkage, capped long-only risk-parity weights, and explicit cash reserve. These weights are not retroactively injected into historical OOS results.
- The optional strategy ensemble combines each fixed strategy's OOS equity curve using rolling risk estimates. Rebalance weights for each date use only returns observed before that date; the first 60 sessions use equal weights. It is a separate research curve, not the paper account ledger.


## Paper ledger integrity checks

The paper-account dashboard runs SQLite `integrity_check` and reconciles cash and per-symbol positions against the fill ledger. It also verifies filled-order/fill-row consistency and checks that fill price multiplied by quantity matches gross notional. Any discrepancy locks new orders, dynamic exits, and paper matching in the dashboard; the standalone worker also stops all trading. Export a full SQLite backup and investigate before attempting recovery. Each fill performs one fresh official price-band verification before account mutation to avoid redundant exchange API requests.

The paper-account area also offers a consistent SQLite snapshot download containing cash, positions, orders, fills, and worker records. Store it offline regularly. This is an export-only backup; it does not sync to cloud storage or perform an unsafe online overwrite restore.


## Point-in-time security-universe snapshots and survivorship bias

After a successful FinMind TaiwanStockInfo fetch, the app stores the first observed listed/OTC security universe of each Taiwan calendar day as an immutable SQLite snapshot alongside market history. `universe_history.load_universe_as_of(as_of)` only returns a snapshot actually captured by the requested instant. If no earlier snapshot exists, it returns an explicit unavailable status rather than substituting today's surviving listings. Continuous capture can preserve future listing, transfer, and delisting changes, but it cannot reconstruct delisted securities or historical index constituents from before the first captured snapshot. Earlier backtests must therefore continue to disclose survivorship-bias limitations. SQLite persistence still depends on the deployment disk; Streamlit Community Cloud temporary storage can be cleared on restart or redeploy.


## Historical data-first factor research (research branch)

A research-only command is available in `historical_research.py`. It audits an OHLCV panel and computes a broad, auditable set of momentum, reversal, trend, breakout, volume, volatility, gap/range and relative-strength factors. Signals are formed from completed closes; forward outcomes assume entry at the next session open.

Download a resumable raw-history candidate panel (current TWSE/TPEx master plus historical delisting announcements):

\x60\x60\x60bash
python historical_data_collector.py --output data/taiwan_history --start-date 2010-01-04
\x60\x60\x60

Set `FINMIND_TOKEN` in the environment to use the higher documented request allowance. The collector writes one CSV per symbol, a candidate symbol list, source snapshots, and a SHA-256 manifest. It resumes by skipping already-downloaded symbol files. The API request allowance is rate-limited (documented as 300/hour without a token and 600/hour with a token), so a full-market pull can take hours and may need multiple resumable runs. The collector does not claim full historical point-in-time constituents or verified delisted-price coverage.

Alternatively, pass a single combined CSV. The research runner can read the collector output directory directly and merge its per-symbol CSVs. For custom input, prepare a CSV with columns:

```text
symbol,date,open,high,low,close,volume
2330,2020-01-02,....
```

Optional `turnover` is used for turnover-shock features. Example:

```bash
python historical_research.py --input data/taiwan_history --output artifacts/historical_research --split-date 2024-01-01 --source "FinMind TaiwanStockPrice" --price-basis raw
```

Outputs: `data_coverage.csv`, `data_audit_summary.json`, `historical_factor_panel.csv`, `factor_research_results.csv`, and `hypothesis_registry.csv`.

Important limitations:
- This command does not download the whole Taiwan market. The panel must be acquired and archived separately; use official exchange sources where available and reconcile vendor feeds.
- `--point-in-time-universe` and `--delisted-coverage` are explicit evidence assertions, not switches that create missing data. Only pass them when the supplied panel's provenance proves the coverage.
- A current security master plus delisting notices is only a candidate symbol list, not historical membership. A current 0050 constituent snapshot is not historical membership. Without verified point-in-time constituents and delisted securities, the audit intentionally remains research-grade-ineligible.
- Factor IC and top-minus-bottom spreads are descriptive screens, not net strategy returns. They do not include transaction costs, portfolio constraints, capacity, or multiple-testing correction; candidates must proceed to nested walk-forward and blind forward paper tests.
- Never pass retrospectively adjusted prices as executable raw fills without a separately validated corporate-action treatment.


## Historical Taiwan market research pipeline

The research branch includes a resumable FinMind raw daily-bar collector and a research-only factor screening stage.

- The full collection workflow builds a stock-only candidate list, including current common equities and historical delisting candidates; ETF/bond/warrant products are filtered from the common-stock universe.
- Smoke tests use deterministic industry-stratified sampling and reserve a portion for delisted candidates. A smoke test is not a representative performance backtest.
- Full collection runs in bounded 330-minute batches. Each batch restores the prior GitHub Actions cache, skips successfully downloaded symbols, checkpoints per symbol, saves the cache, and dispatches the next batch until all candidates have been attempted.
- The workflow only runs factor screening after the candidate collection is complete. It does not assert that the candidate list is a complete historical point-in-time universe.
- The resulting factor IC and quintile spreads are exploratory diagnostics, not a tradable portfolio backtest. Research-grade approval remains blocked until point-in-time universe, delisted-price, corporate-action, official-calendar and exchange-source reconciliation are independently verified.
- Add a repository Actions secret named `FINMIND_TOKEN` to use the higher FinMind request allowance. Without it, collection uses conservative pacing and may require multiple automatically chained runs.

Manual full-universe collection is available through the GitHub Actions workflow_dispatch input `max_symbols=0`; bounded smoke tests use a positive cap.


The full-universe research workflow uses `streaming_factor_research.py` after collection. It processes one symbol at a time, partitions factor panels by year, and reports daily cross-sectional rank IC, gross quintile spreads, horizon-aware HAC t-statistics, Benjamini-Hochberg q-values, per-symbol OHLC quality, candidate-to-file coverage, and SHA-256 manifest checks. Its post-split segment is explicitly descriptive, not a final blind test. It never labels the result research-grade while the PIT-universe and official-source gates remain unresolved.


## Corporate-action-aware historical research (research branch)

- The historical collector can resume collection of `TaiwanStockDividendResult` per symbol plus market-wide `TaiwanStockSplitPrice`, `TaiwanStockCapitalReductionReferencePrice`, and `TaiwanStockParValueChange` events. It writes raw source files, a normalized event table, and a separate manifest. Empty event responses are recorded as completed responses; API failures remain retryable.
- The factor screen uses event-adjusted OHLC for relative price signals and forward-return labels while preserving the original raw OHLCV files for execution-price provenance. Each event's reference-price / pre-event-price ratio is applied only to bars strictly before its effective date. This removes mechanical ex-right / split gaps from realized returns without changing the raw execution bars.
- Volume-based factors are suppressed for 20 sessions after a corporate action because the collector does not yet have a fully reconciled share-count adjustment history. Conflicting same-day event ratios are excluded rather than guessed and are reported as blockers.
- The GitHub Actions pipeline now waits for the resumable corporate-action collection stage before running the factor screen. With the unauthenticated FinMind request allowance, collecting event data for the full candidate list can take multiple batches; set the repository Actions secret `FINMIND_TOKEN` if an appropriately entitled token is available. Never paste the token into source code or chat.
- The collector also stores the first current-security-master snapshot observed each Taiwan calendar day as an immutable CSV with a SHA-256 digest. This improves prospective point-in-time coverage from the first captured date onward; it cannot reconstruct historical membership before capture began.
- This stage does not create a complete historical point-in-time universe, prove complete delisted/suspended-price coverage, or independently reconcile vendor data against official TWSE / TPEx records. The research eligibility gate remains closed until those requirements and a costed continuous-capital blind OOS portfolio backtest are completed.

## TSMC 2330 causal swing turning-point research

The first single-stock research stage is isolated in `tsmc_swing_research.py`. It accepts only symbol 2330, reads raw OHLCV, computes close-known technical/volume/volatility features, and labels multi-scale swing highs/lows with a causal reversal-confirmation date. The pivot date is retrospective ground truth; the later confirmation date records when the reversal threshold became observable. Feature snapshots are exported 1/3/5/10/20 sessions before each pivot to inspect lead/lag behavior.

Run after the 2330 raw daily-bar history has been collected:

```bash
python tsmc_swing_research.py --input data/taiwan_history --output artifacts/tsmc_2330_swing
```

Outputs: `tsmc_2330_pivot_events.csv`, `tsmc_2330_pivot_leadlag.csv`, `tsmc_2330_causal_features.csv`, and `tsmc_2330_research_summary.json`. Five predeclared reversal/ATR thresholds are reported side by side; the script does not select a winning threshold, run a return backtest, or deploy a strategy. 研究工作流會先取得 2330 的現金股利、分割、減資及面額變更事件，依已核對的除權息參考價比例建立 point-in-time-safe 訊號價格序列；原始 OHLC 仍保留作為成交價格參考。事件日及其後 20 個交易日的量能特徵會遮罩，極值日或確認日落在公司行動窗口的轉折標籤會排除。若同日事件比例衝突，工作流 fail-closed，不輸出可用轉折標籤。若自行輸入資料未提供公司行動檔，結果仍屬未校正的初步研究。這個階段仍是轉折樣本與訊號時序研究，不代表已達 200% CAGR。

The follow-on TSMC-only ranking diagnostic is in `tsmc_2330_oos_signal_diagnostics.py`. It uses a chronological 2010–2021 / 2022+ split, purges the training boundary by the forecast horizon, and reports ranking metrics for 1/3/5-session high- and low-pivot labels. It is explicitly not a trading backtest. First-round interpretation and limitations are recorded in `research/TSMC_2330_SWING_PRELIMINARY_FINDINGS.md`.


The TSMC-only workflow also runs `tsmc_2330_pattern_combinations.py`: it evaluates a small, predeclared set of two- and three-feature conjunctions. Feature cutoffs are estimated from the 2010–2021 training segment only; 2022+ metrics are descriptive diagnostics, not a parameter-selection target. Because the 2022+ segment has already been inspected in earlier diagnostics, it is not considered a fresh blind final holdout. Outputs are `tsmc_2330_pattern_combinations.csv/json`; no trades or CAGR are simulated.


A nested chronological rule-development stage is available in `tsmc_2330_frozen_rule_research.py`. It estimates feature cutoffs on 2010–2016, selects one predeclared conjunction per direction using 2017–2021 mean F1 across 1/3/5-session pivot targets, purges five sessions at split boundaries, and reports the frozen rules on 2022+ data without using that period for selection. Since 2022+ has already been examined in earlier diagnostics, this remains exploratory rather than a pristine blind test. No trade simulation or CAGR is included.


Execution testing is implemented in `tsmc_2330_execution_backtest.py`. It uses the frozen high/low rules, next-session raw open execution, FinMind `TaiwanStockPriceLimit` exchange limit fields, cash-only dividend entitlement, standard 0.1425% commission with a NT$20 minimum, 0.3% sell-side transaction tax, and 5/10/20/30 bps adverse slippage scenarios. It compares the rule-driven long-only strategy with a cost-matched TSMC buy-and-hold benchmark for the 2017–2021 rule-selection period and 2022+ historical holdout. This is a daily-OHLC execution approximation, not tick-level fill proof; 2022+ has already been inspected in prior diagnostics and is not a pristine blind holdout.


The baseline execution backtest showed that the first frozen setup-only rules materially underperformed TSMC buy-and-hold. The next development stage, `tsmc_2330_reversal_confirmation_backtest.py`, keeps the training-derived low/high setup thresholds fixed, arms each setup for at most five sessions, and tests five predeclared reversal confirmations (close reversal, prior-range break, RSI turn, SMA5 reclaim, SMA20 reclaim). It selects a confirmation pair using 2017–2021 only, then reports the frozen pair on 2022+ with the same dividend, cost, slippage and official-rule price-limit model. No 2022+ tuning is permitted.


A further validation-only risk study, `tsmc_2330_trend_atr_risk_research.py`, takes the selected prior-range reversal confirmation as its fixed base and compares four predeclared entry regimes (none, above SMA200, positive SMA50 slope, full trend stack) against three ATR trailing-stop settings (none, 2.5 ATR, 4 ATR). The best combination is selected on 2017–2021 only; 2022+ is reported with 5/10/20/30 bps slippage but is not used for selection.
