import json
import os
import tempfile
from pathlib import Path
import streamlit as st
import yfinance as yf
import requests
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import numpy as np
import plotly.graph_objects as go
# Keep this import synchronized with research_engine.py in the same deployed revision.
from strategy_evolution import evolve_strategy
from fundamental_factors import score_fundamental_snapshot
from taiwan_factor_data import load_symbol_history, next_start_date, enrich_industry_comparison
from research_engine import STRATEGIES, daily_strategy_returns, performance, fundamental_snapshot, shared_capital_backtest
from datetime import datetime
from zoneinfo import ZoneInfo
from market_data import quote_freshness
from analytics import equity_risk_metrics, trade_statistics
from historical_store import get_incremental_history, price_cache_summary
from execution_engine import PaperExecutionEngine, OrderIntent, RiskLimits, Side, taiwan_regular_session, realized_pnl_from_trades
from paper_account import PaperAccount
from adaptive_exit_monitor import AdaptiveExitMonitor
from paper_worker import PaperWorker, read_worker_status
from strategy_exits import PROFILES, initial_stop, risk_position_size
from walk_forward import rolling_walk_forward, regime_performance, parameter_sensitivity
from taiwan_market_rules import floor_board_lot
from taiwan_trading_calendar import trading_day_status
from taiwan_price_limits import get_official_price_limit
from market_trading_calendar import market_trading_day_status
from trading_config import TradingConfig
from trading_mode import StagedTradingController
st.set_page_config(page_title="台股模擬交易",layout="wide")
TZ=ZoneInfo("Asia/Taipei")
from stock_universe import fetch_stock_universe, universe_summary, StockUniverseError

@st.cache_data(ttl=21600, show_spinner=False)
def get_stock_master(token):
    return fetch_stock_universe(token)

try:
    finmind_token = st.secrets.get("FINMIND_TOKEN", "")
except Exception:
    finmind_token = ""
universe_error = None
try:
    stock_master = get_stock_master(finmind_token)
    universe_source = "FinMind TaiwanStockInfo（每日更新之全市場主檔）"
except Exception as exc:
    universe_error = str(exc)
    stock_master = pd.DataFrame([
        {"代號": "2330", "名稱": "台積電", "市場": "上市", "產業": "半導體", "市場資料日期": None},
        {"代號": "2454", "名稱": "聯發科", "市場": "上市", "產業": "半導體", "市場資料日期": None},
        {"代號": "2308", "名稱": "台達電", "市場": "上市", "產業": "電子零組件", "市場資料日期": None},
        {"代號": "2317", "名稱": "鴻海", "市場": "上市", "產業": "其他電子", "市場資料日期": None},
        {"代號": "2382", "名稱": "廣達", "市場": "上市", "產業": "電腦及週邊設備", "市場資料日期": None},
        {"代號": "3231", "名稱": "緯創", "市場": "上市", "產業": "電腦及週邊設備", "市場資料日期": None},
        {"代號": "3017", "名稱": "奇鋐", "市場": "上市", "產業": "其他電子", "市場資料日期": None},
        {"代號": "2603", "名稱": "長榮", "市場": "上市", "產業": "航運業", "市場資料日期": None},
        {"代號": "2881", "名稱": "富邦金", "市場": "上市", "產業": "金融保險", "市場資料日期": None},
        {"代號": "2891", "名稱": "中信金", "市場": "上市", "產業": "金融保險", "市場資料日期": None},
    ])
    universe_source = "內建備援清單（非完整資料庫）"
W = dict(zip(stock_master["代號"], stock_master["名稱"]))
market_by_symbol = dict(zip(stock_master["代號"].astype(str), stock_master["市場"].astype(str)))
etf_mask = (
    stock_master["產業"].astype(str).str.contains("ETF|指數股票型基金", case=False, na=False)
    | stock_master["名稱"].astype(str).str.contains("ETF", case=False, na=False)
)
etf_codes = set(stock_master.loc[etf_mask, "代號"].astype(str))
universe_stats = universe_summary(stock_master) if universe_error is None else {
    "總數": len(stock_master), "上市": len(stock_master), "上櫃": 0,
    "資料日期": None, "更新時間": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
}
st.title("台股模擬交易研究系統")
st.caption("研究用途；行情可能延遲，非交易所認證即時行情。不連接券商、不會送出真實委託。")
with st.sidebar:
    st.subheader("全市場證券選取")
    popular = ["2330","2454","2308","2317","2382","3231","3017","2603","2881","2891"]
    defaults = [code for code in popular if code in W]
    ss=st.multiselect("觀察股票（可輸入代號或名稱搜尋）",list(W),default=defaults,format_func=lambda x:x+" "+W.get(x,x))
    custom_codes=st.text_input("額外股票代號（逗號分隔）",placeholder="例如 0050, 006208")
    extra_codes=[x.strip() for x in custom_codes.replace("，",",").split(",") if x.strip().isalnum() and 4<=len(x.strip())<=8]
    ss=list(dict.fromkeys(ss+extra_codes))
    if len(ss)>30:
        st.warning("單次行情分析最多載入 30 檔；完整上市櫃名單仍可在「上市櫃股票庫」分頁搜尋及下載。")
        ss=ss[:30]
    st.caption(f"目前載入 {len(ss)} / 最多 30 檔；股票主檔共 {len(stock_master):,} 檔")
    period=st.selectbox("歷史期間",["5y","3y","2y","1y"])
    capital=st.number_input("模擬本金",100000,10000000,100000,100000)
    risk=st.number_input("單筆風險上限",500,50000,500,500)
    if st.button("重新抓取"): st.cache_data.clear()
if not ss: st.stop()
def fetch_yahoo_history(code, period):
    for suffix in [".TW", ".TWO"]:
        try:
            frame = yf.Ticker(str(code) + suffix).history(
                period=period, interval="1d", auto_adjust=True
            )
            if not frame.empty:
                frame = frame.reset_index().rename(columns={
                    "Date":"date", "Datetime":"date", "Open":"open",
                    "High":"high", "Low":"low", "Close":"close", "Volume":"volume"
                })
                frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
                return frame[["date","open","high","low","close","volume"]].dropna()
        except Exception:
            continue
    return pd.DataFrame()

@st.cache_data(ttl=900, show_spinner=False)
def getdata(code, p, token):
    return get_incremental_history(
        code, p, token=token, fallback_fetcher=fetch_yahoo_history
    )
def ind(d):
    x=d.copy(); c=x.close
    for n in [21,50,200]: x["ema"+str(n)]=c.ewm(span=n,adjust=False).mean()
    u=c.diff().clip(lower=0).ewm(alpha=1/14,adjust=False).mean()
    v=(-c.diff().clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    x["rsi"]=100-100/(1+u/v.replace(0,np.nan))
    x["macd"]=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
    x["sig"]=x.macd.ewm(span=9,adjust=False).mean()
    x["obv"]=(np.sign(c.diff()).fillna(0)*x.volume).cumsum()
    x["atr"]=pd.concat([x.high-x.low,(x.high-c.shift()).abs(),(x.low-c.shift()).abs()],axis=1).max(axis=1).ewm(alpha=1/14,adjust=False).mean()
    x["v20"]=x.volume.rolling(20).mean(); x["h20"]=x.high.shift().rolling(20).max()
    x["r20"]=c.pct_change(20); x["r60"]=c.pct_change(60); x["up"]=x.ema21>x.ema21.shift(5)
    return x

@st.cache_data(ttl=1800)
def get_institutional():
    url = "https://www.twse.com.tw/rwd/zh/fund/T86"
    headers = {"User-Agent": "Mozilla/5.0"}
    now = datetime.now(TZ).date()
    for offset in range(10):
        day = now - timedelta(days=offset)
        if day.weekday() >= 5:
            continue
        try:
            res = requests.get(url, params={"response":"json","date":day.strftime("%Y%m%d"),"selectType":"ALLBUT0999"}, headers=headers, timeout=12)
            j = res.json()
            fields, rows = j.get("fields", []), j.get("data", [])
            if not rows:
                continue
            idx = {name:i for i,name in enumerate(fields)}
            required = ["證券代號","外陸資買賣超股數(不含外資自營商)","投信買賣超股數","自營商買賣超股數","三大法人買賣超股數"]
            if not all(k in idx for k in required):
                continue
            out = {}
            for row in rows:
                code = str(row[idx["證券代號"]]).strip()
                def num(key):
                    try:
                        return int(float(str(row[idx[key]]).replace(",","").strip()))
                    except Exception:
                        return 0
                out[code] = {"外資買賣超股數":num(required[1]),"投信買賣超股數":num(required[2]),"自營商買賣超股數":num(required[3]),"三大法人買賣超股數":num(required[4]),"資料日期":day.strftime("%Y-%m-%d")}
            return out
        except Exception:
            continue
    return {}

@st.cache_data(ttl=60)
def get_quote(code):
    for suffix in [".TW", ".TWO"]:
        try:
            q = yf.Ticker(code+suffix).history(period="1d", interval="1m")
            if not q.empty:
                ts = q.index[-1].to_pydatetime()
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=TZ)
                age = (datetime.now(TZ)-ts.astimezone(TZ)).total_seconds()/60
                return float(q.Close.iloc[-1]), ts.astimezone(TZ).strftime("%Y-%m-%d %H:%M:%S"), round(age,1)
        except Exception:
            pass
    return None, "無報價", None

@st.cache_data(ttl=3600)
def train_ml(d):
    features = ["rsi","macd","sig","r20","r60","atr","v20","ema21","ema50","ema200"]
    z = d.copy()
    z["target"] = (z.close.shift(-5)/z.close-1 > 0.015).astype(float)
    z.loc[z.index[-5:], "target"] = np.nan
    z = z.replace([np.inf,-np.inf], np.nan).dropna(subset=features+["target"])
    if len(z) < 300 or z.target.nunique() < 2:
        return None, None, None
    split = int(len(z)*0.8)
    # Purge the final five training labels: their five-day outcome window
    # overlaps the first test observations and would leak future prices.
    train, test = z.iloc[:max(0, split-5)], z.iloc[split:]
    if len(train) < 100 or train.target.nunique() < 2 or test.target.nunique() < 2:
        return None, None, None
    params=dict(n_estimators=250,max_depth=6,min_samples_leaf=8,
                class_weight="balanced",random_state=42,n_jobs=-1)
    evaluation_model = RandomForestClassifier(**params)
    evaluation_model.fit(train[features], train.target.astype(int))
    metric = balanced_accuracy_score(
        test.target.astype(int), evaluation_model.predict(test[features])
    )
    baseline_model = DummyClassifier(strategy="most_frequent")
    baseline_model.fit(train[features], train.target.astype(int))
    baseline_metric = balanced_accuracy_score(
        test.target.astype(int), baseline_model.predict(test[features])
    )
    last = d.iloc[[-1]][features].replace([np.inf,-np.inf], np.nan)
    if last.isna().any(axis=None):
        return None, metric, baseline_metric
    # Refit a separate production model on every fully labelled observation.
    forecast_model = RandomForestClassifier(**params)
    forecast_model.fit(z[features], z.target.astype(int))
    prob = float(forecast_model.predict_proba(last)[0][list(forecast_model.classes_).index(1)])
    return prob, metric, baseline_metric

inst_data=get_institutional()
data={}; rows=[]; quotes=[]; history_status={}
with st.spinner("讀取歷史行情…"):
    for s in ss:
        d, history_info=getdata(s,period,finmind_token)
        history_status[s]=history_info
        if d.empty: continue
        d=ind(d); d["obvp"]=d.obv.shift(5); data[s]=d; r=d.iloc[-1]
        inst=inst_data.get(s); net=inst.get("三大法人買賣超股數") if inst else None
        base=20*(r.close>r.ema21 and r.up)+10*(r.close>r.ema50)+10*(r.close>r.ema200)+10*(r.r20>0)+5*(r.r60>0)+10*(50<=r.rsi<=70)+10*(r.macd>r.sig)+10*(pd.notna(r.v20) and r.volume>1.5*r.v20 and r.close>r.h20)+5*(r.obv>r.obvp if pd.notna(r.obvp) else False)
        score=base+(10 if net is not None and net>0 else 0)
        prob,metric,baseline_metric=train_ml(d); price,qt,age=get_quote(s)
        quote_status, quote_age = quote_freshness(qt)
        quotes.append({"代號":s,"名稱":W.get(s,s),"盤中參考價":price,"報價時間":qt,
                       "報價狀態":quote_status,"資料約幾分鐘前":quote_age})
        ai_beats_baseline = metric is not None and baseline_metric is not None and metric >= baseline_metric + 0.02
        signal = ("多因子觀察" if score>=70 and prob is not None and prob>=0.55 and ai_beats_baseline
                  else "技術候選（AI資料不足）" if score>=70 and prob is None
                  else "AI未優於基準" if score>=70 and not ai_beats_baseline
                  else "AI未確認" if score>=70 else "等待" if score>=50 else "不進場")
        rows.append({"代號":s,"名稱":W.get(s,s),"規則分數":int(score),"訊號":signal,"AI五日上漲機率":round(prob,3) if prob is not None else None,"模型測試平衡準確率":round(metric,3) if metric is not None else None,"多數類基準平衡準確率":round(baseline_metric,3) if baseline_metric is not None else None,"三大法人買賣超(股)":net,"法人資料日期":inst.get("資料日期") if inst else None,"RSI14":round(r.rsi,1) if pd.notna(r.rsi) else None,"20日報酬":r.r20,"60日報酬":r.r60,"日線日期":r.date.strftime("%Y-%m-%d")})
if not data: st.error("目前無法取得歷史行情，請稍後重試。"); st.stop()
t0,t1,t2,t3,t4,t5,t6,t7=st.tabs(["上市櫃股票庫","行情監控","歷史回測","模擬帳本","資料限制","多因子研究","整合研究中心","自動交易（模擬）"])
with t0:
    st.subheader("台灣上市／上櫃證券主檔")
    st.caption("資料來源："+universe_source+"。依每一證券代號最新市場別紀錄判斷是否仍為上市或上櫃，避免把已轉板股票誤列為興櫃。")
    if universe_error:
        st.error("全市場主檔目前無法更新："+universe_error)
        st.warning("目前顯示的是內建備援名單，不是完整上市櫃資料庫。請稍後按側欄「重新抓取」。")
    m1,m2,m3,m4=st.columns(4)
    m1.metric("主檔證券數",f"{universe_stats['總數']:,}")
    m2.metric("上市",f"{universe_stats['上市']:,}")
    m3.metric("上櫃",f"{universe_stats['上櫃']:,}")
    m4.metric("主檔資料日期",universe_stats["資料日期"] or "未取得")
    search_text=st.text_input("搜尋代號／公司名稱",placeholder="輸入 2330、台積電或公司名稱")
    market_filter=st.selectbox("市場篩選",["全部","上市","上櫃"])
    industries=sorted(stock_master["產業"].dropna().astype(str).unique().tolist())
    industry_filter=st.selectbox("產業篩選",["全部"]+industries)
    directory=stock_master.copy()
    if market_filter!="全部":
        directory=directory[directory["市場"]==market_filter]
    if industry_filter!="全部":
        directory=directory[directory["產業"]==industry_filter]
    if search_text.strip():
        query=search_text.strip().casefold()
        mask=(directory["代號"].astype(str).str.casefold().str.contains(query,regex=False)
              |directory["名稱"].astype(str).str.casefold().str.contains(query,regex=False))
        directory=directory[mask]
    st.write(f"符合條件：{len(directory):,} 檔")
    st.dataframe(directory,use_container_width=True,hide_index=True,height=560)
    st.download_button("下載完整上市櫃證券主檔 CSV",stock_master.to_csv(index=False).encode("utf-8-sig"),
                       "taiwan_listed_otc_stock_master.csv","text/csv")
    st.caption("目前主檔涵蓋現行上市與上櫃股票／ETF 等證券；不含興櫃、權證及歷史下市證券。這是每日更新的現行名冊，不等於歷史成分股資料庫。")
with t1:
    st.subheader("盤中參考行情（可能延遲）")
    quote_frame=pd.DataFrame(quotes)
    st.dataframe(quote_frame,use_container_width=True,hide_index=True)
    stale_labels={"盤中延遲／可能休市","非當日報價","無報價","時間異常"}
    stale_rows=quote_frame[quote_frame["報價狀態"].isin(stale_labels)] if not quote_frame.empty else pd.DataFrame()
    if not stale_rows.empty:
        st.warning("部分股票報價可能延遲、非當日或無資料。請先核對報價時間，不要將舊報價當成可成交價格。")
    if not quote_frame.empty and quote_frame["報價狀態"].eq("當日最後報價（非盤中）").any():
        st.info("目前不是台股一般盤中時段；表內為當日最後可取得報價，不代表盤後仍可成交。")
    st.caption("盤中參考價取自 Yahoo Finance 1 分鐘資料；更新狀態依台灣一般交易時段及報價時間估算，仍可能受休市日、來源延遲與資料缺漏影響。")
    st.subheader("技術面、AI 與法人觀察")
    st.dataframe(pd.DataFrame(rows).sort_values("規則分數",ascending=False),use_container_width=True,hide_index=True)
    st.caption("AI 測試平衡準確率同時列出多數類基準；若模型未明顯勝過基準，不應視為有預測優勢。機率尚未校準，不代表實際上漲機率保證。法人歷史序列、融資融券與分點籌碼尚未接入。")
    s=st.selectbox("查看個股",list(data),format_func=lambda z:z+" "+W.get(z,z)); d=data[s]
    f=go.Figure(); f.add_trace(go.Scatter(x=d.date,y=d.close,name="還原收盤")); f.add_trace(go.Scatter(x=d.date,y=d.ema21,name="EMA21")); f.add_trace(go.Scatter(x=d.date,y=d.ema50,name="EMA50")); st.plotly_chart(f,use_container_width=True)
    st.download_button("下載行情指標 CSV",pd.concat([v.assign(代號=k) for k,v in data.items()]).to_csv(index=False).encode("utf-8-sig"),"tw_stock.csv","text/csv")
with t2:
    st.subheader("歷史回測｜單股與多股票投資組合")
    backtest_mode = st.radio(
        "回測模式", ["多股票共享資金組合", "單一股票"],
        horizontal=True, key="historical_backtest_mode",
        help="多股票模式會讓所選股票共用同一筆本金、持倉上限及風險預算。",
    )
    if backtest_mode == "多股票共享資金組合":
        portfolio_symbols = st.multiselect(
            "選擇要一起回測的股票", options=list(data), default=list(data),
            format_func=lambda code: f"{code} {W.get(code, code)}",
            key="historical_portfolio_symbols",
        )
        p1, p2, p3 = st.columns(3)
        portfolio_strategy = p1.selectbox("投資策略", STRATEGIES, key="historical_portfolio_strategy")
        portfolio_max_positions = p2.slider(
            "最多同時持有檔數", 1, min(10, max(1, len(portfolio_symbols))),
            min(5, max(1, len(portfolio_symbols))), key="historical_portfolio_max_positions"
        )
        default_risk_pct = min(2.0, max(0.1, float(risk) / max(float(capital), 1) * 100))
        portfolio_risk_pct = p3.slider(
            "單筆風險預算（資產百分比）", 0.1, 2.0, float(default_risk_pct), 0.1,
            key="historical_portfolio_risk_pct"
        ) / 100
        portfolio_exit = st.selectbox(
            "出場／停損規則",
            [
                ("baseline_atr2", "ATR × 2（基準）"),
                ("oneil_8pct", "O’Neil 8% 參考停損"),
                ("minervini_structure", "Minervini 結構低點代理"),
                ("atr_1_5", "ATR × 1.5"),
                ("atr_2_5", "ATR × 2.5"),
                ("atr_3", "ATR × 3"),
            ],
            format_func=lambda item: item[1], key="historical_portfolio_exit",
        )
        st.caption(
            f"共選取 {len(portfolio_symbols)} 檔；共用本金 NTD {capital:,.0f}。"
            "訊號只使用當時已完成的歷史日線，下一交易日開盤模擬成交。"
        )
        portfolio_config = (
            tuple(sorted(portfolio_symbols)), portfolio_strategy, float(capital),
            int(portfolio_max_positions), float(portfolio_risk_pct), portfolio_exit[0], period,
        )
        if st.button("比較三種策略（相同條件）", key="compare_historical_portfolio_strategies", disabled=not portfolio_symbols):
            selected_frames = {code: data[code].copy() for code in portfolio_symbols if code in data}
            comparison_rows = []
            with st.spinner("以相同股票、期間、本金及風險設定比較三種策略…"):
                for strategy_name in STRATEGIES:
                    comparison = shared_capital_backtest(
                        selected_frames, strategy_name,
                        initial_capital=float(capital), max_positions=int(portfolio_max_positions),
                        risk_fraction=float(portfolio_risk_pct), warmup=260,
                        exit_profile=portfolio_exit[0],
                    )
                    cm = comparison["metrics"]
                    ct = trade_statistics(comparison["trades"])
                    comparison_rows.append({
                        "策略": strategy_name,
                        "期末資產": cm["期末資產"],
                        "總報酬": cm["總報酬"],
                        "年化報酬": cm["年化報酬"],
                        "最大回撤": cm["最大回撤"],
                        "交易筆數": ct["交易筆數"],
                        "勝率": ct["勝率"],
                        "Profit Factor": ct["Profit Factor"],
                        "Sharpe": cm.get("Sharpe", 0),
                    })
            st.session_state["historical_strategy_comparison"] = comparison_rows
            st.session_state["historical_strategy_comparison_config"] = portfolio_config
        if st.session_state.get("historical_strategy_comparison_config") == portfolio_config:
            st.subheader("三種策略公平比較")
            comparison_df = pd.DataFrame(st.session_state["historical_strategy_comparison"])
            st.dataframe(
                comparison_df.style.format({
                    "期末資產": "NTD {:,.0f}", "總報酬": "{:.1%}",
                    "年化報酬": "{:.1%}", "最大回撤": "{:.1%}",
                    "勝率": "{:.1%}", "Profit Factor": "{:.2f}", "Sharpe": "{:.2f}",
                }),
                use_container_width=True, hide_index=True,
            )
            st.caption("三種策略使用相同股票、期間、本金、持倉上限、風險預算及出場規則；此比較仍是歷史樣本內結果，不能據此直接選擇實盤策略。")
        if st.button("啟動 AI 自主策略演化", key="run_ai_strategy_evolution", disabled=not portfolio_symbols):
            evolution_frames = {code: data[code].copy() for code in portfolio_symbols if code in data}
            with st.spinner("正在搜尋策略參數、訓練前 70% 歷史資料，並保留後 30% 作樣本外驗證…"):
                evolution_result = evolve_strategy(
                    evolution_frames,
                    initial_capital=float(capital),
                    max_positions=int(portfolio_max_positions),
                    risk_fraction=float(portfolio_risk_pct),
                    exit_profile=portfolio_exit[0],
                    warmup=260,
                )
            st.session_state["ai_evolution_result"] = evolution_result
            st.session_state["ai_evolution_config"] = portfolio_config
        if st.session_state.get("ai_evolution_config") == portfolio_config:
            evolution = st.session_state["ai_evolution_result"]
            st.subheader("AI 自主策略演化研究")
            if evolution.get("status") != "ok":
                st.warning(
                    "目前資料不足以進行可靠的訓練／樣本外切分。"
                    f"可用共同交易日：{evolution.get('available_sessions', 0)}；"
                    f"至少需要：{evolution.get('required_sessions', 0)}。"
                )
            else:
                st.write(
                    f"搜尋 {evolution['search_count']} 組候選；"
                    f"使用 {evolution['symbols']} 檔股票；"
                    f"樣本外起始日：{pd.Timestamp(evolution['split_date']).date()}"
                )
                st.write("選出的候選參數（只依前 70% 訓練資料選擇）：", evolution["selected_params"])
                st.caption("樣本外後 30% 完全不參與參數挑選；它只用來檢查策略是否能延續。若樣本外績效不佳，應淘汰策略，而不是用樣本外資料反覆調參。")
                st.subheader("訓練期候選排名")
                st.dataframe(
                    evolution["train_rows"].style.format({
                        "訓練年化報酬": "{:.1%}", "訓練最大回撤": "{:.1%}",
                        "訓練Profit Factor": "{:.2f}", "研究分數": "{:.3f}",
                    }),
                    use_container_width=True, hide_index=True,
                )
                holdout = evolution["holdout"]
                hm = holdout["metrics"]
                hs = evolution["holdout_stats"]
                if holdout["equity"].empty:
                    st.warning("樣本外期間沒有足夠資料形成資產曲線。")
                else:
                    h1, h2, h3, h4, h5 = st.columns(5)
                    h1.metric("樣本外期末資產", f"NTD {hm['期末資產']:,.0f}")
                    h2.metric("樣本外總報酬", f"{hm['總報酬']:.1%}")
                    h3.metric("樣本外年化報酬", f"{hm['年化報酬']:.1%}")
                    h4.metric("樣本外最大回撤", f"{hm['最大回撤']:.1%}")
                    h5.metric("樣本外交易筆數", str(hs.get("交易筆數", len(holdout["trades"]))))
                    h6, h7, h8 = st.columns(3)
                    h6.metric("樣本外勝率", f"{hs.get('勝率', 0):.1%}")
                    h7.metric("樣本外 Profit Factor", "∞" if np.isinf(hs.get("Profit Factor", 0)) else f"{hs.get('Profit Factor', 0):.2f}")
                    h8.metric("樣本外 Sharpe", f"{hm.get('Sharpe', 0):.2f}")
                    annual = evolution["annual_performance"]
                    if annual.empty:
                        st.info("樣本外期間不足以形成完整年度報酬統計。")
                    else:
                        achieved = evolution["annual_100pct_years"]
                        st.write(f"每年 +100% 目標：{achieved} / {len(annual)} 個樣本外年度達成")
                        st.dataframe(
                            annual.style.format({
                                "年初資產（樣本外首日）": "NTD {:,.0f}",
                                "年底資產": "NTD {:,.0f}",
                                "年度報酬": "{:.1%}",
                            }),
                            use_container_width=True, hide_index=True,
                        )
                    st.line_chart(holdout["equity"].set_index("日期")["資產"])
                    st.dataframe(holdout["trades"], use_container_width=True, hide_index=True)
                    st.download_button(
                        "下載 AI 演化樣本外交易紀錄",
                        holdout["trades"].to_csv(index=False).encode("utf-8-sig"),
                        "ai_evolution_holdout_trades.csv", "text/csv",
                    )
                st.warning("這是有限候選空間的量化搜尋原型，不是能無限制自我學習的 AI。只有樣本外結果、跨年度穩定性及紙上交易持續通過，才可考慮升級；目前不會連接券商。")
        if st.button("執行 Walk-forward＋0050 基準比較", key="run_walkforward_benchmark", disabled=not portfolio_symbols):
            wf_frames = {code: data[code].copy() for code in portfolio_symbols if code in data}
            with st.spinner("執行多段樣本外驗證，並取得 0050 還原日線作基準…"):
                wf_result = rolling_walk_forward(
                    wf_frames, portfolio_strategy,
                    initial_capital=float(capital),
                    max_positions=int(portfolio_max_positions),
                    risk_fraction=float(portfolio_risk_pct),
                    lookback_days=756, test_days=126, step_days=126,
                    warmup_bars=260, exit_profile=portfolio_exit[0],
                )
                benchmark_raw, benchmark_info = getdata("0050", period, finmind_token)
                benchmark_frame = benchmark_raw.copy() if benchmark_raw is not None else pd.DataFrame()
                if not benchmark_frame.empty:
                    benchmark_frame["date"] = pd.to_datetime(benchmark_frame["date"])
                    benchmark_frame = benchmark_frame.sort_values("date").drop_duplicates("date")
                    benchmark_frame = benchmark_frame[["date", "close"]].dropna()
                benchmark_rows = []
                if not benchmark_frame.empty and not wf_result["windows"].empty:
                    prices = benchmark_frame.set_index("date")["close"]
                    for _, window in wf_result["windows"].iterrows():
                        start = pd.Timestamp(window["測試開始"])
                        end = pd.Timestamp(window["測試結束"])
                        prior = prices.loc[prices.index < start]
                        finish = prices.loc[prices.index <= end]
                        if prior.empty or finish.empty:
                            continue
                        base_price = float(prior.iloc[-1])
                        end_price = float(finish.iloc[-1])
                        benchmark_rows.append({
                            "區段": int(window["區段"]),
                            "測試開始": window["測試開始"],
                            "測試結束": window["測試結束"],
                            "0050買進持有報酬": end_price / base_price - 1.0,
                            "策略報酬": float(window.get("總報酬", 0.0)),
                        })
                benchmark_comparison = pd.DataFrame(benchmark_rows)
                st.session_state["walk_forward_result"] = wf_result
                st.session_state["walk_forward_benchmark"] = benchmark_comparison
                st.session_state["walk_forward_config"] = portfolio_config
        if st.session_state.get("walk_forward_config") == portfolio_config:
            wf = st.session_state["walk_forward_result"]
            st.subheader("Walk-forward 多段樣本外驗證")
            st.caption(wf.get("note", ""))
            if wf["windows"].empty:
                st.warning("目前歷史資料不足以形成有效的 Walk-forward 區段。")
            else:
                wm = wf["metrics"]
                w1, w2, w3, w4 = st.columns(4)
                w1.metric("鏈結樣本外總報酬", f"{wm.get('總報酬（鏈結）', 0):.1%}")
                w2.metric("鏈結最大回撤", f"{wm.get('最大回撤（鏈結）', 0):.1%}")
                w3.metric("鏈結 Sharpe", f"{wm.get('Sharpe（鏈結）', 0):.2f}")
                w4.metric("樣本外區段", str(len(wf["windows"])))
                st.dataframe(wf["windows"], use_container_width=True, hide_index=True)
                bench = st.session_state.get("walk_forward_benchmark", pd.DataFrame())
                if bench.empty:
                    st.warning("0050 歷史資料未取得或日期無法對齊；本次不顯示基準比較，不能把缺資料解讀為打敗大盤。")
                else:
                    st.subheader("各樣本外區段：策略 vs 0050 買進持有")
                    st.dataframe(
                        bench.style.format({
                            "0050買進持有報酬": "{:.1%}", "策略報酬": "{:.1%}",
                        }),
                        use_container_width=True, hide_index=True,
                    )
                    st.caption("0050 使用還原日線收盤價；每段以測試開始前最近收盤至測試結束收盤計算，未扣 ETF 買賣費用與稅負，僅作基準參考。")
                regime = regime_performance(
                    wf["equity"].rename(columns={"資產": "資產"}),
                    benchmark_frame.assign(date=benchmark_frame["date"]) if not benchmark_frame.empty else pd.DataFrame(),
                    initial_capital=float(capital),
                ) if not benchmark_frame.empty and not wf["equity"].empty else pd.DataFrame()
                if not regime.empty:
                    st.subheader("依 0050 市場環境拆分")
                    st.dataframe(regime, use_container_width=True, hide_index=True)
                st.download_button(
                    "下載 Walk-forward 區段報告",
                    wf["windows"].to_csv(index=False).encode("utf-8-sig"),
                    "walk_forward_windows.csv", "text/csv",
                )
        if st.button("執行多股票組合回測", key="run_historical_multi_portfolio", disabled=not portfolio_symbols):
            selected_frames = {code: data[code].copy() for code in portfolio_symbols if code in data}
            with st.spinner("使用共享資金池逐日模擬所有股票…"):
                result = shared_capital_backtest(
                    selected_frames, portfolio_strategy,
                    initial_capital=float(capital), max_positions=int(portfolio_max_positions),
                    risk_fraction=float(portfolio_risk_pct), warmup=260,
                    exit_profile=portfolio_exit[0],
                )
            st.session_state["historical_multi_result"] = result
            st.session_state["historical_multi_config"] = portfolio_config
        if st.session_state.get("historical_multi_config") == portfolio_config:
            result = st.session_state["historical_multi_result"]
            coverage = result.get("coverage", {})
            st.write(
                f"實際納入：{coverage.get('symbols_used', 0)} / "
                f"{coverage.get('symbols_requested', len(portfolio_symbols))} 檔"
            )
            if result["equity"].empty:
                st.warning("沒有足夠歷史資料可回測。請選擇較長期間（建議 5y），並確認所選股票至少有 203 個有效交易日。")
            else:
                metrics = result["metrics"]
                risk_metrics = equity_risk_metrics(result["equity"], float(capital))
                trade_metrics = trade_statistics(result["trades"])
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("期末資產", f"NTD {metrics['期末資產']:,.0f}")
                m2.metric("總報酬", f"{metrics['總報酬']:.1%}")
                m3.metric("年化報酬", f"{metrics['年化報酬']:.1%}")
                m4.metric("最大回撤", f"{metrics['最大回撤']:.1%}")
                m5.metric("交易筆數", str(trade_metrics["交易筆數"]))
                m6, m7, m8, m9 = st.columns(4)
                m6.metric("勝率", f"{trade_metrics['勝率']:.1%}")
                m7.metric("Profit Factor", "∞" if np.isinf(trade_metrics["Profit Factor"]) else f"{trade_metrics['Profit Factor']:.2f}")
                m8.metric("Sharpe", f"{metrics.get('Sharpe', 0):.2f}")
                m9.metric("Sortino", f"{risk_metrics.get('Sortino', 0):.2f}")
                st.subheader("共享資金組合資產曲線")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=result["equity"]["日期"], y=result["equity"]["資產"], name="組合總資產"))
                fig.add_hline(y=float(capital), line_dash="dash")
                fig.update_layout(yaxis_title="新台幣")
                st.plotly_chart(fig, use_container_width=True)
                trades = result["trades"]
                if not trades.empty:
                    st.subheader("各股票交易貢獻")
                    per_symbol = trades.groupby("代號").agg(
                        交易筆數=("損益", "count"), 累計損益=("損益", "sum"),
                        平均報酬=("報酬率", "mean"),
                        獲利筆數=("損益", lambda values: int((values > 0).sum())),
                    ).reset_index()
                    per_symbol["名稱"] = per_symbol["代號"].map(W).fillna(per_symbol["代號"])
                    st.dataframe(per_symbol[["代號", "名稱", "交易筆數", "累計損益", "平均報酬", "獲利筆數"]].sort_values("累計損益", ascending=False), use_container_width=True, hide_index=True)
                    st.subheader("完整模擬交易紀錄")
                    st.dataframe(trades, use_container_width=True, hide_index=True)
                    st.download_button("下載多股票回測交易 CSV", trades.to_csv(index=False).encode("utf-8-sig"), "multi_stock_backtest_trades.csv", "text/csv")
                else:
                    st.info("此期間沒有符合策略條件的成交訊號。")
                st.download_button("下載組合每日資產 CSV", result["equity"].to_csv(index=False).encode("utf-8-sig"), "multi_stock_backtest_equity.csv", "text/csv")
        elif "historical_multi_result" in st.session_state:
            st.info("回測條件已變更；請重新執行，避免沿用舊結果。")
        st.warning("回測限制：採日線訊號、次日開盤成交及估算手續費／證交稅／滑價；仍未完整涵蓋歷史下市股、歷史成分股、除權息現金流、停牌與漲跌停排隊。結果不是未來獲利保證。")
    else:
        s = st.selectbox("回測標的", list(data), key="bt")
        x = data[s].copy().reset_index(drop=True)
        x["entry"] = (x.close > x.ema21) & x.up & x.rsi.between(50, 70) & (x.macd > x.sig) & (x.volume > 1.5 * x.v20) & (x.close > x.h20)
        x["exit"] = (x.close < x.ema21) | (x.rsi > 75) | (x.macd < x.sig)
        cash = float(capital)
        qty = 0
        stop = 0
        trades = []
        curve = [{"日期": x.iloc[0].date, "資產": float(capital)}]
        for i in range(len(x) - 1):
            r, n = x.iloc[i], x.iloc[i + 1]
            op = float(n.open)
            if qty and (r.exit or n.low <= stop):
                px = min(op, stop) if n.low <= stop else op
                gross = qty * px * .999
                cash += gross * (1 - .001425 - .003)
                trades[-1]["損益"] = cash - trades[-1]["投入前現金"]
                trades[-1]["出場日"] = str(n.date.date())
                qty = 0
            if qty == 0 and r.entry:
                px = op * 1.001
                stop = px - 2 * (r.atr if pd.notna(r.atr) else px * .04)
                q = max(0, min(int(risk / max(px - stop, .01)), int(cash / (px * 1.001425))))
                if q:
                    trades.append({"進場日": str(n.date.date()), "股數": q, "進場價": px, "投入前現金": cash})
                    cash -= q * px * 1.001425
                    qty = q
            curve.append({"日期": n.date, "資產": cash + qty * float(n.close)})
        if qty:
            px = float(x.iloc[-1].close) * .999
            cash += qty * px * (1 - .001425 - .003)
            trades[-1]["損益"] = cash - trades[-1]["投入前現金"]
            trades[-1]["備註"] = "期末平倉"
            curve[-1]["資產"] = cash
        simple_equity = pd.DataFrame(curve)
        simple_risk = equity_risk_metrics(simple_equity, float(capital))
        simple_trade = trade_statistics(pd.DataFrame(trades))
        a, b, c, dcol, ecol = st.columns(5)
        a.metric("期末資產", f"NTD {cash:,.0f}")
        b.metric("交易筆數", len(trades))
        c.metric("勝率", f"{simple_trade['勝率']:.1%}" if trades else "無交易")
        dcol.metric("最大回撤", f"{simple_risk['最大回撤']:.1%}")
        ecol.metric("Profit Factor", "∞" if np.isinf(simple_trade["Profit Factor"]) else f"{simple_trade['Profit Factor']:.2f}")
        st.warning("簡化回測：收盤產生訊號、次日開盤成交，含估算手續費、證交稅及滑價；未完整處理除權息、停牌、漲跌停、流動性與下市股偏誤。")
        fig = go.Figure(go.Scatter(x=[item["日期"] for item in curve], y=[item["資產"] for item in curve]))
        fig.add_hline(y=capital, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pd.DataFrame(trades).drop(columns=["投入前現金"], errors="ignore"), use_container_width=True)
with t3:
    st.subheader("純模擬交易帳本")
    st.caption("目前帳本存在 Streamlit 工作階段，不會連接券商；工作階段或服務重啟時可能清空。請使用下方 JSON 備份功能保存與跨裝置還原。")
    if "paper_cash" not in st.session_state:
        st.session_state.paper_cash=float(capital)
        st.session_state.paper_positions={}
        st.session_state.paper_trades=[]
    ps=st.selectbox("模擬交易股票",list(data),format_func=lambda z:z+" "+W.get(z,z),key="paper_symbol")
    ref=float(data[ps].iloc[-1].close)
    shares=st.number_input("股數（可模擬零股）",min_value=1,value=100,step=1,key="paper_shares")
    st.write("參考價（最近日線收盤）：",f"{ref:,.2f}")
    buycol,sellcol=st.columns(2)
    if buycol.button("模擬買進"):
        cost=shares*ref*(1+0.001425)
        if cost<=st.session_state.paper_cash:
            st.session_state.paper_cash-=cost
            st.session_state.paper_positions[ps]=st.session_state.paper_positions.get(ps,0)+shares
            st.session_state.paper_trades.append({"時間":datetime.now(TZ).strftime("%Y-%m-%d %H:%M"),"代號":ps,"方向":"買進","股數":shares,"參考價":ref,"估算金額":cost})
        else:
            st.error("模擬現金不足。")
    if sellcol.button("模擬賣出"):
        held=st.session_state.paper_positions.get(ps,0)
        if shares<=held:
            gross=shares*ref
            proceeds=gross*(1-0.001425-0.003)
            st.session_state.paper_cash+=proceeds
            st.session_state.paper_positions[ps]=held-shares
            st.session_state.paper_trades.append({"時間":datetime.now(TZ).strftime("%Y-%m-%d %H:%M"),"代號":ps,"方向":"賣出","股數":shares,"參考價":ref,"估算金額":proceeds})
        else:
            st.error("持股不足，無法模擬賣出。")
    holdings=[]
    total=st.session_state.paper_cash
    missing_position_data=[]
    for code,q in st.session_state.paper_positions.items():
        if code not in data:
            if q:
                missing_position_data.append(code)
                holdings.append({"代號":code,"名稱":W.get(code,code),"股數":q,"參考收盤價":"未載入","市值":"未計入"})
            continue
        px=float(data[code].iloc[-1].close)
        value=q*px
        total+=value
        if q:
            holdings.append({"代號":code,"名稱":W.get(code,code),"股數":q,"參考收盤價":px,"市值":value})
    if total > 0:
        for holding in holdings:
            if isinstance(holding.get("市值"), (int, float, np.integer, np.floating)):
                holding["資產占比"] = holding["市值"] / total
    if missing_position_data:
        st.warning("以下持倉未在目前觀察清單載入，帳戶總資產暫未計入其市值："+", ".join(missing_position_data))
    a,b=st.columns(2)
    a.metric("模擬可用現金",f"NTD {st.session_state.paper_cash:,.0f}")
    b.metric("帳戶總資產",f"NTD {total:,.0f}")
    st.subheader("目前持倉")
    st.dataframe(pd.DataFrame(holdings).style.format({"資產占比":"{:.1%}","市值":"{:,.0f}","參考收盤價":"{:,.2f}"}),use_container_width=True,hide_index=True)
    if holdings:
        loaded_weights=[h.get("資產占比",0.0) for h in holdings if isinstance(h.get("資產占比"),(int,float,np.integer,np.floating))]
        if loaded_weights and max(loaded_weights)>=0.5:
            st.warning("單一持倉占帳戶總資產達 50% 以上，集中風險偏高。")
    st.subheader("模擬成交紀錄")
    st.dataframe(pd.DataFrame(st.session_state.paper_trades),use_container_width=True,hide_index=True)
    if st.session_state.paper_trades:
        st.download_button("下載模擬成交紀錄 CSV",pd.DataFrame(st.session_state.paper_trades).to_csv(index=False).encode("utf-8-sig"),"paper_trades.csv","text/csv")

    st.divider()
    st.subheader("帳本備份與跨裝置還原")
    st.caption("目前不提供雲端自動同步；可下載 JSON 備份，換裝置後再匯入。匯入會覆蓋目前模擬帳本。")
    backup={
        "version":1,
        "cash":float(st.session_state.paper_cash),
        "positions":st.session_state.paper_positions,
        "trades":st.session_state.paper_trades,
    }
    st.download_button(
        "下載完整帳本備份 JSON",
        json.dumps(backup,ensure_ascii=False,indent=2).encode("utf-8"),
        "paper_ledger_backup.json","application/json"
    )
    uploaded=st.file_uploader("選擇先前下載的帳本 JSON",type=["json"],key="ledger_upload")
    if uploaded is not None and st.button("匯入並覆蓋目前帳本",key="ledger_restore"):
        try:
            if uploaded.size > 5 * 1024 * 1024:
                raise ValueError("備份檔不得超過 5 MB")
            payload=json.loads(uploaded.getvalue().decode("utf-8"))
            if not isinstance(payload,dict) or payload.get("version") != 1:
                raise ValueError("不支援的帳本版本")
            cash_value=float(payload["cash"])
            raw_positions=payload["positions"]
            trades_value=payload["trades"]
            if not np.isfinite(cash_value) or cash_value < 0:
                raise ValueError("現金必須是有限且不小於零的數值")
            if not isinstance(raw_positions,dict) or not isinstance(trades_value,list) or len(trades_value)>10000:
                raise ValueError("持倉或成交紀錄格式不正確")
            positions_value={}
            for code,value in raw_positions.items():
                code=str(code)
                quantity=int(value)
                if not code.isdigit() or not 4<=len(code)<=6 or quantity<0 or quantity>100000000:
                    raise ValueError("持倉代號或股數不合法")
                positions_value[code]=quantity
            required_trade_fields={"時間","代號","方向","股數","參考價","估算金額"}
            clean_trades=[]
            for trade in trades_value:
                if not isinstance(trade,dict) or not required_trade_fields.issubset(trade):
                    raise ValueError("成交紀錄欄位不完整")
                if not str(trade["代號"]).isdigit() or not 4<=len(str(trade["代號"]))<=6:
                    raise ValueError("成交紀錄股票代號不合法")
                if trade["方向"] not in ("買進","賣出"):
                    raise ValueError("成交方向不合法")
                quantity=int(trade["股數"])
                price=float(trade["參考價"])
                amount=float(trade["估算金額"])
                if quantity<=0 or not np.isfinite(price) or not np.isfinite(amount) or price<=0 or amount<0:
                    raise ValueError("成交股數、價格或金額不合法")
                clean_trades.append({
                    "時間":str(trade["時間"])[:32],"代號":str(trade["代號"]),
                    "方向":trade["方向"],"股數":quantity,"參考價":price,"估算金額":amount
                })
            st.session_state.paper_cash=cash_value
            st.session_state.paper_positions=positions_value
            st.session_state.paper_trades=clean_trades
            st.success("帳本已匯入目前工作階段。")
        except Exception as exc:
            st.error(f"匯入失敗：{exc}")
with t4:
    cache_stats=price_cache_summary()
    st.subheader("歷史行情資料庫狀態")
    c1,c2,c3=st.columns(3)
    c1.metric("SQLite 快取日線",f"{cache_stats['資料列數']:,}")
    c2.metric("已快取標的",f"{cache_stats['已快取標的']:,}")
    c3.metric("資料庫最新交易日",cache_stats["最新交易日"] or "尚無資料")
    st.caption("歷史日線優先使用 FinMind TaiwanStockPriceAdj（還原日線），每檔以最近 10 個日曆日重疊區間增量更新；API 失敗時使用可用快取，首次無快取才以 Yahoo Finance 還原日線備援。")
    status_rows=[]
    for code,info in history_status.items():
        status_rows.append({"代號":code,"來源狀態":info.get("來源狀態"),
                            "最後交易日":info.get("最後交易日"),
                            "更新錯誤":info.get("更新錯誤")})
    st.dataframe(pd.DataFrame(status_rows),use_container_width=True,hide_index=True)
    st.warning("重要：SQLite 目前是 Streamlit 執行環境的本機快取。Streamlit Community Cloud 重新部署或重啟可能清除檔案；尚未連接永久外部資料庫，因此不能宣稱為永久全市場歷史資料庫。") 
    st.markdown("盤中參考價可能延遲，非交易所認證即時行情。三大法人數據使用證交所 T86 日報，取最近可取得交易日；API 不可用時顯示空值。尚未接入法人歷史序列、融資融券與分點籌碼。")
    st.markdown("AI 欄位使用隨機森林，以歷史技術特徵預測未來 5 個交易日報酬是否超過 1.5%，按時間順序切分訓練與測試資料。平衡準確率是歷史測試指標，不代表未來勝率或獲利保證。")
    st.markdown("回測成本假設：手續費 0.1425%、一般股票賣出稅 0.3%、滑價 0.1%。尚未完整處理除權息、停牌、漲跌停、流動性與下市股偏誤。不連接券商、不會下單。")
    st.subheader("觀察清單資料品質")
    quality=[]
    today=pd.Timestamp(datetime.now(TZ).date())
    for code,z in data.items():
        dates=pd.to_datetime(z["date"],errors="coerce")
        gaps=dates.sort_values().diff().dt.days
        last_date=dates.max()
        age=(today-last_date.normalize()).days if pd.notna(last_date) else None
        bad_prices=((z[["open","high","low","close"]]<=0).any(axis=1)).sum()
        missing=int(z[["date","open","high","low","close","volume"]].isna().any(axis=1).sum())
        quality.append({
            "代號":code,"名稱":W.get(code,code),"資料筆數":len(z),
            "起始日期":dates.min().strftime("%Y-%m-%d") if dates.notna().any() else None,
            "最新日期":last_date.strftime("%Y-%m-%d") if pd.notna(last_date) else None,
            "距今天數":age,"超過7日資料間隔":int((gaps>7).sum()),
            "缺漏列":missing,"非正價格列":int(bad_prices),
            "狀態":"檢查" if missing or bad_prices or age is None or age>7 else "可用（仍需人工核對）"
        })
    st.dataframe(pd.DataFrame(quality),use_container_width=True,hide_index=True)
    st.caption("資料品質檢查只標示缺漏、日期間隔與價格異常；不代表資料來源完整或交易所認證。Yahoo Finance 自動調整日線仍可能有修訂、缺漏與公司行動處理差異。")
    st.write("台北時間：",datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"))


with t5:
    st.subheader("投資大師策略：單股策略比較")
    st.caption("比較三種技術策略代理。所有策略共用研究引擎，訊號在收盤後形成、下一交易日開盤成交，包含估算成本與 ATR 停損。")
    bs=st.selectbox("策略回測標的",list(data),format_func=lambda z:z+" "+W.get(z,z),key="master_bt")
    initial=st.number_input("策略回測本金",100000,10000000,1000000,100000,key="master_capital")
    frame=data[bs].copy().reset_index(drop=True)
    if st.button("執行單股策略比較",key="run_single_master"):
        results=[]; curves={}; trade_tables={}
        for strategy in STRATEGIES:
            result=shared_capital_backtest(
                {bs:frame},strategy,initial_capital=float(initial),
                max_positions=1,risk_fraction=min(max(float(risk)/float(initial),0.0001),0.10),warmup=0
            )
            metrics=result["metrics"].copy()
            trades=result["trades"]
            metrics["策略"]=strategy
            metrics.update(equity_risk_metrics(result["equity"],float(initial)))
            metrics.update(trade_statistics(trades))
            results.append(metrics)
            curves[strategy]=result["equity"]
            trade_tables[strategy]=trades
        st.session_state["single_master_results"]=pd.DataFrame(results)
        st.session_state["single_master_curves"]=curves
        st.session_state["single_master_trades"]=trade_tables
        st.session_state["single_master_config"]=(bs,float(initial),period)
    current_single_config=(bs,float(initial),period)
    if st.session_state.get("single_master_config")==current_single_config:
        result_df=st.session_state["single_master_results"]
        st.dataframe(result_df[["策略","期末資產","總報酬","年化報酬","最大回撤","年化波動率","Sharpe","Sortino","Calmar","平均曝險","交易筆數","勝率","平均獲利","平均虧損","Profit Factor","期望值","平均持有日數"]].style.format({
            "期末資產":"{:,.0f}","總報酬":"{:.1%}","年化報酬":"{:.1%}",
            "最大回撤":"{:.1%}","年化波動率":"{:.1%}","Sharpe":"{:.2f}",
            "Sortino":"{:.2f}","Calmar":"{:.2f}","平均曝險":"{:.1%}","勝率":"{:.1%}",
            "平均獲利":"{:,.0f}","平均虧損":"{:,.0f}","Profit Factor":lambda x:"∞" if np.isinf(x) else f"{x:.2f}",
            "期望值":"{:,.0f}","平均持有日數":"{:.1f}",
        }),use_container_width=True,hide_index=True)
        fig=go.Figure()
        for strategy,equity in st.session_state["single_master_curves"].items():
            if not equity.empty:
                fig.add_trace(go.Scatter(x=equity["日期"],y=equity["資產"],name=strategy))
        fig.add_hline(y=initial,line_dash="dash")
        fig.update_layout(title="單股策略資產曲線（統一回測引擎）",yaxis_title="新台幣")
        st.plotly_chart(fig,use_container_width=True)
        chosen=st.selectbox("查看交易明細",STRATEGIES,key="master_trade_detail")
        st.dataframe(st.session_state["single_master_trades"][chosen],use_container_width=True,hide_index=True)
    elif "single_master_results" in st.session_state:
        st.info("標的、期間或本金已變更；請重新執行策略比較。")
    st.markdown("**同一進場策略的出場規則比較**")
    st.caption("固定同一標的、進場訊號、初始本金、帳戶風險預算與交易成本；各規則依自己的初始停損距離計算股數，因此每筆預估價格風險一致，但實際股數會不同。ATR 依前一交易日收盤資料更新移動停損，下一交易日生效；O’Neil 達 +20% 只標記獲利檢視並採較寬 3 ATR 追蹤，不硬性停利；Minervini 達 2R 將停損推至成本價、達 3R 啟動 2.5 ATR 追蹤。這些是待驗證的代理規則，並非完整複製大師策略。")
    exit_entry_strategy=st.selectbox("固定進場策略",STRATEGIES,key="exit_compare_entry")
    exit_profiles=[
        ("baseline_atr2","原始 ATR × 2"),
        ("oneil_8pct","O’Neil 8% 參考停損"),
        ("minervini_structure","Minervini 10 日結構低點代理"),
        ("atr_1_5","ATR × 1.5"),
        ("atr_2","ATR × 2"),
        ("atr_2_5","ATR × 2.5"),
        ("atr_3","ATR × 3"),
    ]
    if st.button("比較不同出場規則",key="run_exit_profile_compare"):
        profile_rows=[]; profile_curves={}; profile_trades={}
        for profile_key,profile_label in exit_profiles:
            result=shared_capital_backtest(
                {bs:frame},exit_entry_strategy,initial_capital=float(initial),
                max_positions=1,risk_fraction=min(max(float(risk)/float(initial),0.0001),0.10),
                warmup=0,exit_profile=profile_key
            )
            m=result["metrics"].copy()
            m.update(equity_risk_metrics(result["equity"],float(initial)))
            m.update(trade_statistics(result["trades"]))
            m["出場規則"]=profile_label
            m["交易筆數"]=len(result["trades"])
            profile_rows.append(m)
            profile_curves[profile_label]=result["equity"]
            profile_trades[profile_label]=result["trades"]
        st.session_state["exit_profile_results"]=pd.DataFrame(profile_rows)
        st.session_state["exit_profile_curves"]=profile_curves
        st.session_state["exit_profile_trades"]=profile_trades
        st.session_state["exit_profile_config"]=(bs,exit_entry_strategy,float(initial),float(risk),period)
    exit_config=(bs,exit_entry_strategy,float(initial),float(risk),period)
    if st.session_state.get("exit_profile_config")==exit_config:
        profile_df=st.session_state["exit_profile_results"]
        cols=["出場規則","期末資產","總報酬","年化報酬","最大回撤","Sharpe","年化波動率","交易筆數","勝率","Profit Factor","期望值"]
        st.dataframe(profile_df[cols].style.format({
            "期末資產":"{:,.0f}","總報酬":"{:.1%}","年化報酬":"{:.1%}",
            "最大回撤":"{:.1%}","Sharpe":"{:.2f}","年化波動率":"{:.1%}",
            "勝率":"{:.1%}","Profit Factor":lambda x:"∞" if np.isinf(x) else f"{x:.2f}",
            "期望值":"{:,.0f}",
        }),use_container_width=True,hide_index=True)
        profile_fig=go.Figure()
        for label,equity in st.session_state["exit_profile_curves"].items():
            if not equity.empty:
                profile_fig.add_trace(go.Scatter(x=equity["日期"],y=equity["資產"],name=label))
        profile_fig.add_hline(y=initial,line_dash="dash")
        profile_fig.update_layout(title="固定進場訊號：不同出場規則資產曲線",yaxis_title="新台幣")
        st.plotly_chart(profile_fig,use_container_width=True)
        selected_profile=st.selectbox("查看出場規則交易明細",list(st.session_state["exit_profile_trades"]),key="exit_profile_detail")
        st.dataframe(st.session_state["exit_profile_trades"][selected_profile],use_container_width=True,hide_index=True)
    st.warning("Minervini、O’Neil、Livermore 為公開理念的技術代理，不是完整複製原策略。回測仍未完整處理除權息、停牌、漲跌停排隊、流動性衝擊及下市股偏誤。");


with t6:
    st.subheader("整合研究中心")

    st.markdown("**市場基準比較與成本敏感度**")
    st.caption("以 0050 還原價格（可投資 ETF）及加權指數價格指數作比較。策略與基準使用相同的完整歷史或樣本外日期區間；基準未扣交易成本、股息稅務與追蹤誤差，加權指數為價格報酬、不含股息。")
    benchmark_period=st.selectbox("分析區間",["樣本外區段","完整歷史期間"],index=0,key="benchmark_period")
    benchmark_button=st.button("執行基準比較與成本敏感度",key="run_benchmark_cost")
    if benchmark_button:
        calendar_all=sorted(set().union(*[
            set(pd.to_datetime(z["date"])) for z in data.values() if not z.empty
        ]))
        if not calendar_all:
            st.warning("觀察清單沒有共同日期資料，無法比較。")
        else:
            train_ratio=float(st.session_state.get("wf_train",60))/100
            valid_ratio=float(st.session_state.get("wf_valid",20))/100
            risk_fraction=float(st.session_state.get("wf_risk_percent",0.5))/100
            analysis_capital=float(st.session_state.get("wf_capital",capital))
            if benchmark_period=="樣本外區段":
                start_idx=min(len(calendar_all)-1,int(len(calendar_all)*(train_ratio+valid_ratio)))
                analysis_start=pd.Timestamp(calendar_all[start_idx])
                analysis_end=pd.Timestamp(calendar_all[-1])
                analysis_frames={}
                for code,z in data.items():
                    zz=z.copy().sort_values("date").reset_index(drop=True)
                    dates=pd.to_datetime(zz["date"])
                    history=zz.loc[dates<analysis_start].tail(260)
                    active=zz.loc[(dates>=analysis_start)&(dates<=analysis_end)]
                    if len(active)>=2:
                        analysis_frames[code]=pd.concat([history,active],ignore_index=True)
                report_start=analysis_start
            else:
                analysis_start=pd.Timestamp(calendar_all[0])
                analysis_end=pd.Timestamp(calendar_all[-1])
                analysis_frames=data
                report_start=None
            min_history=min((len(z) for z in analysis_frames.values()),default=0)
            warmup_n=200 if min_history<263 else 260
            bench_rows=[]; bench_curves={}
            for code,label in [("0050","0050 買進持有"),("^TWII","加權指數價格報酬")]:
                try:
                    if code=="0050":
                        raw_b=getdata("0050",period)
                    else:
                        raw_b=yf.Ticker("^TWII").history(period=period,interval="1d",auto_adjust=True)
                        if not raw_b.empty:
                            raw_b=raw_b.reset_index().rename(columns={"Date":"date","Datetime":"date","Close":"close"})
                            raw_b["date"]=pd.to_datetime(raw_b["date"]).dt.tz_localize(None)
                            raw_b=raw_b[["date","close"]].dropna()
                    if raw_b is None or raw_b.empty:
                        continue
                    b=raw_b.copy()
                    if "close" not in b:
                        b=b.rename(columns={"Close":"close","Date":"date"})
                    b["date"]=pd.to_datetime(b["date"]).dt.tz_localize(None)
                    b=b[(b["date"]>=analysis_start)&(b["date"]<=analysis_end)].sort_values("date")
                    if len(b)<2:
                        continue
                    base=float(b["close"].iloc[0])
                    b["資產"]=analysis_capital*b["close"]/base
                    peak=b["資產"].cummax()
                    dd=float((b["資產"]/peak-1).min())
                    total=float(b["資產"].iloc[-1]/analysis_capital-1)
                    years=max((b["date"].iloc[-1]-b["date"].iloc[0]).days/365.25,1/252)
                    bench_rows.append({"比較標的":label,"起始日":b["date"].iloc[0].date(),"結束日":b["date"].iloc[-1].date(),"總報酬":total,"年化報酬":(1+total)**(1/years)-1,"最大回撤":dd,"資料筆數":len(b)})
                    bench_curves[label]=b[["date","資產"]].rename(columns={"date":"日期"})
                except Exception as exc:
                    st.warning(f"{label} 資料取得失敗：{exc}")
            stress_rows=[]; strategy_curves={}
            for strategy in STRATEGIES:
                for slip,label in [(0.001,"單邊滑價 0.10%"),(0.002,"單邊滑價 0.20%"),(0.003,"單邊滑價 0.30%")]:
                    result=shared_capital_backtest(
                        analysis_frames,strategy,initial_capital=analysis_capital,
                        max_positions=int(st.session_state.get("wf_max_positions",5)),
                        risk_fraction=risk_fraction,
                        fee=0.001425,sell_tax=0.003,slippage=slip,
                        warmup=warmup_n,report_start_date=report_start
                    )
                    m=result["metrics"]
                    stress_rows.append({"策略":strategy,"成本情境":label,"期末資產":m["期末資產"],"總報酬":m["總報酬"],"年化報酬":m["年化報酬"],"最大回撤":m["最大回撤"],"Sharpe":m["Sharpe"],"交易筆數":len(result["trades"])})
                    if slip==0.001:
                        strategy_curves[strategy]=result["equity"][["日期","資產"]].copy()
            st.session_state["benchmark_rows"]=pd.DataFrame(bench_rows)
            st.session_state["benchmark_curves"]=bench_curves
            st.session_state["strategy_benchmark_curves"]=strategy_curves
            st.session_state["cost_stress_rows"]=pd.DataFrame(stress_rows)
            st.session_state["benchmark_config"]=(period,tuple(sorted(data.keys())),analysis_capital,benchmark_period,train_ratio,valid_ratio,int(st.session_state.get("wf_max_positions",5)),risk_fraction)
    current_benchmark_config=(period,tuple(sorted(data.keys())),float(st.session_state.get("wf_capital",capital)),benchmark_period,float(st.session_state.get("wf_train",60))/100,float(st.session_state.get("wf_valid",20))/100,int(st.session_state.get("wf_max_positions",5)),float(st.session_state.get("wf_risk_percent",0.5))/100)
    if st.session_state.get("benchmark_config")==current_benchmark_config:
        if not st.session_state["benchmark_rows"].empty:
            st.subheader("策略與市場基準資產曲線")
            strategy_curve_pick=st.selectbox("比較策略",STRATEGIES,key="benchmark_strategy_pick")
            bf=go.Figure()
            selected_curve=st.session_state["strategy_benchmark_curves"].get(strategy_curve_pick,pd.DataFrame())
            if not selected_curve.empty:
                bf.add_trace(go.Scatter(x=selected_curve["日期"],y=selected_curve["資產"],name=strategy_curve_pick))
            for label,curve in st.session_state["benchmark_curves"].items():
                bf.add_trace(go.Scatter(x=curve["日期"],y=curve["資產"],name=label))
            bf.add_hline(y=float(st.session_state.get("wf_capital",capital)),line_dash="dash")
            bf.update_layout(title="相同區間策略與基準比較（基準未扣成本）",yaxis_title="新台幣")
            st.plotly_chart(bf,use_container_width=True)
            st.dataframe(st.session_state["benchmark_rows"].style.format({"總報酬":"{:.1%}","年化報酬":"{:.1%}","最大回撤":"{:.1%}"}),use_container_width=True,hide_index=True)
        else:
            st.warning("未取得足夠的 0050 或加權指數資料，策略成本測試仍可參考。")
        st.subheader("滑價成本敏感度")
        st.dataframe(st.session_state["cost_stress_rows"].style.format({"期末資產":"{:,.0f}","總報酬":"{:.1%}","年化報酬":"{:.1%}","最大回撤":"{:.1%}","Sharpe":"{:.2f}"}),use_container_width=True,hide_index=True)
        st.caption("三種成本情境固定手續費 0.1425%、一般股票賣出稅 0.3%，僅將單邊滑價由 0.1% 提高至 0.3%。分析區間與選定樣本外／完整歷史一致；ETF 稅率、券商折扣、最低手續費與市場衝擊仍未完整納入。")
    st.markdown("**基本面因子研究（最新可取得快照）**")
    st.caption("以下是目前資料來源回傳的最新快照，不是歷史逐期財報；僅供當下橫向觀察，絕不回填歷史回測。")
    if st.button("載入基本面資料",key="fund_load"):
        with st.spinner("讀取財務資料…"):
            st.session_state["fund_data"]=fundamental_snapshot(tuple(data.keys()))
    if "fund_data" in st.session_state:
        fund=score_fundamental_snapshot(st.session_state["fund_data"].copy())
        st.dataframe(
            fund.sort_values("基本面分數",ascending=False,na_position="last"),
            use_container_width=True,hide_index=True
        )
        st.caption("基本面分數依目前觀察清單內可取得欄位做橫向百分位排名；缺值不當作零分，並顯示資料完整度。不同股票池的分數不可直接跨池比較。")
        industry_rank = enrich_industry_comparison(fund, stock_master)
        if not industry_rank.empty:
            st.markdown("**產業同業比較（最新快照）**")
            st.dataframe(industry_rank, use_container_width=True, hide_index=True)
            st.caption("產業排名僅比較本次已載入且有分數的同業樣本；不是完整產業全體排名，也不回填歷史回測。")

    st.markdown("**歷史月營收與籌碼資料**")
    st.caption("按需向 FinMind API 讀取個股歷史月營收、三大法人、融資融券及借券資料；每次最多 10 檔，避免免費 API 額度被大量請求耗盡。")
    chip_symbols = list(data.keys())[:10]
    if st.button("載入前 10 檔歷史月營收／籌碼", key="load_historical_chip_data", disabled=not chip_symbols):
        loaded, failed = {}, {}
        with st.spinner("讀取歷史月營收、法人、融資融券與借券資料…"):
            for symbol in chip_symbols:
                result = load_symbol_history(symbol, next_start_date(5), finmind_token)
                loaded[symbol] = result
                if result["errors"]:
                    failed[symbol] = result["errors"]
        st.session_state["historical_chip_data"] = loaded
        st.session_state["historical_chip_errors"] = failed
    if "historical_chip_data" in st.session_state:
        loaded = st.session_state["historical_chip_data"]
        st.caption(f"資料來源：FinMind API v4；載入時間：{datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}；股票數：{len(loaded)}")
        chip_rows, margin_rows, revenue_rows = [], [], []
        for symbol, result in loaded.items():
            inst = result["data"].get("institutional", pd.DataFrame())
            if not inst.empty:
                inst2 = inst.copy()
                inst2["買賣超"] = pd.to_numeric(inst2["buy"], errors="coerce") - pd.to_numeric(inst2["sell"], errors="coerce")
                inst2["法人"] = inst2["name"].astype(str)
                latest = inst2.sort_values("date").groupby("法人").tail(1)
                for _, row in latest.iterrows():
                    chip_rows.append({"代號": symbol, "日期": row["date"], "法人": row["法人"], "買賣超（股）": row["買賣超"]})
            margin = result["data"].get("margin_short", pd.DataFrame())
            if not margin.empty:
                row = margin.sort_values("date").iloc[-1].to_dict()
                row["代號"] = symbol
                margin_rows.append(row)
            revenue = result["data"].get("monthly_revenue", pd.DataFrame())
            if not revenue.empty:
                revenue = revenue.copy()
                revenue["代號"] = symbol
                revenue_rows.append(revenue)
        if chip_rows:
            st.markdown("**三大法人歷史買賣超（最近可取得日）**")
            st.dataframe(pd.DataFrame(chip_rows), use_container_width=True, hide_index=True)
        else:
            st.info("目前未取得三大法人歷史資料。")
        if margin_rows:
            st.markdown("**融資融券最新資料**")
            st.dataframe(pd.DataFrame(margin_rows), use_container_width=True, hide_index=True)
        else:
            st.info("目前未取得融資融券資料。")
        if revenue_rows:
            revenue_all = pd.concat(revenue_rows, ignore_index=True)
            st.markdown("**歷史月營收**")
            st.dataframe(revenue_all.sort_values(["代號", "date"], ascending=[True, False]), use_container_width=True, hide_index=True)
            st.caption("FinMind create_time 自 2026-04-21 起才記錄，且只是資料庫寫入日近似值、不是官方公告日；更早月份仍可看歷史數字，但不可當成當時已知資料做回測。")
        else:
            st.info("目前未取得月營收資料。")
        errors = st.session_state.get("historical_chip_errors", {})
        if errors:
            with st.expander("資料來源／額度錯誤明細"):
                st.json(errors)
        st.caption("三大法人、融資融券與借券日資料以交易日後一個日曆日作保守可得日；這是研究假設，不代表精確公告時間。借券資料若 API 方案或資料集不可用，會明確顯示錯誤，不會補零。")


    st.divider()
    st.markdown("**多股票共享資金池：訓練／驗證／樣本外**")
    st.caption("各策略僅用日線技術資料產生訊號；同一個投資組合共用現金、持倉上限及成本假設。")
    train_pct=st.slider("訓練資料比例",50,70,60,5,key="wf_train")/100
    valid_pct=st.slider("驗證資料比例",10,25,20,5,key="wf_valid")/100
    if train_pct + valid_pct >= 0.9:
        st.error("訓練與驗證比例總和必須小於 90%，以保留至少 10% 樣本外資料。")
    else:
        pc=st.number_input("組合測試本金",100000,10000000,1000000,100000,key="wf_capital")
        maxpos=st.slider("最多同時持有股票數",1,10,5,key="wf_max_positions")
        risk_pct=st.slider("單筆風險預算（資產百分比）",0.1,2.0,0.5,0.1,key="wf_risk_percent")/100
        st.caption("單筆風險預算依進場價至 ATR 停損價的距離估算；實際虧損可能因跳空、流動性及滑價超過預算。三段期間依共同交易日曆切分；每段開始會清空前段持倉並以本金重新開始。")
        current_portfolio_config=(float(train_pct),float(valid_pct),float(pc),int(maxpos),float(risk_pct),tuple(sorted(data.keys())),period)
        if st.button("執行全觀察清單組合回測",key="run_shared_portfolio"):
            summaries=[]; curve_map={}; trade_map={}
            # Split by a shared calendar, not by each symbol's row number.
            calendar=sorted(set().union(*[
                set(pd.to_datetime(z["date"])) for z in data.values() if not z.empty
            ]))
            n_total=len(calendar)
            train_end=int(n_total*train_pct)
            valid_end=int(n_total*(train_pct+valid_pct))
            segments=[
                ("訓練",0,train_end),
                ("驗證",train_end,valid_end),
                ("樣本外",valid_end,n_total),
            ]
            for strategy in STRATEGIES:
                periods={}
                for label,lo,hi in segments:
                    if hi <= lo:
                        continue
                    segment_start=pd.Timestamp(calendar[lo])
                    segment_end=pd.Timestamp(calendar[hi-1])
                    segment_frames={}
                    for code,z in data.items():
                        zz=z.copy().sort_values("date").reset_index(drop=True)
                        dates=pd.to_datetime(zz["date"])
                        history=zz.loc[dates < segment_start].tail(260)
                        active=zz.loc[(dates >= segment_start)&(dates <= segment_end)]
                        if len(active) < 40:
                            continue
                        # Keep prior observations for indicators, but report and
                        # trade only from the segment's common start date onward.
                        segment_frames[code]=pd.concat([history,active],ignore_index=True)
                    result=shared_capital_backtest(
                        segment_frames,strategy,initial_capital=float(pc),
                        max_positions=int(maxpos),risk_fraction=float(risk_pct),warmup=260,
                        report_start_date=segment_start
                    ) if segment_frames else None
                    if result:
                        periods[label]=result
                for label,result in periods.items():
                    m=result["metrics"]
                    risk_stats=equity_risk_metrics(result["equity"],float(pc))
                    trade_stats=trade_statistics(result["trades"])
                    summaries.append({
                        "策略":strategy,"期間":label,"期末資產":m["期末資產"],
                        "總報酬":m["總報酬"],"年化報酬":m["年化報酬"],
                        "最大回撤":m["最大回撤"],"Sharpe":m["Sharpe"],
                        "年化波動率":risk_stats["年化波動率"],
                        "Sortino":risk_stats["Sortino"],"Calmar":risk_stats["Calmar"],
                        "平均曝險":risk_stats["平均曝險"],
                        "交易筆數":trade_stats["交易筆數"],"勝率":trade_stats["勝率"],
                        "平均獲利":trade_stats["平均獲利"],"平均虧損":trade_stats["平均虧損"],
                        "Profit Factor":trade_stats["Profit Factor"],
                        "期望值":trade_stats["期望值"],
                        "平均持有日數":trade_stats["平均持有日數"],
                        "有效股票數":result["coverage"]["symbols_used"],
                    })
                    curve_map[(strategy,label)]=result["equity"]
                    trade_map[(strategy,label)]=result["trades"]
            st.session_state["portfolio_summary"]=pd.DataFrame(summaries)
            st.session_state["portfolio_curves"]=curve_map
            st.session_state["portfolio_trades"]=trade_map
            st.session_state["portfolio_config"]=current_portfolio_config

        if "portfolio_summary" in st.session_state and st.session_state.get("portfolio_config")==current_portfolio_config:
            summary=st.session_state["portfolio_summary"]
            st.dataframe(summary.style.format({
                "期末資產":"{:,.0f}","總報酬":"{:.1%}","年化報酬":"{:.1%}",
                "最大回撤":"{:.1%}","年化波動率":"{:.1%}","Sharpe":"{:.2f}",
                "Sortino":"{:.2f}","Calmar":"{:.2f}","平均曝險":"{:.1%}",
                "勝率":"{:.1%}","Profit Factor":lambda x:"∞" if np.isinf(x) else f"{x:.2f}",
                "期望值":"{:,.0f}","平均持有日數":"{:.1f}",
            }),use_container_width=True,hide_index=True)
            strategy_pick=st.selectbox("查看策略資產曲線",STRATEGIES,key="portfolio_curve_strategy")
            period_pick=st.selectbox("查看驗證區段",["訓練","驗證","樣本外"],index=2,key="portfolio_curve_period")
            curve=st.session_state["portfolio_curves"].get((strategy_pick,period_pick),pd.DataFrame())
            if not curve.empty:
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=curve["日期"],y=curve["資產"],name="組合資產"))
                fig.add_hline(y=float(pc) if "pc" in locals() else 1000000,line_dash="dash")
                st.plotly_chart(fig,use_container_width=True)
            trades=st.session_state["portfolio_trades"].get((strategy_pick,period_pick),pd.DataFrame())
            st.subheader("組合成交明細")
            st.dataframe(trades,use_container_width=True,hide_index=True)
            if not trades.empty:
                st.download_button(
                    "下載組合交易紀錄 CSV",
                    trades.to_csv(index=False).encode("utf-8-sig"),
                    "portfolio_trades.csv","text/csv"
                )
            st.caption("切分依觀察清單共同交易日期進行；各股票僅使用該日以前資料暖機，績效只從各區段起始日計算。觀察清單仍不代表全台上市櫃股票。")
        elif "portfolio_summary" in st.session_state:
            st.info("回測參數或觀察清單已變更；請重新執行，避免沿用舊結果。")
            st.warning("這是歷史研究，不是投資建議。仍未完整納入股息再投資、除權息、漲跌停排隊、停牌、流動性衝擊、下市股及歷史成分股偏誤；樣本外結果不能回頭用於調參。")


    st.divider()
    st.markdown("**滾動式樣本外測試（Rolling Walk-Forward）**")
    st.caption("將歷史資料切成連續且不重疊的測試區段；每段只用測試開始前的資料暖機指標，測試期間才允許交易。每段獨立重設本金與持倉，鏈結績效只用來觀察多個 OOS 區段的報酬序列，不代表同一帳戶連續持倉。此功能不會自動挑選最佳參數。")
    wf_strategy=st.selectbox("滾動測試策略",STRATEGIES,key="rolling_wf_strategy")
    wf_lookback=st.slider("每段回看資料（日）",260,1000,756,20,key="rolling_wf_lookback")
    wf_test=st.slider("每段樣本外長度（交易日）",40,252,126,10,key="rolling_wf_test")
    wf_exit=st.selectbox("滾動測試出場規則",[
        ("baseline_atr2","原始 ATR × 2"),
        ("oneil_8pct","O’Neil 8% 參考停損"),
        ("minervini_structure","Minervini 結構低點代理"),
        ("atr_1_5","ATR × 1.5"),
        ("atr_2_5","ATR × 2.5"),
        ("atr_3","ATR × 3"),
    ],format_func=lambda x:x[1],key="rolling_wf_exit")
    wf_config=(wf_strategy,int(wf_lookback),int(wf_test),wf_exit[0],
               float(st.session_state.get("wf_capital",capital)),
               int(st.session_state.get("wf_max_positions",5)),
               float(st.session_state.get("wf_risk_percent",0.5))/100,
               tuple(sorted(data.keys())),period)
    if st.button("執行滾動式樣本外測試",key="run_rolling_wf"):
        with st.spinner("逐段執行獨立樣本外回測…"):
            result=rolling_walk_forward(
                data,wf_strategy,
                initial_capital=wf_config[4],max_positions=wf_config[5],
                risk_fraction=wf_config[6],lookback_days=wf_lookback,
                test_days=wf_test,step_days=wf_test,warmup_bars=200,
                exit_profile=wf_exit[0],
            )
        st.session_state["rolling_wf_result"]=result
        st.session_state["rolling_wf_config"]=wf_config
    if st.session_state.get("rolling_wf_config")==wf_config:
        result=st.session_state["rolling_wf_result"]
        if result["windows"].empty:
            st.warning("目前資料長度不足以形成至少一段有效樣本外區間；請增加歷史期間或縮短回看／測試區間。")
        else:
            st.subheader("各段樣本外績效")
            st.dataframe(result["windows"].style.format({
                "期末資產":"{:,.0f}","總報酬":"{:.1%}","年化報酬":"{:.1%}",
                "最大回撤":"{:.1%}","Sharpe":"{:.2f}","Profit Factor":lambda x:"∞" if np.isinf(x) else f"{x:.2f}",
                "勝率":"{:.1%}","平均曝險":"{:.1%}",
            }),use_container_width=True,hide_index=True)
            st.subheader("跨區段鏈結績效")
            st.json(result["metrics"])
            if not result["equity"].empty:
                benchmark_default = "0050" if "0050" in data else sorted(data.keys())[0]
                benchmark_code = st.selectbox(
                    "盤勢分類基準（使用所選股票作市場代理）",
                    sorted(data.keys()), index=sorted(data.keys()).index(benchmark_default),
                    format_func=lambda x: x + " " + W.get(x, x), key="wf_regime_benchmark"
                )
                regime_table = regime_performance(
                    result["equity"], data[benchmark_code],
                    initial_capital=wf_config[4], ema_window=200, slope_window=20
                )
                st.subheader("不同盤勢下的樣本外表現")
                if regime_table.empty:
                    st.info("基準資料不足以計算 200 日趨勢盤勢分類，請增加歷史資料。")
                else:
                    st.dataframe(regime_table.style.format({
                        "累積報酬":"{:.1%}", "年化波動率":"{:.1%}",
                        "Sharpe":"{:.2f}", "上漲日比例":"{:.1%}",
                    }), use_container_width=True, hide_index=True)
                st.caption("盤勢以基準股收盤價相對 200 日 EMA、以及 EMA 相對 20 日前的方向分類；這是盤勢代理，不等於加權指數。各區段首日報酬以該區段重設本金計算，避免跨區段空窗產生虛假日報酬。")
            if not result["equity"].empty:
                wf_fig=go.Figure()
                wf_fig.add_trace(go.Scatter(x=result["equity"]["日期"],y=result["equity"]["資產"],name="鏈結 OOS 資產"))
                wf_fig.add_hline(y=wf_config[4],line_dash="dash")
                wf_fig.update_layout(title="不重疊樣本外區段鏈結曲線",yaxis_title="新台幣")
                st.plotly_chart(wf_fig,use_container_width=True)
            st.subheader("滾動樣本外交易紀錄")
            st.dataframe(result["trades"],use_container_width=True,hide_index=True)
            if not result["trades"].empty:
                st.download_button("下載滾動樣本外交易 CSV",
                    result["trades"].to_csv(index=False).encode("utf-8-sig"),
                    "rolling_walk_forward_trades.csv","text/csv")
            st.caption(result["note"]+" 每段均採相同成本假設；樣本外區段仍可能受現行股票清單及歷史下市股缺漏影響。")
    elif "rolling_wf_config" in st.session_state:
        st.info("滾動測試設定或股票清單已變更；請重新執行。")

    st.divider()
    st.markdown("**參數敏感度／穩健性矩陣**")
    st.caption("預先指定 3 種單筆風險預算 × 3 種出場規則，共 9 組滾動樣本外測試；這是穩健性檢查，不會自動挑選最佳組合，也不應依最終 OOS 結果反覆調參。")
    if st.button("執行 9 組參數穩健性測試", key="run_wf_sensitivity"):
        with st.spinner("執行 9 組滾動樣本外穩健性測試…"):
            sensitivity = parameter_sensitivity(
                data, wf_strategy,
                risk_fractions=(0.0025, 0.005, 0.01),
                exit_profiles=("baseline_atr2", "atr_1_5", "atr_3"),
                initial_capital=wf_config[4], max_positions=wf_config[5],
                lookback_days=wf_lookback, test_days=wf_test,
                step_days=wf_test, warmup_bars=200,
            )
        st.session_state["wf_sensitivity"] = sensitivity
        st.session_state["wf_sensitivity_config"] = wf_config
    if st.session_state.get("wf_sensitivity_config") == wf_config:
        sensitivity = st.session_state["wf_sensitivity"]
        st.dataframe(sensitivity.style.format({
            "單筆風險預算":"{:.2%}", "鏈結總報酬":"{:.1%}",
            "鏈結最大回撤":"{:.1%}", "鏈結 Sharpe":"{:.2f}",
        }), use_container_width=True, hide_index=True)
        st.download_button(
            "下載參數穩健性矩陣 CSV",
            sensitivity.to_csv(index=False).encode("utf-8-sig"),
            "walk_forward_sensitivity.csv", "text/csv"
        )


with t7:
    st.subheader("自動交易引擎｜模擬模式")
    st.warning("目前為 staged 模式：使用 NT$100,000 起始模擬資金；券商尚未指定，實盤 API 未接通且被安全鎖定，不會送出真實委託。")
    st.caption("券商介接採 BrokerGateway 可替換架構；必須完成券商選定、API Adapter、帳戶核對、模擬驗證及人工啟用後，才可評估開放實盤。")
    st.markdown("**自動策略條件**：買進採用「多因子觀察」訊號，要求 AI 測試優於多數類基準至少 2 個百分點、五日預測分數至少 55%；賣出則檢查 EMA21、RSI 與 MACD 退出條件。買賣都必須取得盤中更新的參考報價。")
    staged_config = TradingConfig(initial_capital=float(capital))
    limits = staged_config.risk_limits()
    staged_controller = StagedTradingController()
    engine = PaperExecutionEngine(
        limits=limits,
        price_limit_provider=lambda symbol, market: get_official_price_limit(
            symbol, market, expected_date=session_now.date()
        ),
        calendar_provider=market_trading_day_status,
    )
    paper_account = PaperAccount(engine.db_path, initial_cash=staged_config.initial_capital)
    adaptive_monitor = AdaptiveExitMonitor(
        engine.db_path, max_age_seconds=staged_config.max_quote_age_seconds
    )
    st.divider()
    st.subheader("瀏覽器雲端模擬 Worker｜免安裝模式")
    st.caption(
        "此模式直接在目前的 Streamlit 網頁工作階段執行單輪模擬 Worker。"
        "只有網頁保持開啟且工作階段仍存活時，才能依設定自動刷新；"
        "關閉頁面、服務休眠、重啟或網路中斷時不會持續監控。"
    )
    browser_symbols = st.multiselect(
        "雲端監控標的（最多 5 檔，先從少量標的驗證）",
        options=[code for code in ss if code in market_by_symbol],
        default=[code for code in ss if code in market_by_symbol][:5],
        format_func=lambda code: f"{code} {W.get(code, code)}",
        key="browser_worker_symbols",
    )
    browser_auto_entries = st.checkbox(
        "允許自動建立買進模擬委託（預設關閉）",
        value=False,
        key="browser_worker_auto_entries",
        help="只會建立模擬委託，不會連接券商或動用真實資金。",
    )
    browser_auto_refresh = st.checkbox(
        "網頁開啟期間每 30 秒自動執行一輪",
        value=False,
        key="browser_worker_auto_refresh",
        help="僅在此網頁工作階段存活時執行，不是 24 小時常駐服務。",
    )
    if len(browser_symbols) > 5:
        st.error("瀏覽器免安裝模式目前最多選擇 5 檔，避免免費雲端環境超時或觸及行情來源限制。")
    elif not browser_symbols:
        st.warning("請至少選擇一檔監控標的。")
    else:
        market_db_path = os.getenv(
            "MARKET_DB_PATH",
            str(Path(tempfile.gettempdir()) / "tw_stock_market.sqlite3"),
        )

        @st.fragment(run_every="30s" if browser_auto_refresh else None)
        def browser_worker_panel():
            should_run = browser_auto_refresh or st.button(
                "立即執行一輪模擬監控", key="browser_worker_run_once"
            )
            if should_run:
                try:
                    browser_worker = PaperWorker(
                        db_path=engine.db_path,
                        market_db_path=market_db_path,
                        watchlist={
                            code: market_by_symbol.get(str(code), "上市")
                            for code in browser_symbols
                        },
                        config=TradingConfig(initial_capital=float(capital)),
                        auto_entries=browser_auto_entries,
                    )
                    with st.spinner("正在取得新鮮報價、檢查交易日曆及執行模擬風控…"):
                        cycle_summary = browser_worker.run_once()
                    st.write("最近一輪執行結果")
                    st.json(cycle_summary)
                except Exception as exc:
                    st.error(f"瀏覽器模擬 Worker 執行失敗：{exc}")
            else:
                st.info("自動刷新目前關閉。可按「立即執行一輪模擬監控」手動測試。")
            latest_status = read_worker_status(engine.db_path)
            if latest_status:
                st.caption(
                    "最近心跳：" + str(latest_status["heartbeat_at"])
                    + "｜狀態：" + str(latest_status["status"])
                )
                st.json(latest_status.get("summary", {}))
            if browser_auto_refresh:
                st.caption("自動刷新已啟用；每輪仍會重新檢查交易日曆、報價時間及風控條件。")
        browser_worker_panel()
    st.warning(
        "免費雲端限制：Streamlit Community Cloud 的執行環境可能休眠或重啟，"
        "本機 SQLite 也可能在重新部署／重啟後遺失。請定期下載帳本備份；"
        "此模式不能保證 24/7、無限執行或永久保存資料。"
    )
    worker_status = read_worker_status(engine.db_path)
    if worker_status:
        heartbeat = datetime.fromisoformat(worker_status["heartbeat_at"])
        heartbeat_age = max(0, (datetime.now(TZ) - heartbeat.astimezone(TZ)).total_seconds())
        worker_label = "心跳正常" if heartbeat_age <= 120 else "心跳逾時"
        st.write(
            f"常駐 Worker：{worker_label}｜程序狀態：{worker_status['status']}｜最近心跳：{worker_status['heartbeat_at']} "
            f"（{heartbeat_age:.0f} 秒前）"
        )
        if worker_status.get("summary"):
            st.caption("Worker 最近一輪：" + json.dumps(worker_status["summary"], ensure_ascii=False))
    else:
        st.info("尚未偵測到常駐 Paper Worker。需在持久化主機啟動 Docker Compose 的 worker 服務。")
    account_snapshot = paper_account.snapshot()
    account_positions = {p["symbol"]: int(p["quantity"]) for p in account_snapshot["positions"]}
    account_costs = {p["symbol"]: float(p["average_cost"]) for p in account_snapshot["positions"]}
    adaptive_monitor.clear_absent_positions(account_positions.keys())
    session_now = datetime.now(TZ)
    expired_orders = engine.expire_day_orders(session_now)
    market_day_open, market_day_note = trading_day_status(session_now.date())
    session_open = taiwan_regular_session(session_now) and market_day_open
    st.write("交易時段：", "台股一般盤中" if session_open else "非台股一般盤中時段")
    if not market_day_open:
        st.warning(market_day_note)
    else:
        st.caption(market_day_note + "；一般盤中時間仍依 09:00–13:25 判斷。官方年度日曆每 6 小時快取；未能驗證時會停止建立委託。")
    l1,l2,l3,l4=st.columns(4)
    l1.metric("單筆委託上限",f"NTD {limits.max_order_notional:,.0f}")
    l2.metric("單一標的曝險上限",f"NTD {limits.max_symbol_exposure:,.0f}")
    l3.metric("投資組合曝險上限",f"NTD {limits.max_portfolio_exposure:,.0f}")
    l4.metric("每日虧損停手",f"NTD {limits.max_daily_loss:,.0f}")
    st.caption("整股委託限 09:00–13:25；盤中零股委託限 09:00–13:30、1–999 股，均採限價當日有效。模擬成交不重建交易所每 5 秒集合競價及真實排隊順位。")
    st.subheader("手動模擬委託（整股／盤中零股）")
    manual_symbols = sorted([str(x) for x in stock_master["代號"].tolist()])
    manual_symbol = st.selectbox("股票代號", manual_symbols, key="manual_paper_symbol")
    manual_side_label = st.radio("買賣方向", ["買進", "賣出"], horizontal=True, key="manual_paper_side")
    manual_mode = st.radio("委託單位", ["整股（1,000 股倍數）", "盤中零股（1–999 股）"], horizontal=True, key="manual_paper_mode")
    manual_is_odd = manual_mode.startswith("盤中")
    manual_qty = st.number_input("委託股數", min_value=1, max_value=999 if manual_is_odd else 1000000, value=100 if manual_is_odd else 1000, step=1 if manual_is_odd else 1000, key="manual_paper_qty")
    manual_price = st.number_input("限價（元）", min_value=0.01, value=100.0, step=0.05, format="%.2f", key="manual_paper_price")
    if st.button("送出手動模擬委託", key="submit_manual_paper", disabled=not session_open):
        snap = paper_account.snapshot()
        positions_now = {p["symbol"]: int(p["quantity"]) for p in snap["positions"]}
        value_now = float(snap["cash"]) + sum(int(p["quantity"]) * float(p["average_cost"]) for p in snap["positions"])
        orders_now = engine.list_orders()
        pending_now = [x for x in orders_now if x["status"] == "ACCEPTED"]
        reserved_buy_cash = sum(float(x["quantity"]) * float(x["limit_price"]) * 1.001425
                                for x in pending_now if x["side"] == "BUY")
        for pending in pending_now:
            if pending["side"] == "SELL":
                positions_now[pending["symbol"]] = max(0, positions_now.get(pending["symbol"], 0) - int(pending["quantity"]))
        cash_for_new_order = max(0.0, float(snap["cash"]) - reserved_buy_cash)
        today_now = session_now.strftime("%Y-%m-%d")
        fills_now = [f for f in snap["fills"] if str(f["filled_at"]).startswith(today_now)]
        pnl_now = sum(float(f["realized_pnl"]) for f in fills_now)
        manual_intent = OrderIntent(
            symbol=str(manual_symbol), side=Side.BUY if manual_side_label == "買進" else Side.SELL,
            quantity=int(manual_qty), limit_price=float(manual_price),
            reason="使用者手動建立模擬限價委託",
            idempotency_key=f"manual-{manual_symbol}-{manual_side_label}-{manual_qty}-{manual_price}-{session_now.isoformat()}",
            instrument_type="etf" if str(manual_symbol) in etf_codes else "stock",
            odd_lot=manual_is_odd,
            market=market_by_symbol.get(str(manual_symbol), "上市"),
        )
        manual_result = engine.submit(
            manual_intent, cash=cash_for_new_order, positions=positions_now, equity=value_now,
            daily_realized_pnl=pnl_now,
            open_order_count=sum(x["status"] == "ACCEPTED" for x in orders_now),
            daily_order_count=sum(str(x["created_at"]).startswith(today_now) for x in orders_now),
            now=session_now,
        )
        if manual_result.status == "ACCEPTED":
            st.success(f"模擬委託已接受：{manual_result.order_id}")
        else:
            st.error(f"委託遭拒：{manual_result.message}")
    auto_candidates = []
    quote_lookup = {q["代號"]: q for q in quotes}
    row_lookup = {r["代號"]: r for r in rows}
    existing_orders_for_candidates = engine.list_orders()
    pending_for_candidates = [x for x in existing_orders_for_candidates if x["status"] == "ACCEPTED"]
    reserved_buy_cash_for_candidates = sum(float(x["quantity"]) * float(x["limit_price"]) * 1.001425
                                           for x in pending_for_candidates if x["side"] == "BUY")
    pending_sell_by_symbol = {}
    for pending in pending_for_candidates:
        if pending["side"] == "SELL":
            pending_sell_by_symbol[pending["symbol"]] = pending_sell_by_symbol.get(pending["symbol"], 0) + int(pending["quantity"])
    paper_positions = {code: max(0, qty - pending_sell_by_symbol.get(code, 0))
                       for code, qty in account_positions.items()}
    cash_after_reservations = max(0.0, float(account_snapshot["cash"]) - reserved_buy_cash_for_candidates)
    for code, row in row_lookup.items():
        quote = quote_lookup.get(code, {})
        price = quote.get("盤中參考價")
        if (quote.get("報價狀態") != "盤中更新"
                or price is None or not np.isfinite(price) or price <= 0):
            continue
        frame = data[code]
        latest = frame.iloc[-1]
        held = int(paper_positions.get(code, 0))
        exit_signal = (
            float(latest.close) < float(latest.ema21)
            or (pd.notna(latest.rsi) and float(latest.rsi) > 75)
            or (pd.notna(latest.macd) and pd.notna(latest.sig) and float(latest.macd) < float(latest.sig))
        )
        if held > 0 and exit_signal:
            sell_qty_raw = min(held, int(limits.max_order_notional / float(price)))
            sell_qty = floor_board_lot(sell_qty_raw)
            sell_is_odd = sell_qty == 0 and 0 < sell_qty_raw < 1000
            if sell_is_odd:
                sell_qty = sell_qty_raw
            if sell_qty > 0:
                auto_candidates.append({
                    "方向":"賣出","代號":code,"名稱":row["名稱"],
                    "參考價":round(float(price),2),"股數":sell_qty,
                    "估算金額":round(sell_qty*float(price),2),
                    "策略依據":"跌破 EMA21／RSI 過熱／MACD 轉弱",
                    "報價時間":quote.get("報價時間"),
                    "零股":sell_is_odd,
                })
        elif row.get("訊號") == "多因子觀察":
            atr_value = float(latest.atr) if pd.notna(latest.atr) else float(price)*0.04
            stop_distance = max(2.0*atr_value, float(price)*0.01)
            risk_qty = int(float(risk) / stop_distance)
            notional_qty = int(limits.max_order_notional / float(price))
            cash_available = cash_after_reservations
            cash_qty = int(cash_available / (float(price)*1.001425))
            quantity = floor_board_lot(min(risk_qty, notional_qty, cash_qty))
            if quantity > 0:
                auto_candidates.append({
                    "方向":"買進","代號":code,"名稱":row["名稱"],
                    "參考價":round(float(price),2),"股數":quantity,
                    "估算金額":round(quantity*float(price),2),
                    "策略依據":"AI 多因子觀察＋ATR 風險預算控量",
                    "AI五日分數":row.get("AI五日上漲機率"),
                    "報價時間":quote.get("報價時間"),
                    "零股":False,
                })
    st.write("符合自動策略條件的候選委託：",len(auto_candidates))
    if auto_candidates:
        st.dataframe(pd.DataFrame(auto_candidates),use_container_width=True,hide_index=True)
    else:
        st.info("目前沒有同時符合 AI 條件與新鮮盤中報價的標的；不會使用舊收盤價代替盤中價格建立自動委託。")
    enable_paper_auto = st.checkbox("啟用本輪自動模擬委託（不是真實下單）",value=False)
    today_for_risk = session_now.strftime("%Y-%m-%d")
    today_fills = [f for f in account_snapshot["fills"] if str(f["filled_at"]).startswith(today_for_risk)]
    daily_pnl = sum(float(f["realized_pnl"]) for f in today_fills)
    if daily_pnl is None:
        st.error("無法由完整模擬成交紀錄核對每日已實現損益（可能有匯入持倉缺少成本）；自動委託已鎖定。")
    else:
        st.caption(f"今日模擬已實現損益：NTD {daily_pnl:,.0f}")
    if st.button("依策略訊號建立模擬委託",disabled=(not enable_paper_auto or not session_open or not auto_candidates or daily_pnl is None)):
        positions = dict(account_positions)
        cash_available = max(0.0, float(account_snapshot["cash"]) - reserved_buy_cash_for_candidates)
        market_value = sum(
            int(qty)*float(data[code].iloc[-1]["close"])
            for code,qty in positions.items() if code in data
        )
        equity_value = cash_available + market_value
        existing_orders = engine.list_orders()
        pending_count = sum(1 for order in existing_orders if order["status"] == "ACCEPTED")
        today_text = session_now.strftime("%Y-%m-%d")
        daily_count = sum(1 for order in existing_orders if str(order["created_at"]).startswith(today_text))
        created_count = 0
        for candidate in auto_candidates:
            signal_row = row_lookup[candidate["代號"]]
            side = Side.BUY if candidate["方向"] == "買進" else Side.SELL
            intent = OrderIntent(
                symbol=candidate["代號"], side=side,
                quantity=int(candidate["股數"]), limit_price=float(candidate["參考價"]),
                reason=candidate["策略依據"],
                idempotency_key=f"auto-{side.value.lower()}-{candidate['代號']}-{today_text}-{signal_row['日線日期']}",
                instrument_type="etf" if candidate["代號"] in etf_codes else "stock",
                odd_lot=bool(candidate.get("零股", False)),
                market=market_by_symbol.get(str(candidate["代號"]), "上市"),
            )
            result = engine.submit(
                intent, cash=cash_available, positions=positions,
                equity=equity_value, daily_realized_pnl=daily_pnl,
                open_order_count=pending_count, daily_order_count=daily_count,
                now=session_now,
            )
            if result.status == "ACCEPTED":
                created_count += 1
                pending_count += 1
                daily_count += 1
                # Reserve cash and exposure for subsequent candidates in this batch.
                reserved = candidate["股數"] * candidate["參考價"] * 1.001425
                cash_available = max(0.0, cash_available - reserved)
                if side == Side.BUY:
                    positions[candidate["代號"]] = positions.get(candidate["代號"], 0) + candidate["股數"]
                else:
                    positions[candidate["代號"]] = max(0, positions.get(candidate["代號"], 0) - candidate["股數"])
        st.success(f"本輪建立 {created_count} 筆模擬委託。重複訊號會由識別碼阻擋。")
    st.subheader("模擬帳戶資產")
    position_value = sum(
        int(p["quantity"]) * float(quote_lookup.get(p["symbol"], {}).get("盤中參考價") or p["average_cost"])
        for p in account_snapshot["positions"]
    )
    equity_now = float(account_snapshot["cash"]) + position_value
    m1,m2,m3,m4=st.columns(4)
    m1.metric("可用現金",f"NTD {account_snapshot['cash']:,.0f}")
    m2.metric("持股市值估算",f"NTD {position_value:,.0f}")
    m3.metric("帳戶資產估算",f"NTD {equity_now:,.0f}")
    m4.metric("今日已實現損益",f"NTD {daily_pnl:,.0f}")
    if account_snapshot["positions"]:
        st.dataframe(pd.DataFrame(account_snapshot["positions"]),use_container_width=True,hide_index=True)
    else:
        st.info("目前模擬帳戶沒有持股。")
    st.caption("持股市值使用當前可取得報價；若無報價則暫以平均成本估算。帳戶資料存在執行環境 SQLite，雲端重啟可能清除。")
    st.subheader("策略化出場與風險預算（研究版）")
    profile_key = st.selectbox("出場策略", list(PROFILES),
        format_func=lambda key: PROFILES[key]["name"], key="exit_profile")
    st.caption(PROFILES[profile_key]["description"])
    entry_ref = st.number_input("假設進場價", min_value=0.01, value=100.0, step=1.0, key="exit_entry")
    invalidation = st.number_input("技術失效價（Minervini 必填；其他策略選填）",
        min_value=0.0, value=0.0, step=0.5, key="exit_invalidation")
    atr_value = st.number_input("ATR（ATR 策略使用）", min_value=0.0, value=2.0, step=0.1, key="exit_atr")
    atr_multiple = st.selectbox("ATR 倍數（待驗證）", [1.5, 2.0, 2.5, 3.0], index=1)
    risk_pct = st.selectbox("帳戶單筆風險預算候選", [0.005, 0.0075, 0.01], index=0,
        format_func=lambda x: f"{x:.2%}")
    if st.button("計算策略停損與風險股數", key="calc_exit_profile"):
        try:
            kwargs = {"technical_invalidation": invalidation if invalidation > 0 else None}
            if profile_key == "atr_dynamic":
                kwargs.update({"atr": atr_value, "atr_multiple": atr_multiple})
            stop_result = initial_stop(entry_ref, profile_key, **kwargs)
            sizing = risk_position_size(float(capital), risk_pct, entry_ref,
                                        stop_result["stop_price"])
            st.write({"策略停損": stop_result, "風險預算／股數": sizing})
        except ValueError as exc:
            st.error(str(exc))
    st.caption("股數僅按帳戶風險預算與每股停損距離計算；尚未扣除現金、曝險、交易單位、費稅及流動性限制。候選參數未經台股樣本外驗證，不構成交易建議。")
    st.divider()
    st.subheader("世界級操盤策略｜動態停損與獲利管理")
    adaptive_profile = st.selectbox(
        "持倉出場策略",
        ["atr_dynamic", "oneil_growth", "minervini_sepa"],
        format_func=lambda key: {
            "atr_dynamic": "ATR 動態停損（預設）",
            "oneil_growth": "O'Neil 成長股（獲利奔跑）",
            "minervini_sepa": "Minervini SEPA（結構失效＋R 倍數）",
        }[key],
        key="adaptive_exit_profile",
    )
    adaptive_atr_multiple = st.selectbox(
        "ATR 移動停損倍數（研究候選）", [1.5, 2.0, 2.5, 3.0],
        index=1, key="adaptive_exit_atr_multiple"
    )
    st.caption(
        "不再把固定 5% 停損／10% 停利當成預設操盤規則。"
        "ATR 策略以波動度移動保護停損；O'Neil 20% 僅進入獲利檢視，不強制賣出；"
        "Minervini 依結構失效價，達 2R 移至成本保護、達 3R 啟動 ATR 追蹤。"
        "所有 ATR／R 倍數均為待樣本外驗證的候選參數。"
    )
    if st.button("依最新已完成日線同步持倉停損", key="sync_adaptive_stops"):
        configured, skipped = [], []
        for position in account_snapshot["positions"]:
            symbol = position["symbol"]
            frame = data.get(symbol)
            if frame is None or len(frame) < 16:
                skipped.append(f"{symbol}：歷史日線不足")
                continue
            completed = frame.copy().sort_values("date").reset_index(drop=True)
            last_date = pd.to_datetime(completed["date"].iloc[-1]).date()
            # Never use today's unfinished daily bar to calculate ATR or structure.
            if last_date == session_now.date():
                completed = completed.iloc[:-1].copy()
            if len(completed) < 15:
                skipped.append(f"{symbol}：缺少足夠的已完成日線")
                continue
            previous_close = completed["close"].shift(1)
            true_range = pd.concat([
                completed["high"] - completed["low"],
                (completed["high"] - previous_close).abs(),
                (completed["low"] - previous_close).abs(),
            ], axis=1).max(axis=1)
            atr_series = true_range.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
            atr_value = float(atr_series.iloc[-1])
            if not np.isfinite(atr_value) or atr_value <= 0:
                skipped.append(f"{symbol}：ATR 無效")
                continue
            technical_invalidation = None
            if adaptive_profile == "minervini_sepa":
                technical_invalidation = float(completed["low"].tail(10).min())
                if technical_invalidation >= float(position["average_cost"]):
                    skipped.append(f"{symbol}：找不到低於成本的結構失效價")
                    continue
            try:
                state = adaptive_monitor.configure_position(
                    symbol, int(position["quantity"]), float(position["average_cost"]),
                    atr_value, position_version=int(position["position_version"]),
                    profile=adaptive_profile, atr_multiple=float(adaptive_atr_multiple),
                    technical_invalidation=technical_invalidation, now=session_now,
                )
                configured.append({
                    "代號": symbol, "策略": state["profile"],
                    "初始／目前保護價": round(state["stop_price"], 2),
                    "ATR": round(atr_value, 2),
                })
            except ValueError as exc:
                skipped.append(f"{symbol}：{exc}")
        if configured:
            st.dataframe(pd.DataFrame(configured), use_container_width=True, hide_index=True)
        if skipped:
            st.warning("未同步項目：" + "；".join(skipped))
        if not configured and not skipped:
            st.info("目前沒有持倉需要設定動態停損。")
    if st.button(
        "掃描新鮮報價並建立動態退出模擬委託",
        key="scan_adaptive_exit_orders",
        disabled=not session_open,
    ):
        stop_created = []
        current_orders = engine.list_orders()
        pending_sell_symbols = {
            o["symbol"] for o in current_orders
            if o["status"] == "ACCEPTED" and o["side"] == "SELL"
        }
        pending_count = sum(1 for o in current_orders if o["status"] == "ACCEPTED")
        today_count = sum(
            1 for o in current_orders
            if str(o["created_at"]).startswith(today_for_risk)
        )
        current_equity = float(account_snapshot["cash"]) + sum(
            int(p["quantity"]) * float(
                quote_lookup.get(p["symbol"], {}).get("盤中參考價") or p["average_cost"]
            ) for p in account_snapshot["positions"]
        )
        for quote in quotes:
            if quote.get("報價狀態") != "盤中更新" or quote.get("盤中參考價") is None:
                continue
            try:
                signal = adaptive_monitor.evaluate_quote(
                    quote["代號"], float(quote["盤中參考價"]),
                    str(quote.get("報價時間") or ""),
                    "Yahoo Finance 1-minute", now=session_now,
                )
            except ValueError:
                continue
            if not signal or signal["symbol"] in pending_sell_symbols:
                continue
            held = int(account_positions.get(signal["symbol"], 0))
            raw_qty = min(
                held, int(signal["quantity"]),
                int(limits.max_order_notional / signal["price"]),
            )
            if raw_qty >= 1000:
                qty, odd_lot = floor_board_lot(raw_qty), False
            else:
                qty, odd_lot = raw_qty, True
            if qty <= 0:
                continue
            intent = OrderIntent(
                symbol=signal["symbol"], side=Side.SELL, quantity=qty,
                limit_price=float(signal["price"]), reason=signal["reason"],
                idempotency_key=(
                    f"adaptive-stop-{signal['symbol']}-{today_for_risk}-"
                    f"{signal['quote_time']}-{signal['trigger_price']}"
                ),
                instrument_type="etf" if signal["symbol"] in etf_codes else "stock",
                odd_lot=odd_lot,
                market=market_by_symbol.get(str(signal["symbol"]), "上市"),
            )
            result = engine.submit(
                intent, cash=float(account_snapshot["cash"]),
                positions=account_positions, equity=current_equity,
                daily_realized_pnl=daily_pnl, open_order_count=pending_count,
                daily_order_count=today_count, now=session_now,
            )
            if result.status == "ACCEPTED":
                pending_sell_symbols.add(signal["symbol"])
                pending_count += 1
                today_count += 1
                stop_created.append({
                    "代號": signal["symbol"], "觸發原因": signal["reason"],
                    "保護停損價": signal["trigger_price"],
                    "最新參考價": signal["price"], "委託股數": qty,
                    "零股": odd_lot, "委託狀態": result.status,
                })
        if stop_created:
            st.success(f"已建立 {len(stop_created)} 筆動態退出模擬委託。")
            st.dataframe(pd.DataFrame(stop_created), use_container_width=True, hide_index=True)
        else:
            st.info("本輪沒有觸發動態停損，或沒有可建立的合規退出委託。")
        st.rerun()
    adaptive_events = adaptive_monitor.events()
    if adaptive_events:
        st.dataframe(pd.DataFrame(adaptive_events), use_container_width=True, hide_index=True)
    st.subheader("用盤中報價自動撮合")
    st.caption("只處理 Yahoo Finance 1 分鐘資料觸及限價的委託；以委託限價作保守模擬成交，不代表真實排隊成交或可成交深度。支援整股及 1–999 股盤中零股委託，並依台股升降單位檢查。")
    if st.button("使用目前新鮮盤中報價撮合等待委託", disabled=not session_open):
        fill_results=[]
        for q in quotes:
            if q.get("報價狀態") != "盤中更新" or q.get("盤中參考價") is None:
                continue
            try:
                fill_results.extend(paper_account.process_quote(
                    q["代號"], float(q["盤中參考價"]), str(q.get("報價時間") or ""),
                    "Yahoo Finance 1-minute", now=session_now,
                    market=market_by_symbol.get(str(q["代號"]), "上市")
                ))
            except ValueError:
                continue
        if fill_results:
            st.success(f"本次完成 {len(fill_results)} 筆模擬成交，已同步現金與持股。")
            st.dataframe(pd.DataFrame(fill_results),use_container_width=True,hide_index=True)
        else:
            st.info("本次沒有符合限價、報價新鮮度及交易時段條件的成交。")
        st.rerun()
    if expired_orders:
        st.info(f"已將 {expired_orders} 筆前一交易日未成交委託標記為到期。")
    st.subheader("模擬委託佇列與成交狀態")
    order_rows = engine.list_orders()
    if order_rows:
        order_display = pd.DataFrame(order_rows)
        order_display["委託類型"] = order_display["odd_lot"].map({True: "盤中零股", False: "整股"})
        order_display["方向"] = order_display["side"].map({"BUY": "買進", "SELL": "賣出"})
        order_display["狀態"] = order_display["status"].map({
            "ACCEPTED": "等待中", "FILLED": "已模擬成交", "CANCELLED": "已取消",
            "REJECTED": "遭拒", "EXPIRED": "已到期"
        }).fillna(order_display["status"])
        st.dataframe(order_display, use_container_width=True, hide_index=True)
    else:
        st.info("目前沒有模擬委託紀錄。")
    pending_orders = [order for order in order_rows if order["status"] == "ACCEPTED"]
    if pending_orders:
        pending_ids = [order["order_id"] for order in pending_orders]
        cancel_id = st.selectbox(
            "選擇要取消的等待委託", pending_ids,
            format_func=lambda oid: next(
                f"{item['symbol']} {'零股' if item.get('odd_lot') else '整股'} {'買進' if item['side'] == 'BUY' else '賣出'} {item['quantity']} 股 @ {item['limit_price']}"
                for item in pending_orders if item["order_id"] == oid
            ),
        )
        if st.button("取消選取的模擬委託"):
            cancelled = engine.cancel_order(cancel_id)
            st.success(f"已取消 {cancelled.order_id}")
    current_snapshot = paper_account.snapshot()
    if current_snapshot["fills"]:
        st.subheader("模擬成交明細")
        st.dataframe(pd.DataFrame(current_snapshot["fills"]),use_container_width=True,hide_index=True)
        st.download_button(
            "下載模擬成交紀錄 CSV",
            pd.DataFrame(current_snapshot["fills"]).to_csv(index=False).encode("utf-8-sig"),
            "paper_fills.csv","text/csv"
        )
    if order_rows:
        st.download_button(
            "下載模擬委託紀錄 CSV",
            pd.DataFrame(order_rows).to_csv(index=False).encode("utf-8-sig"),
            "paper_order_log.csv","text/csv"
        )
    st.caption("已新增獨立常駐 Paper Worker：可按設定觀察清單掃描三種技術策略、處理新鮮盤中報價、模擬撮合及動態退出。需在持久化主機以 Docker Compose 啟動，並與網頁共用 SQLite 資料卷；預設禁止自動買進，需明確設定 PAPER_AUTO_ENTRIES=true。此 Worker 仍只做模擬交易，沒有券商 API 或真實委託能力。")
