import json
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
from research_engine import STRATEGIES, daily_strategy_returns, performance, fundamental_snapshot, shared_capital_backtest
from datetime import datetime
from zoneinfo import ZoneInfo
from market_data import quote_freshness
from analytics import equity_risk_metrics, trade_statistics
from historical_store import get_incremental_history, price_cache_summary
from execution_engine import PaperExecutionEngine, OrderIntent, RiskLimits, Side, taiwan_regular_session, realized_pnl_from_trades
from paper_account import PaperAccount
from stop_monitor import StopMonitor
from strategy_exits import PROFILES, initial_stop, risk_position_size
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
    capital=st.number_input("模擬本金",100000,10000000,1000000,100000)
    risk=st.number_input("單筆風險上限",500,50000,5000,500)
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
t0,t1,t2,t3,t4,t5,t6,t7=st.tabs(["上市櫃股票庫","行情監控","歷史回測","模擬帳本","資料限制","大師策略回測","整合研究中心","自動交易（模擬）"])
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
    s=st.selectbox("回測標的",list(data),key="bt"); x=data[s].copy().reset_index(drop=True)
    x["entry"]=(x.close>x.ema21)&x.up&x.rsi.between(50,70)&(x.macd>x.sig)&(x.volume>1.5*x.v20)&(x.close>x.h20)
    x["exit"]=(x.close<x.ema21)|(x.rsi>75)|(x.macd<x.sig)
    cash=float(capital); qty=0; stop=0; trades=[]
    curve=[{"日期":x.iloc[0].date,"資產":float(capital)}]
    for i in range(len(x)-1):
        r,n=x.iloc[i],x.iloc[i+1]; op=float(n.open)
        if qty and (r.exit or n.low<=stop):
            px=min(op,stop) if n.low<=stop else op; gross=qty*px*.999
            cash+=gross*(1-.001425-.003); trades[-1]["損益"]=cash-trades[-1]["投入前現金"]; trades[-1]["出場日"]=str(n.date.date()); qty=0
        if qty==0 and r.entry:
            px=op*1.001; stop=px-2*(r.atr if pd.notna(r.atr) else px*.04)
            q=max(0,min(int(risk/max(px-stop,.01)),int(cash/(px*1.001425))))
            if q: trades.append({"進場日":str(n.date.date()),"股數":q,"進場價":px,"投入前現金":cash}); cash-=q*px*1.001425; qty=q
        curve.append({"日期":n.date,"資產":cash+qty*float(n.close)})
    if qty:
        px=float(x.iloc[-1].close)*.999; cash+=qty*px*(1-.001425-.003); trades[-1]["損益"]=cash-trades[-1]["投入前現金"]; trades[-1]["備註"]="期末平倉"
        curve[-1]["資產"]=cash
    simple_equity=pd.DataFrame(curve)
    simple_risk=equity_risk_metrics(simple_equity,float(capital))
    simple_trade=trade_statistics(pd.DataFrame(trades))
    a,b,c,dcol,ecol=st.columns(5)
    a.metric("期末資產",f"NTD {cash:,.0f}")
    b.metric("交易筆數",len(trades))
    c.metric("勝率",f"{simple_trade['勝率']:.1%}" if trades else "無交易")
    dcol.metric("最大回撤",f"{simple_risk['最大回撤']:.1%}")
    ecol.metric("Profit Factor","∞" if np.isinf(simple_trade["Profit Factor"]) else f"{simple_trade['Profit Factor']:.2f}")
    st.warning("簡化回測：收盤產生訊號、次日開盤成交，含估算手續費、證交稅及滑價；未完整處理除權息、停牌、漲跌停、流動性與下市股偏誤。")
    if curve:
        f=go.Figure(go.Scatter(x=[z["日期"] for z in curve],y=[z["資產"] for z in curve])); f.add_hline(y=capital,line_dash="dash"); st.plotly_chart(f,use_container_width=True)
    st.dataframe(pd.DataFrame(trades).drop(columns=["投入前現金"],errors="ignore"),use_container_width=True)
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
    st.markdown("**基本面快照（最新可取得資料）**")
    st.caption("以下為目前可取得的最新快照，不是歷史逐期財報；不會用來回填歷史交易訊號。")
    if st.button("載入基本面資料",key="fund_load"):
        with st.spinner("讀取財務資料…"):
            st.session_state["fund_data"]=fundamental_snapshot(tuple(data.keys()))
    if "fund_data" in st.session_state:
        fund=st.session_state["fund_data"].copy()
        st.dataframe(fund,use_container_width=True,hide_index=True)
        st.markdown("**巴菲特／彼得林區風格：最新資料簡化篩選**")
        st.caption("這是當前財務快照的條件檢查，不是完整投資大師模型，也不能用於歷史回測；缺值不會視為通過。")
        def style_check(row, rules):
            known=0; passed=0
            for field, predicate in rules:
                value=row.get(field)
                if pd.notna(value):
                    known+=1
                    passed+=int(bool(predicate(float(value))))
            return f"{passed}/{known} 項符合" if known else "資料不足"
        buffett_rules=[
            ("ROE",lambda x:x>=0.15),
            ("自由現金流",lambda x:x>0),
            ("負債／權益",lambda x:x<=100),
            ("本益比",lambda x:0<x<=25),
        ]
        lynch_rules=[
            ("營收成長率",lambda x:x>=0.10),
            ("獲利成長率",lambda x:x>=0.10),
            ("本益比",lambda x:0<x<=25),
        ]
        fund["巴菲特風格條件"] = fund.apply(lambda r:style_check(r,buffett_rules),axis=1)
        fund["林區風格條件"] = fund.apply(lambda r:style_check(r,lynch_rules),axis=1)
        st.dataframe(fund[["代號","巴菲特風格條件","林區風格條件","資料狀態"]],use_container_width=True,hide_index=True)

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


with t7:
    st.subheader("自動交易引擎｜模擬模式")
    st.warning("目前只建立與撮合模擬委託，不會連接券商或送出真實委託。實盤交易開關尚未實作，預設不存在。")
    st.markdown("**自動策略條件**：買進採用「多因子觀察」訊號，要求 AI 測試優於多數類基準至少 2 個百分點、五日預測分數至少 55%；賣出則檢查 EMA21、RSI 與 MACD 退出條件。買賣都必須取得盤中更新的參考報價。")
    limits = RiskLimits()
    engine = PaperExecutionEngine(limits=limits)
    paper_account = PaperAccount(engine.db_path, initial_cash=float(capital))
    stop_monitor = StopMonitor(engine.db_path, stop_loss_pct=0.05, take_profit_pct=0.10, trailing_stop_pct=0.05)
    account_snapshot = paper_account.snapshot()
    account_positions = {p["symbol"]: int(p["quantity"]) for p in account_snapshot["positions"]}
    account_costs = {p["symbol"]: float(p["average_cost"]) for p in account_snapshot["positions"]}
    session_now = datetime.now(TZ)
    session_open = taiwan_regular_session(session_now)
    st.write("交易時段：", "台股一般盤中" if session_open else "非台股一般盤中時段")
    l1,l2,l3,l4=st.columns(4)
    l1.metric("單筆委託上限",f"NTD {limits.max_order_notional:,.0f}")
    l2.metric("單一標的曝險上限",f"NTD {limits.max_symbol_exposure:,.0f}")
    l3.metric("投資組合曝險上限",f"NTD {limits.max_portfolio_exposure:,.0f}")
    l4.metric("每日虧損停手",f"NTD {limits.max_daily_loss:,.0f}")
    st.caption("其他保護：不允許放空、必須在 09:00–13:25 建立委託、最多 5 筆未完成委託、每日最多 20 次委託、防重複委託識別碼、限價單。")
    auto_candidates = []
    quote_lookup = {q["代號"]: q for q in quotes}
    row_lookup = {r["代號"]: r for r in rows}
    paper_positions = account_positions
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
            sell_qty = min(held, int(limits.max_order_notional / float(price)))
            if sell_qty > 0:
                auto_candidates.append({
                    "方向":"賣出","代號":code,"名稱":row["名稱"],
                    "參考價":round(float(price),2),"股數":sell_qty,
                    "估算金額":round(sell_qty*float(price),2),
                    "策略依據":"跌破 EMA21／RSI 過熱／MACD 轉弱",
                    "報價時間":quote.get("報價時間"),
                })
        elif row.get("訊號") == "多因子觀察":
            atr_value = float(latest.atr) if pd.notna(latest.atr) else float(price)*0.04
            stop_distance = max(2.0*atr_value, float(price)*0.01)
            risk_qty = int(float(risk) / stop_distance)
            notional_qty = int(limits.max_order_notional / float(price))
            cash_available = float(account_snapshot["cash"])
            cash_qty = int(cash_available / (float(price)*1.001425))
            quantity = min(risk_qty, notional_qty, cash_qty)
            if quantity > 0:
                auto_candidates.append({
                    "方向":"買進","代號":code,"名稱":row["名稱"],
                    "參考價":round(float(price),2),"股數":quantity,
                    "估算金額":round(quantity*float(price),2),
                    "策略依據":"AI 多因子觀察＋ATR 風險預算控量",
                    "AI五日分數":row.get("AI五日上漲機率"),
                    "報價時間":quote.get("報價時間"),
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
        cash_available = float(account_snapshot["cash"])
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
    st.subheader("舊版固定百分比退出（實驗功能，預設停用）")
    st.warning("固定 5% 停損／10% 停利／5% 移動停利沒有完成台股樣本外驗證，不應作為正式策略。只有明確勾選後才可手動掃描；仍只建立模擬限價單。")
    if st.checkbox("我了解限制，暫時啟用舊版固定百分比實驗規則", value=False, key="enable_experimental_fixed") and st.button("掃描持倉並建立實驗性模擬退出委託", disabled=not session_open):
        stop_created = []
        current_orders = engine.list_orders()
        pending_sell_symbols = {o["symbol"] for o in current_orders
                                if o["status"] == "ACCEPTED" and o["side"] == "SELL"}
        pending_count = sum(1 for o in current_orders if o["status"] == "ACCEPTED")
        today_count = sum(1 for o in current_orders
                          if str(o["created_at"]).startswith(today_for_risk))
        current_equity = float(account_snapshot["cash"]) + sum(
            int(p["quantity"]) * float(quote_lookup.get(p["symbol"], {}).get("盤中參考價") or p["average_cost"])
            for p in account_snapshot["positions"])
        for q in quotes:
            if q.get("報價狀態") != "盤中更新" or q.get("盤中參考價") is None:
                continue
            try:
                signal = stop_monitor.evaluate_quote(
                    q["代號"], float(q["盤中參考價"]), str(q.get("報價時間") or ""),
                    "Yahoo Finance 1-minute", now=session_now)
            except ValueError:
                continue
            if not signal or signal["symbol"] in pending_sell_symbols:
                continue
            held = int(account_positions.get(signal["symbol"], 0))
            qty = min(held, signal["quantity"],
                      int(limits.max_order_notional / signal["price"]))
            if qty <= 0:
                continue
            intent = OrderIntent(
                symbol=signal["symbol"], side=Side.SELL, quantity=qty,
                limit_price=float(signal["price"]), reason=signal["reason"],
                idempotency_key=f"stop-{signal['symbol']}-{signal['reason']}-{today_for_risk}-{signal['quote_time']}")
            result = engine.submit(
                intent, cash=float(account_snapshot["cash"]),
                positions=account_positions, equity=current_equity,
                daily_realized_pnl=daily_pnl, open_order_count=pending_count,
                daily_order_count=today_count, now=session_now)
            if result.status == "ACCEPTED":
                pending_sell_symbols.add(signal["symbol"])
                pending_count += 1
                today_count += 1
                stop_created.append({"代號":signal["symbol"],"觸發":signal["reason"],
                                     "觸發價":signal["trigger_price"],"參考價":signal["price"],
                                     "委託股數":qty,"委託狀態":result.status})
        if stop_created:
            st.success(f"已建立 {len(stop_created)} 筆停損／停利模擬限價委託。")
            st.dataframe(pd.DataFrame(stop_created),use_container_width=True,hide_index=True)
        else:
            st.info("本輪沒有觸發條件，或沒有可建立的合規退出委託。")
        st.rerun()
    stop_events = stop_monitor.events()
    if stop_events:
        st.dataframe(pd.DataFrame(stop_events),use_container_width=True,hide_index=True)
    st.subheader("用盤中報價自動撮合")
    st.caption("只處理報價狀態為「盤中更新」的標的；來源標記 Yahoo Finance 1 分鐘資料，並由引擎再次檢查報價時間、120 秒新鮮度及交易時段。")
    if st.button("使用目前新鮮盤中報價撮合等待委託", disabled=not session_open):
        fill_results=[]
        for q in quotes:
            if q.get("報價狀態") != "盤中更新" or q.get("盤中參考價") is None:
                continue
            try:
                fill_results.extend(paper_account.process_quote(
                    q["代號"], float(q["盤中參考價"]), str(q.get("報價時間") or ""),
                    "Yahoo Finance 1-minute", now=session_now
                ))
            except ValueError:
                continue
        if fill_results:
            st.success(f"本次完成 {len(fill_results)} 筆模擬成交，已同步現金與持股。")
            st.dataframe(pd.DataFrame(fill_results),use_container_width=True,hide_index=True)
        else:
            st.info("本次沒有符合限價、報價新鮮度及交易時段條件的成交。")
        st.rerun()
    st.subheader("模擬委託佇列與成交狀態")
    order_rows = engine.list_orders()
    st.dataframe(pd.DataFrame(order_rows),use_container_width=True,hide_index=True)
    pending_orders = [order for order in order_rows if order["status"] == "ACCEPTED"]
    if pending_orders:
        pending_ids = [order["order_id"] for order in pending_orders]
        cancel_id = st.selectbox(
            "選擇要取消的等待委託", pending_ids,
            format_func=lambda oid: next(
                f"{item['symbol']} {item['side']} {item['quantity']} 股 @ {item['limit_price']}"
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
    st.caption("目前自動策略可產生模擬買進與賣出委託：買進依 AI 多因子訊號與 ATR 風險距離控量；賣出依 EMA21、RSI、MACD 退出條件。已加入 SQLite 模擬帳戶成交後現金／持股同步及新鮮報價觸發撮合；目前仍需使用者按鈕觸發每輪報價撮合，尚未有背景常駐即時監控、撤單重送或券商真實成交回報；不得視為已具備實盤自動交易能力。")
