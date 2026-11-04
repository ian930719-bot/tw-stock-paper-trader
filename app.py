import streamlit as st
import yfinance as yf
import requests
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from research_engine import STRATEGIES, daily_strategy_returns, performance, fundamental_snapshot, shared_capital_backtest
from datetime import datetime
from zoneinfo import ZoneInfo
TZ=ZoneInfo("Asia/Taipei")
W={"2330":"台積電","2454":"聯發科","2308":"台達電","2317":"鴻海","2382":"廣達","3231":"緯創","3017":"奇鋐","2603":"長榮","2881":"富邦金","2891":"中信金"}
st.set_page_config(page_title="台股模擬交易",layout="wide")
st.title("台股模擬交易研究系統")
st.caption("研究用途；行情可能延遲，非交易所認證即時行情。不連接券商、不會送出真實委託。")
with st.sidebar:
    ss=st.multiselect("觀察股票",list(W),default=list(W),format_func=lambda x:x+" "+W[x])
    period=st.selectbox("歷史期間",["5y","3y","2y","1y"])
    capital=st.number_input("模擬本金",100000,10000000,1000000,100000)
    risk=st.number_input("單筆風險上限",500,50000,5000,500)
    if st.button("重新抓取"): st.cache_data.clear()
if not ss: st.stop()
@st.cache_data(ttl=300)
def getdata(code,p):
    for suffix in [".TW",".TWO"]:
        try:
            d=yf.Ticker(code+suffix).history(period=p,interval="1d",auto_adjust=True)
            if not d.empty:
                d=d.reset_index(); d=d.rename(columns={"Date":"date","Datetime":"date","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
                d["date"]=pd.to_datetime(d.date).dt.tz_localize(None)
                return d[["date","open","high","low","close","volume"]].dropna()
        except Exception: pass
    return pd.DataFrame()
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
        return None, None
    split = int(len(z)*0.8)
    # Purge the final five training labels: their five-day outcome window
    # overlaps the first test observations and would leak future prices.
    train, test = z.iloc[:max(0, split-5)], z.iloc[split:]
    if len(train) < 100 or train.target.nunique() < 2 or test.target.nunique() < 2:
        return None, None
    params=dict(n_estimators=250,max_depth=6,min_samples_leaf=8,
                class_weight="balanced",random_state=42,n_jobs=-1)
    evaluation_model = RandomForestClassifier(**params)
    evaluation_model.fit(train[features], train.target.astype(int))
    metric = balanced_accuracy_score(
        test.target.astype(int), evaluation_model.predict(test[features])
    )
    last = d.iloc[[-1]][features].replace([np.inf,-np.inf], np.nan)
    if last.isna().any(axis=None):
        return None, metric
    # Refit a separate production model on every fully labelled observation.
    forecast_model = RandomForestClassifier(**params)
    forecast_model.fit(z[features], z.target.astype(int))
    prob = float(forecast_model.predict_proba(last)[0][list(forecast_model.classes_).index(1)])
    return prob, metric

inst_data=get_institutional()
data={}; rows=[]; quotes=[]
with st.spinner("讀取歷史行情…"):
    for s in ss:
        d=getdata(s,period)
        if d.empty: continue
        d=ind(d); d["obvp"]=d.obv.shift(5); data[s]=d; r=d.iloc[-1]
        inst=inst_data.get(s); net=inst.get("三大法人買賣超股數") if inst else None
        base=20*(r.close>r.ema21 and r.up)+10*(r.close>r.ema50)+10*(r.close>r.ema200)+10*(r.r20>0)+5*(r.r60>0)+10*(50<=r.rsi<=70)+10*(r.macd>r.sig)+10*(pd.notna(r.v20) and r.volume>1.5*r.v20 and r.close>r.h20)+5*(r.obv>r.obvp if pd.notna(r.obvp) else False)
        score=base+(10 if net is not None and net>0 else 0)
        prob,metric=train_ml(d); price,qt,age=get_quote(s)
        quotes.append({"代號":s,"名稱":W[s],"盤中參考價":price,"報價時間":qt,"資料約幾分鐘前":age})
        rows.append({"代號":s,"名稱":W[s],"規則分數":int(score),"訊號":("多因子觀察" if score>=70 and prob is not None and prob>=0.55 else "技術候選（AI資料不足）" if score>=70 and prob is None else "AI未確認" if score>=70 else "等待" if score>=50 else "不進場"),"AI五日上漲機率":round(prob,3) if prob is not None else None,"模型測試平衡準確率":round(metric,3) if metric is not None else None,"三大法人買賣超(股)":net,"法人資料日期":inst.get("資料日期") if inst else None,"RSI14":round(r.rsi,1) if pd.notna(r.rsi) else None,"20日報酬":r.r20,"60日報酬":r.r60,"日線日期":r.date.strftime("%Y-%m-%d")})
if not data: st.error("目前無法取得歷史行情，請稍後重試。"); st.stop()
t1,t2,t3,t4,t5,t6=st.tabs(["行情監控","歷史回測","模擬帳本","資料限制","大師策略回測","整合研究中心"])
with t1:
    st.subheader("盤中參考行情（可能延遲）")
    st.dataframe(pd.DataFrame(quotes),use_container_width=True,hide_index=True)
    st.caption("盤中價取自 Yahoo Finance 1 分鐘資料，可能延遲或缺漏；請留意報價時間與分鐘差。")
    st.subheader("技術面、AI 與法人觀察")
    st.dataframe(pd.DataFrame(rows).sort_values("規則分數",ascending=False),use_container_width=True,hide_index=True)
    st.caption("規則分數已加入最新可取得的三大法人買賣超加分；AI 機率由隨機森林模型提供。法人歷史序列、融資融券與分點籌碼尚未接入。")
    s=st.selectbox("查看個股",list(data),format_func=lambda z:z+" "+W[z]); d=data[s]
    f=go.Figure(); f.add_trace(go.Scatter(x=d.date,y=d.close,name="還原收盤")); f.add_trace(go.Scatter(x=d.date,y=d.ema21,name="EMA21")); f.add_trace(go.Scatter(x=d.date,y=d.ema50,name="EMA50")); st.plotly_chart(f,use_container_width=True)
    st.download_button("下載行情指標 CSV",pd.concat([v.assign(代號=k) for k,v in data.items()]).to_csv(index=False).encode("utf-8-sig"),"tw_stock.csv","text/csv")
with t2:
    s=st.selectbox("回測標的",list(data),key="bt"); x=data[s].copy().reset_index(drop=True)
    x["entry"]=(x.close>x.ema21)&x.up&x.rsi.between(50,70)&(x.macd>x.sig)&(x.volume>1.5*x.v20)&(x.close>x.h20)
    x["exit"]=(x.close<x.ema21)|(x.rsi>75)|(x.macd<x.sig)
    cash=float(capital); qty=0; stop=0; trades=[]; curve=[]
    for i in range(len(x)-1):
        r,n=x.iloc[i],x.iloc[i+1]; op=float(n.open)
        if qty and (r.exit or n.low<=stop):
            px=min(op,stop) if n.low<=stop else op; gross=qty*px*.999
            cash+=gross*(1-.001425-.003); trades[-1]["損益"]=cash-trades[-1]["投入前現金"]; trades[-1]["出場日"]=str(n.date.date()); qty=0
        if qty==0 and r.entry:
            px=op*1.001; stop=px-2*(r.atr if pd.notna(r.atr) else px*.04)
            q=max(0,min(int(risk/max(px-stop,.01)),int(cash/(px*1.001425))))
            if q: trades.append({"進場日":str(n.date.date()),"股數":q,"進場價":px,"投入前現金":cash}); cash-=q*px*1.001425; qty=q
        curve.append({"日期":r.date,"資產":cash+qty*r.close})
    if qty:
        px=float(x.iloc[-1].close)*.999; cash+=qty*px*(1-.001425-.003); trades[-1]["損益"]=cash-trades[-1]["投入前現金"]; trades[-1]["備註"]="期末平倉"
    a,b,c=st.columns(3); a.metric("期末資產",f"NTD {cash:,.0f}"); b.metric("交易筆數",len(trades))
    c.metric("勝率",f"{sum(z.get('損益',0)>0 for z in trades)/len(trades):.1%}" if trades else "無交易")
    st.warning("簡化回測：收盤產生訊號、次日開盤成交，含估算手續費、證交稅及滑價；未完整處理除權息、停牌、漲跌停、流動性與下市股偏誤。")
    if curve:
        f=go.Figure(go.Scatter(x=[z["日期"] for z in curve],y=[z["資產"] for z in curve])); f.add_hline(y=capital,line_dash="dash"); st.plotly_chart(f,use_container_width=True)
    st.dataframe(pd.DataFrame(trades).drop(columns=["投入前現金"],errors="ignore"),use_container_width=True)
with t3:
    st.subheader("純模擬交易帳本")
    st.caption("只在目前瀏覽器工作階段保存，不會連接券商；工作階段重啟或服務重啟時可能清空。")
    if "paper_cash" not in st.session_state:
        st.session_state.paper_cash=float(capital)
        st.session_state.paper_positions={}
        st.session_state.paper_trades=[]
    ps=st.selectbox("模擬交易股票",list(data),format_func=lambda z:z+" "+W[z],key="paper_symbol")
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
    for code,q in st.session_state.paper_positions.items():
        px=float(data[code].iloc[-1].close)
        value=q*px
        total+=value
        if q:
            holdings.append({"代號":code,"名稱":W[code],"股數":q,"參考收盤價":px,"市值":value})
    a,b=st.columns(2)
    a.metric("模擬可用現金",f"NTD {st.session_state.paper_cash:,.0f}")
    b.metric("帳戶總資產",f"NTD {total:,.0f}")
    st.subheader("目前持倉")
    st.dataframe(pd.DataFrame(holdings),use_container_width=True,hide_index=True)
    st.subheader("模擬成交紀錄")
    st.dataframe(pd.DataFrame(st.session_state.paper_trades),use_container_width=True,hide_index=True)
    if st.session_state.paper_trades:
        st.download_button("下載模擬成交紀錄 CSV",pd.DataFrame(st.session_state.paper_trades).to_csv(index=False).encode("utf-8-sig"),"paper_trades.csv","text/csv")
with t4:
    st.markdown("歷史日線使用 Yahoo Finance 自動調整資料。盤中參考價可能延遲，非交易所認證即時行情。三大法人數據使用證交所 T86 日報，取最近可取得交易日；API 不可用時顯示空值。尚未接入法人歷史序列、融資融券與分點籌碼。")
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
    st.subheader("投資大師策略：歷史比較回測")
    st.caption("目前先測試三種可用日線資料量化的技術策略代理。買賣以訊號日後一交易日開盤價成交；成本包含估算手續費、賣出證交稅及滑價。")
    bs=st.selectbox("策略回測標的",list(data),format_func=lambda z:z+" "+W[z],key="master_bt")
    initial=st.number_input("策略回測本金",100000,10000000,1000000,100000,key="master_capital")
    frame=data[bs].copy().reset_index(drop=True)
    close=frame.close
    high52=frame.high.rolling(252,min_periods=120).max()
    low52=frame.low.rolling(252,min_periods=120).min()
    sma150=close.rolling(150).mean()
    sma200=close.rolling(200).mean()
    sma200_old=sma200.shift(20)
    sma50=close.rolling(50).mean()
    rel20=close.pct_change(20)
    rel60=close.pct_change(60)
    # Technical proxies only; not full CAN SLIM or fundamental screens.
    signal_map={
        "Minervini 趨勢模板":(
            (close>sma50)&(sma50>sma150)&(sma150>sma200)&(sma200>sma200_old)&
            (close>=low52*1.25)&(close>=high52*0.75)&(rel20>0)&(rel60>0),
            (close<sma50)|(close<sma150)
        ),
        "O’Neil CAN SLIM 技術代理":(
            (close>frame.h20)&(frame.volume>1.4*frame.v20)&(close>sma50)&(rel20>0),
            (close<sma50)|(close<frame.ema21)
        ),
        "Livermore 順勢突破":(
            (close>frame.h20)&(close>frame.ema50)&frame.up&(rel20>0),
            (close<frame.ema21)|(close<frame.ema50)
        )
    }
    def run_master_backtest(entry_signal, exit_signal, hist, starting_cash):
        cash=float(starting_cash); qty=0.0; entry_cost=0.0; stop=0.0
        trades=[]; curve=[]; fee=0.001425; tax=0.003; slip=0.001
        for i in range(len(hist)-1):
            today=hist.iloc[i]; nxt=hist.iloc[i+1]
            next_open=float(nxt.open)
            if qty>0 and (bool(exit_signal.iloc[i]) or float(nxt.low)<=stop):
                fill=min(next_open,stop) if float(nxt.low)<=stop else next_open
                fill*=1-slip
                proceeds=qty*fill*(1-fee-tax)
                pnl=proceeds-entry_cost
                trades.append({"進場日":entry_date,"出場日":str(nxt.date.date()),"股數":qty,"損益":pnl,"報酬率":pnl/entry_cost if entry_cost else 0})
                cash+=proceeds; qty=0.0; entry_cost=0.0; stop=0.0
            if qty==0 and bool(entry_signal.iloc[i]):
                fill=next_open*(1+slip)
                atr=float(today.atr) if pd.notna(today.atr) else fill*0.04
                stop=fill-2*atr
                risk_budget=starting_cash*0.005
                shares=max(0,int(risk_budget/max(fill-stop,0.01)))
                shares=min(shares,int(cash/(fill*(1+fee))))
                if shares>0:
                    entry_cost=shares*fill*(1+fee)
                    cash-=entry_cost; qty=float(shares); entry_date=str(nxt.date.date())
            curve.append({"日期":today.date,"資產":cash+qty*float(today.close)})
        if qty>0:
            last=hist.iloc[-1]; fill=float(last.close)*(1-slip)
            proceeds=qty*fill*(1-fee-tax); pnl=proceeds-entry_cost
            trades.append({"進場日":entry_date,"出場日":str(last.date.date()),"股數":qty,"損益":pnl,"報酬率":pnl/entry_cost if entry_cost else 0})
            cash+=proceeds
        eq=pd.DataFrame(curve)
        final_value=float(cash)
        if not eq.empty:
            vals=eq["資產"].astype(float)
            peaks=vals.cummax()
            maxdd=float((vals/peaks-1).min())
            daily=vals.pct_change().dropna()
            sharpe=float((daily.mean()/daily.std())*np.sqrt(252)) if daily.std()>0 else 0.0
            years=max((pd.to_datetime(eq["日期"].iloc[-1])-pd.to_datetime(eq["日期"].iloc[0])).days/365.25,0.1)
            cagr=(final_value/starting_cash)**(1/years)-1 if final_value>0 else -1.0
        else:
            maxdd=0.0; sharpe=0.0; cagr=0.0
        winrate=float(np.mean([t["損益"]>0 for t in trades])) if trades else 0.0
        return {"期末資產":final_value,"總報酬":final_value/starting_cash-1,"年化報酬":cagr,"最大回撤":maxdd,"Sharpe":sharpe,"交易筆數":len(trades),"勝率":winrate},eq,pd.DataFrame(trades)
    results=[]; curves={}; trade_tables={}
    for name,(ent,ex) in signal_map.items():
        metrics,equity,trades=run_master_backtest(ent.fillna(False),ex.fillna(False),frame,initial)
        metrics["策略"]=name
        results.append(metrics); curves[name]=equity; trade_tables[name]=trades
    result_df=pd.DataFrame(results)[["策略","期末資產","總報酬","年化報酬","最大回撤","Sharpe","交易筆數","勝率"]]
    st.dataframe(result_df.style.format({"期末資產":"{:,.0f}","總報酬":"{:.1%}","年化報酬":"{:.1%}","最大回撤":"{:.1%}","Sharpe":"{:.2f}","勝率":"{:.1%}"}),use_container_width=True,hide_index=True)
    fig=go.Figure()
    for name,equity in curves.items():
        if not equity.empty:
            fig.add_trace(go.Scatter(x=equity["日期"],y=equity["資產"],name=name))
    fig.add_hline(y=initial,line_dash="dash")
    fig.update_layout(title="不同策略資產曲線（含估算交易成本）",yaxis_title="新台幣")
    st.plotly_chart(fig,use_container_width=True)
    chosen=st.selectbox("查看交易明細",list(signal_map),key="master_trade_detail")
    st.dataframe(trade_tables[chosen],use_container_width=True,hide_index=True)
    st.warning("這是單一股票、單策略、簡化的歷史比較，不是全市場逐期選股測試。尚未納入股息再投資、漲跌停成交限制、流動性衝擊及完整存活者偏誤處理。樣本外／滾動驗證尚待下一階段完成；歷史績效不保證未來表現。")


with t6:
    st.subheader("整合研究中心")
    st.markdown("**基本面快照（最新可取得資料）**")
    st.caption("以下為目前可取得的最新快照，不是歷史逐期財報；不會用來回填歷史交易訊號。")
    if st.button("載入基本面資料",key="fund_load"):
        with st.spinner("讀取財務資料…"):
            st.session_state["fund_data"]=fundamental_snapshot(tuple(data.keys()))
    if "fund_data" in st.session_state:
        st.dataframe(st.session_state["fund_data"],use_container_width=True,hide_index=True)

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
        st.caption("三段期間依共同交易日曆切分；每段開始會清空前段持倉並以本金重新開始。策略參數不會依樣本外結果調整。")
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
                        max_positions=int(maxpos),warmup=260,
                        report_start_date=segment_start
                    ) if segment_frames else None
                    if result:
                        periods[label]=result
                for label,result in periods.items():
                    m=result["metrics"]
                    summaries.append({
                        "策略":strategy,"期間":label,"期末資產":m["期末資產"],
                        "總報酬":m["總報酬"],"年化報酬":m["年化報酬"],
                        "最大回撤":m["最大回撤"],"Sharpe":m["Sharpe"],
                        "交易筆數":len(result["trades"]),
                        "有效股票數":result["coverage"]["symbols_used"],
                    })
                    curve_map[(strategy,label)]=result["equity"]
                    trade_map[(strategy,label)]=result["trades"]
            st.session_state["portfolio_summary"]=pd.DataFrame(summaries)
            st.session_state["portfolio_curves"]=curve_map
            st.session_state["portfolio_trades"]=trade_map

        if "portfolio_summary" in st.session_state:
            summary=st.session_state["portfolio_summary"]
            st.dataframe(summary.style.format({
                "期末資產":"{:,.0f}","總報酬":"{:.1%}","年化報酬":"{:.1%}",
                "最大回撤":"{:.1%}","Sharpe":"{:.2f}",
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
            st.warning("這是歷史研究，不是投資建議。仍未完整納入股息再投資、除權息、漲跌停排隊、停牌、流動性衝擊、下市股及歷史成分股偏誤；樣本外結果不能回頭用於調參。")
