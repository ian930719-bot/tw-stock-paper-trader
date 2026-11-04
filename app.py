import streamlit as st
import yfinance as yf
import requests
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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

def train_ml(d):
    features = ["rsi","macd","sig","r20","r60","atr","v20","ema21","ema50","ema200"]
    z = d.copy()
    z["target"] = (z.close.shift(-5)/z.close-1 > 0.015).astype(float)
    z.loc[z.index[-5:], "target"] = np.nan
    z = z.replace([np.inf,-np.inf], np.nan).dropna(subset=features+["target"])
    if len(z) < 300 or z.target.nunique() < 2:
        return None, None
    split = int(len(z)*0.8)
    train, test = z.iloc[:split], z.iloc[split:]
    if train.target.nunique() < 2 or test.target.nunique() < 2:
        return None, None
    model = RandomForestClassifier(n_estimators=250,max_depth=6,min_samples_leaf=8,class_weight="balanced",random_state=42,n_jobs=-1)
    model.fit(train[features], train.target.astype(int))
    metric = balanced_accuracy_score(test.target.astype(int), model.predict(test[features]))
    last = d.iloc[[-1]][features].replace([np.inf,-np.inf], np.nan)
    if last.isna().any(axis=None):
        return None, metric
    prob = float(model.predict_proba(last)[0][list(model.classes_).index(1)])
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
        rows.append({"代號":s,"名稱":W[s],"規則分數":int(score),"訊號":"觀察候選" if score>=70 else "等待" if score>=50 else "不進場","AI五日上漲機率":round(prob,3) if prob is not None else None,"模型測試平衡準確率":round(metric,3) if metric is not None else None,"三大法人買賣超(股)":net,"法人資料日期":inst.get("資料日期") if inst else None,"RSI14":round(r.rsi,1) if pd.notna(r.rsi) else None,"20日報酬":r.r20,"60日報酬":r.r60,"日線日期":r.date.strftime("%Y-%m-%d")})
if not data: st.error("目前無法取得歷史行情，請稍後重試。"); st.stop()
t1,t2,t3=st.tabs(["行情監控","歷史回測","資料限制"])
with t1:
    st.subheader("盤中參考行情（可能延遲）")
    st.dataframe(pd.DataFrame(quotes),use_container_width=True,hide_index=True)
    st.caption("盤中價取自 Yahoo Finance 1 分鐘資料，可能延遲或缺漏；請留意報價時間與分鐘差。")
    st.subheader("技術面、AI 與法人觀察")
    st.dataframe(pd.DataFrame(rows).sort_values("規則分數",ascending=False),use_container_width=True,hide_index=True)
    st.caption("規則評分為技術面初版，不是經訓練驗證的機器學習模型；未納入法人與融資融券。")
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
    st.markdown("歷史日線使用 Yahoo Finance 自動調整資料。盤中參考價可能延遲，非交易所認證即時行情。三大法人數據使用證交所 T86 日報，取最近可取得交易日；API 不可用時顯示空值。尚未接入法人歷史序列、融資融券與分點籌碼。")
    st.markdown("AI 欄位使用隨機森林，以歷史技術特徵預測未來 5 個交易日報酬是否超過 1.5%，按時間順序切分訓練與測試資料。平衡準確率是歷史測試指標，不代表未來勝率或獲利保證。")
    st.markdown("回測成本假設：手續費 0.1425%、一般股票賣出稅 0.3%、滑價 0.1%。尚未完整處理除權息、停牌、漲跌停、流動性與下市股偏誤。不連接券商、不會下單。")
    st.write("台北時間：",datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"))
