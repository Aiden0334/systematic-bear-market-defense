"""
# 5. 전략 실행 파일. 
# main_6y.py - 6년 Walk-Forward 
# Learning Data Revised.
# Systematic Bear Market Defense via XGBOOST Model
# 6 Year WalkForward Backtest
"""

import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from xgboost import XGBClassifier

BASE_DIR     = Path(__file__).parent
feature_cols = json.load(open(BASE_DIR/'models'/'feature_cols.json'))
feature_cols = [c for c in feature_cols if not any(c.startswith(p) for p in ['smc_','upper_wick','lower_wick','wick_'])]
if 'regime3' not in feature_cols: feature_cols.append('regime3')

TARGET_COL = 'target_dir_strong_4h'
MAX_HOLD, COOLDOWN, FEE_RATE = 64, 16, 0.0005
LONG_P, SHORT_P, SL_ATR = 0.40, 0.40, 2.0
ENABLE_SHORT = False
SEED = 42
MIN_TRAIN_DAYS = 700   # 약 1.92년 (729일이 윤년/평년 따라 변동되니 700으로 마진)
LABEL_MAP = {-1.0:0, 0.0:1, 1.0:2}
DATA_PATH = BASE_DIR/'data'/'processed'/'BTCUSDT_features_15m.parquet'
print(f"[6년 Walk-Forward + 안전장치] {TARGET_COL} | LongOnly:{not ENABLE_SHORT} | seed:{SEED}")
print(f"  안전장치: 학습 데이터 < {MIN_TRAIN_DAYS}일 시 거래 차단")


def add_features(df):
    df = df.copy(); c = df['kl_close']
    df['ma_short']=c.rolling(96).mean(); df['ma_mid']=c.rolling(480).mean(); df['ma_long']=c.rolling(19200).mean()
    df['regime3']=np.where((df['ma_short']>df['ma_mid'])&(df['ma_mid']>df['ma_long']),2,np.where((df['ma_short']<df['ma_mid'])&(df['ma_mid']<df['ma_long']),0,1))
    h,l,v=df['kl_high'].values,df['kl_low'].values,df['kl_volume'].values
    hlc3=(h+l+c.values)/3; n,prd=len(df),20
    phi,pli=np.zeros(n,int),np.zeros(n,int)
    for i in range(prd,n):
        phi[i]=i if h[i]==h[max(0,i-prd):i+1].max() else phi[i-1]
        pli[i]=i if l[i]==l[max(0,i-prd):i+1].min() else pli[i-1]
    d=np.where(phi>pli,1,-1); vw=np.full(n,np.nan); cpv=cv=pd_=0
    for i in range(n):
        if d[i]!=pd_:
            ai=phi[i] if d[i]>0 else pli[i]; cpv,cv,pd_=hlc3[ai]*v[ai],v[ai],d[i]
        else: cpv+=hlc3[i]*v[i]; cv+=v[i]
        if cv>0: vw[i]=cpv/cv
    df['swing_vwap']=vw; df['swing_vwap_dist']=(c.values-vw)/(c.values+1e-12)
    df['swing_vwap_dir']=d.astype(float); df['swing_vwap_above']=(c.values>vw).astype(float)
    return df


def compute_atr(df,length=14):
    h,l,c=df['kl_high'],df['kl_low'],df['kl_close']
    tr=pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/length,adjust=False,min_periods=length).mean()


def train_xgb(tr_df):
    tr=tr_df.dropna(subset=[TARGET_COL])
    mdl=XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.05,subsample=0.8,
                      colsample_bytree=0.8,min_child_weight=10,gamma=1,
                      reg_alpha=0.1,reg_lambda=1.0,random_state=SEED,n_jobs=-1,tree_method='hist')
    mdl.fit(tr[feature_cols],tr[TARGET_COL].map(LABEL_MAP),verbose=False)
    print(f"  XGB 학습: {len(tr):,}개"); return mdl


def calc_bnh(df, start, end):
    ts,te=pd.Timestamp(start,tz='UTC'),pd.Timestamp(end,tz='UTC')
    tdf=df[(df.index>=ts)&(df.index<=te)]
    if len(tdf)<2: return 0.0
    return (tdf['kl_close'].iloc[-1]/tdf['kl_close'].iloc[0]-1)*100


def backtest(df,xgb,start,end,cap=10000):
    ts,te=pd.Timestamp(start,tz='UTC'),pd.Timestamp(end,tz='UTC')
    buf=df[(df.index>=ts-pd.Timedelta(days=210))&(df.index<=te)].copy()
    buf['atr']=compute_atr(buf); tdf=buf[buf.index>=ts].copy()
    if len(tdf)<100: return None
    proba=xgb.predict_proba(tdf[feature_cols])
    up_p,dn_p=proba[:,2],proba[:,0]
    print(f"  proba[up] 50%={np.percentile(up_p,50):.3f} 90%={np.percentile(up_p,90):.3f} max={up_p.max():.3f}")
    ls=(up_p>LONG_P).astype(int)
    if ENABLE_SHORT:
        ss=(dn_p>SHORT_P).astype(int)
    else:
        ss=np.zeros_like(ls)
    print(f"  롱신호:{ls.sum()} 숏신호:{ss.sum()}")

    cl,at=tdf['kl_close'].values,tdf['atr'].values
    pos,ep,ei,lei,sl,trades=0,0.,0,-COOLDOWN,0.,[]
    sl_hits=0
    for i in range(1,len(cl)):
        if np.isnan(at[i]) or np.isnan(cl[i]): continue
        if pos==1 and cl[i]<=sl:
            pnl=(cl[i]-ep)/ep-FEE_RATE; cap*=(1+pnl); trades.append({'t':'long','pnl':pnl,'exit':'sl'}); pos=0; lei=i; sl_hits+=1; continue
        elif pos==-1 and cl[i]>=sl:
            pnl=(ep-cl[i])/ep-FEE_RATE; cap*=(1+pnl); trades.append({'t':'short','pnl':pnl,'exit':'sl'}); pos=0; lei=i; sl_hits+=1; continue
        if pos==1 and (i-ei)>=MAX_HOLD:
            pnl=(cl[i]-ep)/ep-FEE_RATE; cap*=(1+pnl); trades.append({'t':'long','pnl':pnl,'exit':'t'}); pos=0; lei=i
        elif pos==-1 and (i-ei)>=MAX_HOLD:
            pnl=(ep-cl[i])/ep-FEE_RATE; cap*=(1+pnl); trades.append({'t':'short','pnl':pnl,'exit':'t'}); pos=0; lei=i
        if pos==0 and (i-lei)>=COOLDOWN:
            if ls[i]: pos,ep,ei,sl=1,cl[i],i,cl[i]-SL_ATR*at[i]
            elif ss[i]: pos,ep,ei,sl=-1,cl[i],i,cl[i]+SL_ATR*at[i]
    bnh=(tdf['kl_close'].iloc[-1]/tdf['kl_close'].iloc[0]-1)*100
    if not trades:
        print("  거래없음")
        return {'ret':0,'sharpe':0,'mdd':0,'n':0,'win':0,'bnh':bnh,'cagr':0,'pf':0}
    td=pd.DataFrame(trades); cv=pd.Series([10000])
    for p in td['pnl']: cv=pd.concat([cv,pd.Series([cv.iloc[-1]*(1+p)])])
    cv=cv.reset_index(drop=True); r=td['pnl']; days=(tdf.index.max()-tdf.index.min()).days; yr=max(days/365,0.01)
    lt=td[td['t']=='long']; st=td[td['t']=='short']
    print(f"  손절히트:{sl_hits}/{len(td)} ({sl_hits/len(td):.1%})")
    if len(lt)>0: print(f"  롱:{len(lt)}개 승률:{(lt['pnl']>0).mean():.3f} 평균:{lt['pnl'].mean()*100:.3f}%")
    if len(st)>0: print(f"  숏:{len(st)}개 승률:{(st['pnl']>0).mean():.3f} 평균:{st['pnl'].mean()*100:.3f}%")
    return {'ret':(cap/10000-1)*100,'cagr':((cap/10000)**(1/yr)-1)*100,
            'sharpe':r.mean()/r.std()*np.sqrt(252) if r.std()>0 else 0,
            'mdd':((cv-cv.cummax())/cv.cummax()).min()*100,'win':(r>0).mean(),
            'pf':r[r>0].sum()/r[r<0].abs().sum() if r[r<0].abs().sum()>0 else 0,
            'n':len(td),'bnh':bnh}


if __name__=='__main__':
    print("데이터 로드 중..."); df=pd.read_parquet(DATA_PATH)
    df=add_features(df)
    print(f"BTC:{df.shape} 범위:{df.index.min()} ~ {df.index.max()}\n{'='*70}")

    wf=[('2021', '2019-09-25', '2020-12-31', '2021-01-01', '2021-12-31'),
        ('2022', '2020-01-01', '2021-12-31', '2022-01-01', '2022-12-31'),
        ('2023', '2021-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
        ('2024', '2022-01-01', '2023-12-31', '2024-01-01', '2024-12-31'),
        ('2025', '2023-01-01', '2024-12-31', '2025-01-01', '2025-12-31'),
        ('2026', '2024-01-01', '2025-12-31', '2026-01-01', '2026-04-23')]

    results=[]
    for yr,trs,tre,ts,te2 in wf:
        print(f"\n[{yr}] 학습:{trs}~{tre} → 테스트:{ts}~{te2}")
        trs_utc,tre_utc=pd.Timestamp(trs,tz='UTC'),pd.Timestamp(tre,tz='UTC')
        train_days=(tre_utc-trs_utc).days
        print(f"  학습 기간: {train_days}일")

        if train_days < MIN_TRAIN_DAYS:
            bnh=calc_bnh(df, ts, te2)
            print(f"  {train_days}일 < {MIN_TRAIN_DAYS}일 → 거래 차단 (현금 보유)")
            print(f"  수익:0.00% 거래:0 BnH:{bnh:.1f}%")
            results.append({'year':yr,'ret':0.0,'sharpe':0,'mdd':0,'n':0,'win':0,
                           'bnh':bnh,'cagr':0,'pf':0,'blocked':True})
            continue

        tr_df=df[(df.index>=trs_utc)&(df.index<=tre_utc)]
        if len(tr_df.dropna(subset=[TARGET_COL]))<5000:
            print(f"  학습 데이터 부족({len(tr_df)}개), 건너뜀")
            continue
        xm=train_xgb(tr_df)
        res=backtest(df,xm,ts,te2)
        if res is not None:
            res['year']=yr; res['blocked']=False
            results.append(res)
            print(f"  수익:{res['ret']:.2f}% Sharpe:{res['sharpe']:.3f} MDD:{res['mdd']:.2f}% 거래:{res['n']} 승률:{res['win']:.3f} BnH:{res['bnh']:.1f}%")

    if results:
        dr=pd.DataFrame(results)
        print(f"\n{'='*70}\n## 6년 종합 (안전장치 포함) ##\n{'='*70}")
        print(f"\n{'년도':>6} | {'상태':>5} | {'수익':>8} | {'Sharpe':>7} | {'MDD':>7} | {'거래':>5} | {'승률':>6} | {'BnH':>8} | {'알파':>8}")
        print("-"*90)
        for _,row in dr.iterrows():
            alpha = row['ret'] - row['bnh']
            status = "차단" if row['blocked'] else "운용"
            print(f"{row['year']:>6} | {status:>5} | {row['ret']:>7.2f}% | {row['sharpe']:>7.3f} | {row['mdd']:>6.2f}% | {int(row['n']):>5} | {row['win']:>6.3f} | {row['bnh']:>7.1f}% | {alpha:>+7.2f}%")

        active = dr[~dr['blocked']]
        all_yrs = dr

        print(f"\n## 운용 연도만 ({len(active)}개) ##")
        if len(active)>0:
            print(f"평균 수익: {active['ret'].mean():.2f}%")
            print(f"중앙값 수익: {active['ret'].median():.2f}%")
            print(f"수익 std: {active['ret'].std():.2f}%")
            print(f"평균 Sharpe: {active['sharpe'].mean():.3f}")
            print(f"평균 MDD: {active['mdd'].mean():.2f}%")
            print(f"평균 거래 수: {active['n'].mean():.1f}")
            print(f"총 거래 수: {int(active['n'].sum())}")
            print(f"흑자 연도: {(active['ret']>0).sum()}/{len(active)}")
            print(f"BnH 대비 알파 양수: {((active['ret']-active['bnh'])>0).sum()}/{len(active)}")

        print(f"\n## 전체 6년 (차단 포함) ##")
        print(f"평균 수익: {all_yrs['ret'].mean():.2f}%")
        print(f"흑자 연도: {(all_yrs['ret']>0).sum()}/{len(all_yrs)}")
        print(f"손실 연도: {(all_yrs['ret']<0).sum()}/{len(all_yrs)}")
        print(f"차단(0% 수익): {all_yrs['blocked'].sum()}/{len(all_yrs)}")

        cum_ret = 1.0
        for r in all_yrs['ret']:
            cum_ret *= (1 + r/100)
        cum_bnh = 1.0
        for r in all_yrs['bnh']:
            cum_bnh *= (1 + r/100)
        print(f"\n## 복리 누적 수익 (6년) ##")
        print(f"전략: {(cum_ret-1)*100:+.1f}% (초기 1.0 → 최종 {cum_ret:.2f})")
        print(f"BnH:  {(cum_bnh-1)*100:+.1f}% (초기 1.0 → 최종 {cum_bnh:.2f})")

        dr.to_csv(BASE_DIR/'6y_walkforward_safe_results.csv', index=False)
        print(f"\n저장: {BASE_DIR/'6y_walkforward_safe_results.csv'}")