# BTC와 모델, 파라미터, 임계값 상관없이 모두 동일한 셋업. - BTC 모델 과적합인지 다른 종목에도 검증.
# Validation File (BTC = In-sample, ETH = Out-Sample)
# walkforward_eth1.py (=main_eth.py)
# main_eth.py - ETH Out-of-Sample 검증
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).parent
feature_cols = json.load(open(BASE_DIR/'models'/'feature_cols.json'))
feature_cols = [c for c in feature_cols if not any(c.startswith(p) for p in ['smc_','upper_wick','lower_wick','wick_'])]
if 'regime3' not in feature_cols: feature_cols.append('regime3')

TARGET_COL = 'target_dir_strong_4h'
MAX_HOLD, COOLDOWN, FEE_RATE = 64, 16, 0.0005
LONG_P, SL_ATR = 0.40, 2.0
ENABLE_SHORT, SEED, MIN_TRAIN_DAYS = False, 42, 700
LABEL_MAP = {-1.0:0, 0.0:1, 1.0:2}

# === ETH 경로 ===
DATA_PATH = BASE_DIR/'data'/'processed'/'ETHUSDT_features_15m.parquet'
ASSET = 'ETH'
print(f"[{ASSET} Out-of-Sample 검증] {TARGET_COL} | LongOnly | seed:{SEED}")
print(f"  features: {len(feature_cols)}개 (BTC와 동일)")


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
    return (tdf['kl_close'].iloc[-1]/tdf['kl_close'].iloc[0]-1)*100 if len(tdf)>=2 else 0.0


def backtest(df,xgb,start,end,cap=10000):
    ts,te=pd.Timestamp(start,tz='UTC'),pd.Timestamp(end,tz='UTC')
    buf=df[(df.index>=ts-pd.Timedelta(days=210))&(df.index<=te)].copy()
    buf['atr']=compute_atr(buf); tdf=buf[buf.index>=ts].copy()
    if len(tdf)<100: return None
    proba=xgb.predict_proba(tdf[feature_cols])
    up_p,dn_p=proba[:,2],proba[:,0]
    print(f"  proba[up] 50%={np.percentile(up_p,50):.3f} 90%={np.percentile(up_p,90):.3f} max={up_p.max():.3f}")
    ls=(up_p>LONG_P).astype(int)
    ss=np.zeros_like(ls)
    print(f"  롱신호:{ls.sum()}")

    cl,at=tdf['kl_close'].values,tdf['atr'].values
    pos,ep,ei,lei,sl,trades=0,0.,0,-COOLDOWN,0.,[]
    sl_hits=0
    entry_regimes = []
    for i in range(1,len(cl)):
        if np.isnan(at[i]) or np.isnan(cl[i]): continue
        if pos==1 and cl[i]<=sl:
            pnl=(cl[i]-ep)/ep-FEE_RATE; cap*=(1+pnl); trades.append({'pnl':pnl,'exit':'sl'}); pos=0; lei=i; sl_hits+=1; continue
        if pos==1 and (i-ei)>=MAX_HOLD:
            pnl=(cl[i]-ep)/ep-FEE_RATE; cap*=(1+pnl); trades.append({'pnl':pnl,'exit':'t'}); pos=0; lei=i
        if pos==0 and (i-lei)>=COOLDOWN and ls[i]:
            pos,ep,ei,sl=1,cl[i],i,cl[i]-SL_ATR*at[i]
    bnh=(tdf['kl_close'].iloc[-1]/tdf['kl_close'].iloc[0]-1)*100
    if not trades:
        print("  거래없음")
        return {'ret':0,'sharpe':0,'mdd':0,'n':0,'win':0,'bnh':bnh,'cagr':0}
    td=pd.DataFrame(trades); cv=pd.Series([10000])
    for p in td['pnl']: cv=pd.concat([cv,pd.Series([cv.iloc[-1]*(1+p)])])
    cv=cv.reset_index(drop=True); r=td['pnl']; days=(tdf.index.max()-tdf.index.min()).days; yr=max(days/365,0.01)
    print(f"  거래:{len(td)} 손절히트:{sl_hits} 승률:{(r>0).mean():.3f} 평균:{r.mean()*100:.3f}%")
    return {'ret':(cap/10000-1)*100,
            'cagr':((cap/10000)**(1/yr)-1)*100,
            'sharpe':r.mean()/r.std()*np.sqrt(len(r)/yr) if r.std()>0 else 0,
            'mdd':((cv-cv.cummax())/cv.cummax()).min()*100,
            'n':len(td),'win':(r>0).mean(),'bnh':bnh}


if __name__=='__main__':
    print(f"\n{ASSET} 데이터 로드...")
    if not DATA_PATH.exists():
        print(f" 파일 없음: {DATA_PATH}")
        import sys; sys.exit(1)
    df = pd.read_parquet(DATA_PATH); df = add_features(df)
    print(f"{ASSET}: {df.shape} 범위: {df.index.min()} ~ {df.index.max()}")

    if TARGET_COL not in df.columns:
        print(f" Target 컬럼 없음: {TARGET_COL}")
        import sys; sys.exit(1)

    # Target 분포
    print(f"\n## {ASSET} {TARGET_COL} 분포 ##")
    print(df[TARGET_COL].value_counts(normalize=True, dropna=False).sort_index())
    print(f"  (참고) BTC: -1: 15.6%, 0: 67.7%, 1: 16.7%")

    print(f"\n{'='*70}")
    wf=[('2021','2019-11-27','2020-12-31','2021-01-01','2021-12-31'),  # ETH 시작일 맞춤
        ('2022','2020-01-01','2021-12-31','2022-01-01','2022-12-31'),
        ('2023','2021-01-01','2022-12-31','2023-01-01','2023-12-31'),
        ('2024','2022-01-01','2023-12-31','2024-01-01','2024-12-31'),
        ('2025','2023-01-01','2024-12-31','2025-01-01','2025-12-31'),
        ('2026','2024-01-01','2025-12-31','2026-01-01','2026-04-23')]

    results=[]
    for yr,trs,tre,ts,te2 in wf:
        print(f"\n[{yr}] 학습:{trs}~{tre} → 테스트:{ts}~{te2}")
        trs_u,tre_u=pd.Timestamp(trs,tz='UTC'),pd.Timestamp(tre,tz='UTC')
        train_days=(tre_u-trs_u).days
        print(f"  학습 기간: {train_days}일")
        if train_days < MIN_TRAIN_DAYS:
            bnh=calc_bnh(df,ts,te2)
            print(f"  {train_days}일 < {MIN_TRAIN_DAYS}일 → 거래 차단")
            print(f"  BnH:{bnh:.1f}%")
            results.append({'year':yr,'ret':0,'cagr':0,'sharpe':0,'mdd':0,'n':0,'win':0,'bnh':bnh,'blocked':True}); continue
        tr_df=df[(df.index>=trs_u)&(df.index<=tre_u)]
        if len(tr_df.dropna(subset=[TARGET_COL]))<5000:
            print(f"  학습 데이터 부족"); continue
        xm=train_xgb(tr_df)
        res=backtest(df,xm,ts,te2)
        if res:
            res['year']=yr; res['blocked']=False; results.append(res)
            alpha=res['ret']-res['bnh']
            print(f"  수익:{res['ret']:.2f}% Sharpe:{res['sharpe']:.3f} MDD:{res['mdd']:.2f}% 거래:{res['n']} BnH:{res['bnh']:.1f}% 알파:{alpha:+.1f}%p")

    if results:
        dr=pd.DataFrame(results)
        print(f"\n{'='*70}\n## {ASSET} 결과 ##\n{'='*70}")
        print(f"\n{'년도':>6} | {'상태':>5} | {'수익':>8} | {'Sharpe':>7} | {'MDD':>7} | {'거래':>5} | {'승률':>6} | {'BnH':>8} | {'알파':>8}")
        print("-"*90)
        for _,r in dr.iterrows():
            alpha=r['ret']-r['bnh']
            s="차단" if r['blocked'] else "운용"
            print(f"{r['year']:>6} | {s:>5} | {r['ret']:>+7.2f}% | {r['sharpe']:>+7.3f} | {r['mdd']:>+6.2f}% | {int(r['n']):>5} | {r['win']:>6.3f} | {r['bnh']:>+7.2f}% | {alpha:>+7.2f}%")

        active = dr[~dr['blocked']]
        all_yrs = dr

        print(f"\n## 운용 연도만 ({len(active)}개) ##")
        if len(active)>0:
            print(f"  평균 수익: {active['ret'].mean():.2f}%")
            print(f"  평균 Sharpe: {active['sharpe'].mean():.3f}")
            print(f"  평균 MDD: {active['mdd'].mean():.2f}%")
            print(f"  평균 거래 수: {active['n'].mean():.1f}")
            print(f"  총 거래: {int(active['n'].sum())}")
            print(f"  흑자: {(active['ret']>0).sum()}/{len(active)}")
            print(f"  BnH 대비 알파 양수: {((active['ret']-active['bnh'])>0).sum()}/{len(active)}")

        # 복리 누적
        cum=1.0; cum_b=1.0
        for r in all_yrs['ret']: cum*=(1+r/100)
        for r in all_yrs['bnh']: cum_b*=(1+r/100)
        cagr=(cum**(1/6)-1)*100
        cagr_b=(cum_b**(1/6)-1)*100

        print(f"\n## 6년 누적 (차단 포함) ##")
        print(f"  {ASSET} 전략: {(cum-1)*100:+.1f}% (CAGR {cagr:+.2f}%)")
        print(f"  {ASSET} BnH:  {(cum_b-1)*100:+.1f}% (CAGR {cagr_b:+.2f}%)")

        print(f"\n## BTC 결과와 비교 (참고) ##")
        print(f"  BTC 전략: +39.0% (CAGR 5.64%) / MDD 평균 -8.29%")
        print(f"  BTC BnH:  +175.4% (CAGR 18.39%)")
        print(f"  BTC 운용 평균: +7.50% / 흑자 4/5 / 2022폭락 -11.69%")

        dr.to_csv(BASE_DIR/f'{ASSET}_walkforward_results.csv', index=False)
        print(f"\n저장: {ASSET}_walkforward_results.csv")