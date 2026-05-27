""" 
# Systematic Bear Market Defense via XGBOOST Model
# real_alpha3.py (main_6y.py) - 6 year Walk-Forward Backtest
# Learning Data Revised.

# Key Strategy
  - Catch a short-term rebound for Liquidity Event (e.g., Liquidation, Volatility Explosion)
  - Intentionally deactivated in quiet bull-market.
  - Activated in bear markets as an event-based mean reversion strategy.
    (Risk Management Modeling) 
"""

# Package Setting 
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from xgboost import XGBClassifier

#------------------------------------------------
# Feature 목록 작성.
# feature_cols.json 에서 사전으로 선정되어 있떤 100개 features 로딩.
# features engineering 파트에서 190개 이상의 features들을 생성했지만 데이터 품질 누락으로 인해서 SMC (Smart Money Concept의 유동성 data) feature 부분들을 제거함.
# 제거 이후 MDD 대폭 개선됨을 확인함.
# regime3가 add_features() 내부에서 kl_close 기반으로 계산되니 json에 없어도 본 파일에 추가시킴.
#------------------------------------------------

BASE_DIR     = Path(__file__).parent # 어느 환경에서 실행해도 경로가 깨지지 않게하기 위함.
feature_cols = json.load(open(BASE_DIR/'models'/'feature_cols.json'))
feature_cols = [c for c in feature_cols if not any(c.startswith(p) for p in ['smc_','upper_wick','lower_wick','wick_'])]
if 'regime3' not in feature_cols: feature_cols.append('regime3')

#------------------------------------------------
# Target 정의:
# 본래 1분봉 데이터였으나, 데이터 검증 시에 노이즈가 많아 15분봉 타겟으로 변경. 
# 노이즈와 과적합을 피하기 위해서 15분봉 타겟에서 4시간봉 타겟으로 변경. 
# 15분봉 기준 16봉(4시간 기준) 이후 수익률이 +-0.8% 이상이면 방향 라벨을 부여함.
# 15분봉 타겟(lag-1 자기상관 -0.05)은 노이즈 수준이었지만, 4시간봉 target(lag-1 자기상관 0.81)은 학습 가능한 모멘텀 신호 확인 후에 채택함.
#------------------------------------------------
TARGET_COL = 'target_dir_strong_4h'
MAX_HOLD, COOLDOWN, FEE_RATE = 64, 16, 0.0005 # 1분 64봉, 15분 16봉, 보수적인 펀딩비.
LONG_P, SHORT_P, SL_ATR = 0.50, 0.50, 2.0 
# xgboost가 예측한 회귀 상승 확률 P가 40% 넘을때만 진입. 그리고 손절가는 진입가 기준 ATR 2배 아래에 설정.
# 시장 변동성이 클수록 손절 거리도 넓어지는 동적 손절 구조.
ENABLE_SHORT = False # 숏 임계값이었으나 사용하지 않음.
SEED = 42 # (10개 시드로 견고성 검증 완료.) -> 전체 시드 CAGR 양수, 평균 11%, std 3.9%로 시드 의존성 낮음을 확인함.
MIN_TRAIN_DAYS = 700   # 거의 2년 (WalkForward 시 학습 데이터가 700일 미만이면 거래 차단.)
                       # 불충분한 학습 데이터로 인한 과적합과 잘못된 신호 방지를 위해 거래 차단. 
LABEL_MAP = {-1.0:0, 0.0:1, 1.0:2} # xgboost 모델은 정수 라벨만 받기에 float 라벨을 정수로 변환. (-1.0 하락, 0 횡보, 1, 1.0 상승 -> 2)
DATA_PATH = BASE_DIR/'data'/'processed'/'BTCUSDT_features_15m.parquet'
print(f"[6년 Walk-Forward + 안전 장치] {TARGET_COL} | LongOnly:{not ENABLE_SHORT} | seed:{SEED}")
print(f"  안전 장치: 학습 데이터 < {MIN_TRAIN_DAYS}일 시 거래 차단")


#------------------------------------------------
# 원본 데이터 보존을 위해서 복사본 생성.
# 원본 건드리면 walk-forward roop에서 다음 구간 데이터가 오염될 수 있음.
# 단기/중기/장기 이동평균 계산.
#------------------------------------------------
def add_features(df):
    df = df.copy(); c = df['kl_close']
    df['ma_short']=c.rolling(96).mean(); df['ma_mid']=c.rolling(480).mean(); df['ma_long']=c.rolling(19200).mean()
    # 15분봉 기준: 96봉 = 1일, 480봉 = 5일이며, 19200봉은 200일임. 
    df['regime3']=np.where((df['ma_short']>df['ma_mid'])&(df['ma_mid']>df['ma_long']),2,np.where((df['ma_short']<df['ma_mid'])&(df['ma_mid']<df['ma_long']),0,1))
    # -------------------------------------------
    # regime 변화를 구분하기 위해서 세 MA 정렬 상태 만듦. 
    # 시장 추세 구분 
    """ 
    0 = 하락 추세 (역배열) ma_short < ma_mid <ma_long
    1 = 횡보장 (정배열 X, 역배열 X)
    2 = 상승장 (정배열) # ma_short > ma_mid > ma_long

    Regime 구분에 대한 limitation 있었음. 
    Regime 판별 개선을 위해 HMM (Hidden Markov Model)을 add 해보았지만 성능 저하 확인.
    단순 ma 정렬 방식이 더 안정적임을 walk-forward로 검증함. 
    """
    h,l,v=df['kl_high'].values,df['kl_low'].values,df['kl_volume'].values
    hlc3=(h+l+c.values)/3; n,prd=len(df),20 # 고가+저가+종가/3, VWAP 기준이기에 단순 종가보다 노이즈 적음.
    phi,pli=np.zeros(n,int),np.zeros(n,int) # phi란, 직전 20봉 내에 가장 높은 고가의 인덱스.
    for i in range(prd,n):
        phi[i]=i if h[i]==h[max(0,i-prd):i+1].max() else phi[i-1] # 최근 스윙 고점이 저점보다 나중이면 상승(1), else: 하락(-1)
        pli[i]=i if l[i]==l[max(0,i-prd):i+1].min() else pli[i-1] 
    d=np.where(phi>pli,1,-1); vw=np.full(n,np.nan); cpv=cv=pd_=0 # 방향 바뀌는 시점이 유동성 이벤트 발생 구간과 일치하는 경향이 있음.
    for i in range(n): # 스윙 방향이 바뀌는 시점. -> vwap 초기화.
        if d[i]!=pd_:
            ai=phi[i] if d[i]>0 else pli[i]; cpv,cv,pd_=hlc3[ai]*v[ai],v[ai],d[i]
        else: cpv+=hlc3[i]*v[i]; cv+=v[i]
        if cv>0: vw[i]=cpv/cv
    df['swing_vwap']=vw; df['swing_vwap_dist']=(c.values-vw)/(c.values+1e-12) # swing_vwap_dist : 현재 가격과 swing vwap의 거리.
    df['swing_vwap_dir']=d.astype(float); df['swing_vwap_above']=(c.values>vw).astype(float)
    return df

    """ 
    # swing VWAP 계산을 위해서 NUMPY ARRAY로 변환함. - 루프 연산 속도 빠르기 때문. 
    # vw 정의: 각 봉의 swing vwap 값이고, cpv란, 누적 가격*거래량. 
    # 양수가 나온다면 vwap 위(과매수), 음수면 vwap 아래(과매도) -> 평균 회귀 신호 생성될거임.
    # swing_vwap_above로 현재 가격이 swing vwap 위에 있는 가격 위치를 이진으로 단순화해서 모델 학습 용이하게 함.
    """

#------------------------------------------------
# 시장 변동성 지표 ATR (Average True Range) - 손절가 계산에 용이, 변동성 클수록 손절 거리 넓어지는 동적 손절 구조.
# 1. 당일 변동폭 (고가 - 저가)
# 2. 갭 상승 |고가 - 전일 종가|
# 3. 갭 하락 |저가 - 전일 종가|
#------------------------------------------------
def compute_atr(df,length=14):
    h,l,c=df['kl_high'],df['kl_low'],df['kl_close']
    tr=pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/length,adjust=False,min_periods=length).mean()
# 단순 ema 대신 ewm (지수 이동 평균) leverage 함.
# length 시장 표준 14로 설정. min_periods 14 미만 구간은 NaN으로 처리.


# 과적합 방지를 위한 regularization 설정.
def train_xgb(tr_df):
    tr=tr_df.dropna(subset=[TARGET_COL]) # NaN이 있는 행은 학습에서 제외시킴. 
    # 불완전한 데이터로 인한 모델 오염 방지를 위함.
    # xgboost 모델 파라미터 설정: 트리 100개, 트리 깊이 4로 제한, 과적합 방지를 위해 낮은 학습률로 천천히 수렴.
    mdl=XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.05,subsample=0.8, # 0.8 = 매 트리마다 80% 데이터만 사용함. (과적합 방지용.)
                      colsample_bytree=0.8,min_child_weight=10,gamma=1, # 매 트리마다 80% feature만 사용 - 과적합 방지.
                      reg_alpha=0.1,reg_lambda=1.0,random_state=SEED,n_jobs=-1,tree_method='hist')
    mdl.fit(tr[feature_cols],tr[TARGET_COL].map(LABEL_MAP),verbose=False) # target 라벨을 LABEL_MAP으로 정수 변환 후에 학습시키는 방향.
    print(f"  XGB 학습: {len(tr):,}개"); return mdl


def calc_bnh(df, start, end): # Buy and Hold 수익률 계산. 
    ts,te=pd.Timestamp(start,tz='UTC'),pd.Timestamp(end,tz='UTC')
    tdf=df[(df.index>=ts)&(df.index<=te)]
    if len(tdf)<2: return 0.0 # data가 2개 미만이면 수익률 계산 불가. -> 0으로 return.
    return (tdf['kl_close'].iloc[-1]/tdf['kl_close'].iloc[0]-1)*100
    # 전략이 단순 보유 대비 얼마나 위험을 줄였는지를 평가. 


def backtest(df,xgb,start,end,cap=10000):
    ts,te=pd.Timestamp(start,tz='UTC'),pd.Timestamp(end,tz='UTC')
    buf=df[(df.index>=ts-pd.Timedelta(days=210))&(df.index<=te)].copy()
    # 테스트 시작일 210일 전 데이터부터 포함시켜야 충분히 안정적으로 ATR 계산. (Length = 14)
    buf['atr']=compute_atr(buf); tdf=buf[buf.index>=ts].copy()
    if len(tdf)<30: return None # 통계 샘플 데이터 30봉 미만 의미 없으니 건너뜀. (CLT 적용 위함.)
    proba=xgb.predict_proba(tdf[feature_cols])
    up_p,dn_p=proba[:,2],proba[:,0] # 상승 확률, 하락 확률. proba[:,1] = 횡보 확률.
    print(f"  proba[up] 50%={np.percentile(up_p,50):.3f} 90%={np.percentile(up_p,90):.3f} max={up_p.max():.3f}")
    print(f"  proba[dn] 50%={np.percentile(dn_p,50):.3f} 90%={np.percentile(dn_p,90):.3f} max={dn_p.max():.3f}")
    # 50% 너무 높으면 과적합, 너무 낮으면 신호 부족 의심해야함. 
    ls=(up_p>LONG_P).astype(int)
    if ENABLE_SHORT:
        ss=(dn_p>SHORT_P).astype(int)
    else:
        ss=np.zeros_like(ls)
    print(f"  long signal:{ls.sum()} short signal:{ss.sum()}") # 롱, 숏 시그널 거래 수 출력.
    regime3_labels = {0:'하락', 1:'횡보', 2:'상승'}
    regime3_dist = tdf['regime3'].value_counts().sort_index().to_dict()
    print(f" 전체 구간 regime3: { {regime3_labels[k]:v for k,v in regime3_dist.items()} }")

    cl,at=tdf['kl_close'].values,tdf['atr'].values
    pos,ep,ei,lei,sl,trades=0,0.,0,-COOLDOWN,0.,[] # 현재 포지션 (0,1, -1로 구분), ep = entry price, ei = entry index, lei = last exit index (쿨다운으로 초기화 - 첫 봉부터 계산)
    sl_hits=0
    entry_regimes = []
    for i in range(1,len(cl)):
        if np.isnan(at[i]) or np.isnan(cl[i]): continue 
        if pos==1 and cl[i]<=sl: # 롱 손절.
            pnl=(cl[i]-ep)/ep-FEE_RATE; cap*=(1+pnl); trades.append({'t':'long','pnl':pnl,'exit':'sl'}); pos=0; lei=i; sl_hits+=1; continue
        elif pos==-1 and cl[i]>=sl: # 숏 손절.
            pnl=(ep-cl[i])/ep-FEE_RATE; cap*=(1+pnl); trades.append({'t':'short','pnl':pnl,'exit':'sl'}); pos=0; lei=i; sl_hits+=1; continue
        if pos==1 and (i-ei)>=MAX_HOLD: 
            pnl=(cl[i]-ep)/ep-FEE_RATE; cap*=(1+pnl); trades.append({'t':'long','pnl':pnl,'exit':'t'}); pos=0; lei=i
        elif pos==-1 and (i-ei)>=MAX_HOLD:
            pnl=(ep-cl[i])/ep-FEE_RATE; cap*=(1+pnl); trades.append({'t':'short','pnl':pnl,'exit':'t'}); pos=0; lei=i
        if pos==0 and (i-lei)>=COOLDOWN: # 포지션 없음.
            if ls[i]:
                pos, ep, ei, sl = 1, cl[i], i, cl[i]-SL_ATR*at[i]
                entry_regimes.append(int(tdf['regime3'].iloc[i]))
            elif ss[i]: pos, ep, ei, sl = -1, cl[i], i, cl[i]+SL_ATR*at[i]

    bnh=(tdf['kl_close'].iloc[-1]/tdf['kl_close'].iloc[0]-1)*100
    if not trades:
        print("  거래없음")
        return {'ret':0,'sharpe':0,'mdd':0,'n':0,'win':0,'bnh':bnh,'cagr':0,'pf':0}
    # 거래 한 건도 없으면 강세장에서 유동성 이벤트 없을때 정상적으로 발생. - edge가 없는 구간에서 의도적으로 쉬는 곳.
    td=pd.DataFrame(trades); cv=pd.Series([10000])
    # 거래 기록 dataframe으로 변환 후에 자본 곡선 cv로 모아줌.
    for p in td['pnl']: cv=pd.concat([cv,pd.Series([cv.iloc[-1]*(1+p)])])
    cv=cv.reset_index(drop=True); r=td['pnl']; days=(tdf.index.max()-tdf.index.min()).days; yr=max(days/365,0.01)
    lt=td[td['t']=='long']; st=td[td['t']=='short']
    print(f"  손절 히트:{sl_hits}/{len(td)} ({sl_hits/len(td):.1%})")
    if entry_regimes:
        er = pd.Series(entry_regimes).value_counts().sort_index().to_dict()
        er_labeled = {regime3_labels[k]: v for k, v in er.items()}
        print(f" 진입 시에 regime3: {er_labeled}")
    if len(lt)>0: print(f"  롱:{len(lt)}개 승률:{(lt['pnl']>0).mean():.3f} 평균:{lt['pnl'].mean()*100:.3f}%")
    if len(st)>0: print(f"  숏:{len(st)}개 승률:{(st['pnl']>0).mean():.3f} 평균:{st['pnl'].mean()*100:.3f}%")
    return {'ret':(cap/10000-1)*100,'cagr':((cap/10000)**(1/yr)-1)*100,
            'sharpe':r.mean()/r.std()*np.sqrt(len(r)/yr) if r.std()>0 else 0,
            'mdd':((cv-cv.cummax())/cv.cummax()).min()*100,'win':(r>0).mean(),
            'pf':r[r>0].sum()/r[r<0].abs().sum() if r[r<0].abs().sum()>0 else 0,
            'n':len(td),'bnh':bnh}
#------------------------------------------------
# ret: 테스트 기간 총 수익률.
# cagr: 연환산 수익률 
# sharpe ratio (위험 대비 수익) - 252일 기준 연환산.
# mdd (max drawdown) - 최대 낙폭 
# pf: profit factor - 총 수익 / 총 손실 (1.0 이상이면 수익임.)
#------------------------------------------------


#------------------------------------------------
# 데이터 로드 및 feature 추가. 
# Binance USDT.P 선물 15분봉 데이터 (2019-09-25 to 2026-05-20)
if __name__=='__main__':
    print("데이터 로드 중..."); df=pd.read_parquet(DATA_PATH)
    df=add_features(df)
    print(f"BTC:{df.shape} 범위:{df.index.min()} ~ {df.index.max()}\n{'='*70}")

    wf=[('2021', '2019-09-25', '2020-12-31', '2021-01-01', '2021-12-31'),
        ('2022', '2020-01-01', '2021-12-31', '2022-01-01', '2022-12-31'),
        ('2023', '2021-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
        ('2024', '2022-01-01', '2023-12-31', '2024-01-01', '2024-12-31'),
        ('2025', '2023-01-01', '2024-12-31', '2025-01-01', '2025-12-31'),
        ('2026', '2024-01-01', '2025-12-31', '2026-01-01', '2026-05-20')]
        # 마지막 연도 아직 데이터 부족. (하지만 방어력 검증 확인. bnh: -11% 대비 전략 수익 1.65%)
        # rolling 2년 학습 후 그 다음 해 1년 테스트 구조 (look-ahead 방지)
        # expanding window가 아닌 rolling window를 택한 이유: 가상화폐 시장은 분포 shift가 잦기 때문에, 오래된 데이터가 오히려 노이즈가 될 수 있음.
        # 그래서 2021-2025년 5개 구간에서만 평가. 2026년은 5개월 데이터라 아직 미완성.
    results=[]
    for yr,trs,tre,ts,te2 in wf:
        print(f"\n[{yr}] 학습:{trs}~{tre} → 테스트:{ts}~{te2}")
        trs_utc,tre_utc=pd.Timestamp(trs,tz='UTC'),pd.Timestamp(tre,tz='UTC')
        train_days=(tre_utc-trs_utc).days
        print(f"  학습 기간: {train_days}일") 

        if train_days < MIN_TRAIN_DAYS: 
            continue # 학습 데이터 700 미만시 거래 차단.
        tr_df = df[(df.index>=trs_utc)&(df.index<=tre_utc)]
        if len(tr_df.dropna(subset=[TARGET_COL])) < 5000:
            print(f"  학습 데이터 부족({len(tr_df)}개), 건너뜀")
            continue # TARGET NaN 포함 행이 5000개 미만이면 학습 불가로 판단함.

        # 학습 & 백테스트 실행.
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

        # 거래 차단 연도 제외한 실제 운용 연도만 따로 집계함. 
        # 차단 연도 포함시 평균 왜곡되기 때문임.
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

        print(f"\n## 전체 6년 (차단 포함) ##") # 차단 연도도 포함해서 전체 성과에 포함시킴.
        print(f"평균 수익: {all_yrs['ret'].mean():.2f}%")
        print(f"흑자 연도: {(all_yrs['ret']>0).sum()}/{len(all_yrs)}")
        print(f"손실 연도: {(all_yrs['ret']<0).sum()}/{len(all_yrs)}")
        print(f"차단(0% 수익): {all_yrs['blocked'].sum()}/{len(all_yrs)}")

        # 복리 누적 수익 계산.
        # 단순 수익보단 복리로 계산.
        cum_ret = 1.0
        for r in all_yrs['ret']:
            cum_ret *= (1 + r/100)
        cum_bnh = 1.0
        for r in all_yrs['bnh']:
            cum_bnh *= (1 + r/100)
        print(f"\n## 복리 누적 수익 (6년) ##")
        print(f"전략: {(cum_ret-1)*100:+.1f}% (초기 1.0 → 최종 {cum_ret:.2f})")
        print(f"BnH:  {(cum_bnh-1)*100:+.1f}% (초기 1.0 → 최종 {cum_bnh:.2f})")

        # 6년 walkforward 백테스트 결과 저장.
        dr.to_csv(BASE_DIR/'6y_walkforward_safe_results.csv', index=False)
        print(f"\n저장: {BASE_DIR/'6y_walkforward_safe_results.csv'}")