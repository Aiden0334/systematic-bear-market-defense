# 4. make_targets.py 
# 새 타겟 생성 후에 parquet에 저장. 
# raw feature 데이터에 미래 가격 방향 라벨(target)추가 후 저장.
# 이 파일 실행해야 walkforward_btc0.py 실행 가능. 
import pandas as pd, numpy as np
from pathlib import Path

#------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR/'data'/'processed'/'BTCUSDT_features_15m.parquet'
#------------------------------------------

print(f"데이터 로드: {DATA_PATH}")
df = pd.read_parquet(DATA_PATH)
print(f"원본 shape: {df.shape}")

# 기존 15분봉 target 분포 확인. - 이후 새 타겟과 비교 기준으로 활용함. 
print(f"기존 target_dir_strong_15m 분포:")
print(df['target_dir_strong_15m'].value_counts(normalize=True).sort_index())

c = df['kl_close']

# 타겟 컬럼 생성. 
# 1시간 horizon (4봉 후) ±0.3%
# 15분 * 4 = 60분 (1시간), 1시간 기준 의미 있는 방향성 움직임 기준 ±0.3% 임계값
fwd_1h = c.pct_change(4).shift(-4)
df['target_dir_strong_1h'] = np.where(fwd_1h > 0.003, 1.0,
                              np.where(fwd_1h < -0.003, -1.0, 0.0))
#------------------------------------------
# 검증 결과 lag-1 자기상관 0.59로 학습 가능성 확인하였으나, 4시간봉보다 낮아서 최종적으로 4시간봉 target 채택함.
# 마지막 4봉은 미래 데이터 없음. -> look ahead 방지하기 위해 NaN 처리함. 
df.loc[df.index[-4:], 'target_dir_strong_1h'] = np.nan

#------------------------------------------
# 4시간 horizon (16봉 후) ±0.8%
# - 너무 작으면 (±-.3%)수수료 대비 수익 미미함.
# - 너무 크면 (±1.5%) 라벨 적어져서 학습 데이터 부족해짐.
# 자기상관 0.81로 15분봉(-0.05), 1시간봉 (0.59) 대비 압도적으로 높아서 4시간봉 채택!
fwd_4h = c.pct_change(16).shift(-16)
df['target_dir_strong_4h'] = np.where(fwd_4h > 0.008, 1.0,
                              np.where(fwd_4h < -0.008, -1.0, 0.0))
df.loc[df.index[-16:], 'target_dir_strong_4h'] = np.nan # 마지막 16봉은 미래 데이터 없음. -> look-ahead 방지 위해 NaN 처리.
#------------------------------------------

# 분포 확인
print(f"\n=== 새 target_dir_strong_1h 분포 ===")
print(df['target_dir_strong_1h'].value_counts(normalize=True, dropna=False).sort_index())
print(f"\n=== 새 target_dir_strong_4h 분포 ===")
print(df['target_dir_strong_4h'].value_counts(normalize=True, dropna=False).sort_index())

#------------------------------------------
# 라벨별 실제 수익률 검증
# 분포가 극단적으로 치우치는 것을 방지하기 위해서 -1, 0, +1 비율로 적절한지 확인. 
# mean이 양수면 라벨링이 올바르게 됐다는 증거 -> 양수 확인. 
# 검증 결과: +1 평균 +1.82%,
#           -1 평균 -1.82% 로 명확한 분리를 확인함. 

# A. 1시간봉 기준 실제 수익률 검증.
print(f"\n=== target_dir_strong_1h: 라벨별 4봉 후 수익률(%) ===")
fwd_1h_pct = fwd_1h * 100
tg = df['target_dir_strong_1h']
print(tg.groupby(tg).apply(lambda s: pd.Series({
    'mean': fwd_1h_pct.loc[s.index].mean(),
    'median': fwd_1h_pct.loc[s.index].median(),
    'std': fwd_1h_pct.loc[s.index].std(),
    'n': len(s)
})).unstack())

# B. 4시간봉 기준 실제 수익률 검증. 
print(f"\n=== target_dir_strong_4h: 라벨별 16봉 후 수익률(%) ===")
fwd_4h_pct = fwd_4h * 100
tg = df['target_dir_strong_4h']
print(tg.groupby(tg).apply(lambda s: pd.Series({
    'mean': fwd_4h_pct.loc[s.index].mean(),
    'median': fwd_4h_pct.loc[s.index].median(),
    'std': fwd_4h_pct.loc[s.index].std(),
    'n': len(s)
})).unstack())
#------------------------------------------


#------------------------------------------
# 자기상관 분석 - TARGET의 예측 가능성 척도. 
# 자기 상관이 높을수록 모멘텀 신호 존재하기에 학습 가능성이 높음. 
# 하지만 자기 상관이 0에 가까우면 랜덤 워크 -> 학습 데이터로 레버리지 못함. 
print(f"\n=== 자기상관 비교 ===")
for col in ['target_dir_strong_15m', 'target_dir_strong_1h', 'target_dir_strong_4h']:
    print(f"  {col}:")
    for lag in [1, 4, 16]:
        print(f"    lag {lag:2d}: {df[col].autocorr(lag):.4f}")

"""결과치 해석: 
a. 15분봉 기준 Autocorrelation = -0.05 (노이즈 심함, 버림)
b. 1시간봉 기준 Autocorrelation = 0.59 (레버리지 가능)
c. 4시간봉 기준 Autocorrelation = 0.81 (Strong Momentum - 최종적으로 사용.)
"""

#------------------------------------------
# walkforward_btc0.py (백테스트 파일)에서 target_dir_strong_4h 컬럼을 바로 사용할 것임.
# 저장
df.to_parquet(DATA_PATH)
print(f"\n저장 완료: {DATA_PATH}")
print(f"최종 shape: {df.shape} (컬럼 2개 추가)")
#------------------------------------------
