"""
# 1. config.py 수집 설정 정의 파일. 
이 파일은 real_alpha3.py와는 무관한 데이터 파이프라인 설정임. 
전략 실행전 데이터 수집 단계에서만 레버리지함. 
"""
from pathlib import Path
from datetime import datetime, timezone

# ============================================================
# 프로젝트 경로
# ============================================================
PROJECT_ROOT = Path(r"C:\Quant_Project")

DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw" # Binanace API rwa 데이터 저장 위치. 
RESAMPLED_DIR = DATA_DIR / "resampled" # 1분봉 -> 15분봉 리샘플링한 결과 데이터 공간.
ALIGNED_DIR   = DATA_DIR / "aligned" # BTC/ETH 시간 정렬 후 데이터셋.
META_DIR      = DATA_DIR / "meta" # 다운로드 로그 및 검증 리포트. 

for d in [RAW_DIR, RESAMPLED_DIR, ALIGNED_DIR, META_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# 거래소 / 종목 / 시장
# ============================================================
EXCHANGE = "binance"
MARKET   = "usdm_perpetual" # Binance USDT-M 무기한 선물 시장 기준.
# 현물이 아닌 선물 선택 이유: 펀딩비, 미결제약정, 롱&숏 비율 등 파생 데이터 수집 가능했기 때문이고, 전략의 대부분이 유동성 feature를 사용하기 때문에 선물 시장을 사용할 수 밖에 없었음. 

SYMBOLS = ["BTCUSDT", "ETHUSDT"] # BTC 기준, ETH는 out-of-sample 전략 검증용. 

FUTURES_BASE_URL = "https://fapi.binance.com"

# ============================================================
# 데이터 검증 기간
# BTC 기준: 2019-09-25 ~ (바이낸스 USDT-M 선물 상품 출시일)
# ETH 기준: 2019-11-27 ~ (바이낸스 USDT-M 선물 상품 출시일)
# ============================================================
START_DATE = datetime(2019, 9, 25, 8, 0, tzinfo=timezone.utc)
END_DATE   = datetime.now(timezone.utc) # 실행 시점 기준 최신 데이터까지 자동적으로 수집.

# ============================================================
# 타임프레임
# ============================================================
BASE_INTERVAL = "1m" # 1분봉으로 수집 후에 15분봉으로 리샘플링. 
RESAMPLE_RULES = {
    "3m":  "3min",
    "5m":  "5min",
    "10m": "10min",
    "15m": "15min", # 전략에서 사용하는 타임프레임. 
}

# ============================================================
# Binance API는 분당 WEIGHT 제한이 있음. -> rate limit 설정. 
# 데이터 수집 중 제한 초과되어 마진 설정을 수정함. 
# ============================================================
WEIGHT_SAFETY_LIMIT = 4800 # weight를 max = 6000 기준 80% 정도 사용. 
KLINES_LIMIT_PER_REQUEST = 1500
REQUEST_INTERVAL_SEC = 0.1 # 요청 간 0.1초 대기 -> 과부하 방지. 
MAX_RETRIES = 5 # 네트워크 에러뜨면 5회까지 다시 시도 셋팅.

# ============================================================
# 수집 과정에서 발생한 오류나 누락 구간들을 json 파일에 기록 저장. 
# 데이터 품질 문제 추적에 활용되었음. 
# ============================================================
DOWNLOAD_LOG = META_DIR / "download_log.json"
SANITY_REPORT = META_DIR / "sanity_report.json"
RESAMPLE_VALIDATION = META_DIR / "resample_validation.json"

if __name__ == "__main__":
    print("=" * 60)
    print("Quant Project Configuration")
    print("=" * 60)
    print(f"Project root:  {PROJECT_ROOT}")
    print(f"Data dir:      {DATA_DIR}")
    print(f"Exchange:      {EXCHANGE} ({MARKET})")
    print(f"Symbols:       {SYMBOLS}")
    print(f"Period:        {START_DATE.date()} ~ {END_DATE.date()}")
    print(f"Base TF:       {BASE_INTERVAL}")
    print(f"Resample TFs:  {list(RESAMPLE_RULES.keys())}")
    print("=" * 60)
