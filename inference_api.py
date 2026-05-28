from fastapi import FastAPI
from pydantic import BaseModel
import datetime
import uvicorn

app = FastAPI()

# 응답 데이터 형식
class HmmResponse(BaseModel):
    timestamp: str
    symbol: str
    trend_score: int

# 4시간마다 호출될 점수 계산 함수
@app.get("/api/v1/hmm/predict", response_model=HmmResponse)
def get_latest_hmm_score(symbol: str = "BTC/USDT"):
    # ------------------------------------------------------------
    # 추후 실제 로직이 들어갈 자리
    # ------------------------------------------------------------
    
    # 테스트용 Mock
    now = datetime.datetime.now()
    current_candle_time = now.replace(minute=0, second=0, microsecond=0)
    
    # 테스트용 추세 점수
    dummy_trend_score = 75

    return HmmResponse(
        timestamp=current_candle_time.isoformat(),
        symbol=symbol,
        trend_score=dummy_trend_score
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)