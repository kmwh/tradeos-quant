import datetime
from fastapi import FastAPI
import uvicorn

from schemas import HmmResponse, PerformanceRequest
from hmm_service import calculate_realtime_hmm_score
from ai_service import analyze_trading_performance

app = FastAPI(title="TradeOS Quant API", version="1.0.0")

@app.get("/api/v1/hmm/predict", response_model=HmmResponse)
def get_latest_hmm_score(symbol: str = "BTC/USDT"):
    now = datetime.datetime.now()
    
    try:
        score = calculate_realtime_hmm_score(symbol)
    except Exception as e:
        print(f"HMM Score Calculation Failed: {e}")
        score = 50 
        
    return HmmResponse(
        timestamp=now.isoformat(),
        symbol=symbol,
        trend_score=score
    )

@app.post("/api/v1/ai/tendency")
def analyze_performance(request: PerformanceRequest):
    try:
        summary = analyze_trading_performance(request)
        return {"summary_text": summary}
    except Exception as e:
        print(f"AI Analysis Failed: {e}")
        return {"summary_text": "평가 중 오류가 발생했습니다."}

if __name__ == "__main__":
    # 서버 실행 시 이제 inference_api 대신 main.py를 실행합니다.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)