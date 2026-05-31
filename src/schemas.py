from pydantic import BaseModel
from typing import List

class HmmResponse(BaseModel):
    timestamp: str
    symbol: str
    trend_score: int

class JournalData(BaseModel):
    ticker: str
    position: str
    leverage: float
    entry_reason: str
    exit_reason: str
    emotion: str
    pnl: float
    roi: float
    duration_seconds: int
    hmm_score: int

class PerformanceRequest(BaseModel):
    batch_size: int
    historical_win_rate: float
    historical_avg_pnl: float
    batch_win_rate: float
    batch_avg_pnl: float
    journals: List[JournalData]