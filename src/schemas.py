from pydantic import BaseModel
from typing import List

class HmmResponse(BaseModel):
    timestamp: str
    symbol: str
    trend_score: int

class JournalData(BaseModel):
    entry_reason: str
    exit_reason: str
    emotion: str
    pnl: float
    hmm_score: int

class PerformanceRequest(BaseModel):
    batch_size: int
    historical_win_rate: float
    historical_avg_pnl: float
    batch_win_rate: float
    batch_avg_pnl: float
    journals: List[JournalData]