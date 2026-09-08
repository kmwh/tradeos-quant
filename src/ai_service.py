import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from schemas import PerformanceRequest
from prompts import SINGLE_PROMPT_SYSTEM

load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.1)

HMM_TREND_THRESHOLD = 60

def calculate_hmm_regime(journals: List[Dict[str, Any]]) -> Dict[str, Any]:
    trend = [j for j in journals if j.get("hmm_score", 50) > HMM_TREND_THRESHOLD]
    chop = [j for j in journals if j.get("hmm_score", 50) <= HMM_TREND_THRESHOLD]
    
    def win_rate(trades):
        if not trades:
            return 0.0
        return round((sum(1 for t in trades if t.get("pnl", 0) > 0) / len(trades)) * 100, 2)

    return {
        "trend_market": {"trade_count": len(trend), "win_rate": win_rate(trend)},
        "chop_market": {"trade_count": len(chop), "win_rate": win_rate(chop)}
    }

def to_clean_text(content: Any) -> str:
    """Gemini 응답 블록 리스트를 순수 마크다운 문자열로 정제"""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
            elif hasattr(item, "text"):
                parts.append(item.text)
        return "\n".join(parts)
    return str(content)

async def analyze_trading_performance(request: PerformanceRequest) -> str:
    """
    트레이더의 성과 메타데이터와 개별 매매 저널을 바탕으로
    단일 프롬프트를 통해 통합 성과 분석 피드백 리포트를 생성합니다.
    """
    if not request.journals:
        return "분석할 매매 기록이 존재하지 않습니다."
    
    journals = [j.model_dump() for j in request.journals]
    regime = calculate_hmm_regime(journals)
    
    sys_prompt = SINGLE_PROMPT_SYSTEM.format(
        hist_win=request.historical_win_rate,
        hist_pnl=request.historical_avg_pnl,
        batch_win=request.batch_win_rate,
        batch_pnl=request.batch_avg_pnl,
        hmm_threshold=HMM_TREND_THRESHOLD,
        trend_cnt=regime["trend_market"]["trade_count"],
        trend_win=regime["trend_market"]["win_rate"],
        chop_cnt=regime["chop_market"]["trade_count"],
        chop_win=regime["chop_market"]["win_rate"]
    )
    user_msg = f"매매 기록 데이터:\n{json.dumps(journals, ensure_ascii=False)}"
    
    res = await llm.ainvoke([
        SystemMessage(content=sys_prompt),
        HumanMessage(content=user_msg)
    ])
    
    return to_clean_text(res.content)