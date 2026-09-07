import os
import json
from typing import List, Dict, Any
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from schemas import PerformanceRequest
from prompts import (
    RISK_MANAGER_SYSTEM,
    TECHNICAL_ANALYST_SYSTEM,
    PSYCHOLOGY_COACH_SYSTEM,
    SYNTHESIZER_SYSTEM
)

load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL")
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.1)

HMM_TREND_THRESHOLD = 60

def calculate_hmm_regime(journals: List[Dict[str, Any]]) -> Dict[str, Any]:
    trend = [j for j in journals if j.get("hmm_score", 50) > HMM_TREND_THRESHOLD]
    chop = [j for j in journals if j.get("hmm_score", 50) <= HMM_TREND_THRESHOLD]
    
    def win_rate(trades):
        if not trades: return 0.0
        return round((sum(1 for t in trades if t.get("pnl", 0) > 0) / len(trades)) * 100, 2)

    return {
        "trend_market": {"trade_count": len(trend), "win_rate": win_rate(trend)},
        "chop_market": {"trade_count": len(chop), "win_rate": win_rate(chop)}
    }

class AgentState(TypedDict):
    journals: List[Dict[str, Any]]
    regime_data: Dict[str, Any]
    request_meta: Dict[str, Any]
    risk_analysis: str
    technical_analysis: str
    psychology_analysis: str
    final_report: str

# 1. 병렬 실행 전문 분석 노드
async def risk_manager_node(state: AgentState):
    meta = state["request_meta"]
    sys_prompt = RISK_MANAGER_SYSTEM.format(
        hist_win=meta['historical_win_rate'], hist_pnl=meta['historical_avg_pnl'],
        batch_win=meta['batch_win_rate'], batch_pnl=meta['batch_avg_pnl']
    )
    user_msg = f"매매 저널 데이터:\n{json.dumps(state['journals'], ensure_ascii=False)}"
    res = await llm.ainvoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_msg)])
    return {"risk_analysis": res.content}

async def technical_analyst_node(state: AgentState):
    regime = state["regime_data"]
    sys_prompt = TECHNICAL_ANALYST_SYSTEM.format(
        hmm_threshold=HMM_TREND_THRESHOLD,
        trend_cnt=regime['trend_market']['trade_count'], trend_win=regime['trend_market']['win_rate'],
        chop_cnt=regime['chop_market']['trade_count'], chop_win=regime['chop_market']['win_rate']
    )
    user_msg = f"매매 저널 데이터:\n{json.dumps(state['journals'], ensure_ascii=False)}"
    res = await llm.ainvoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_msg)])
    return {"technical_analysis": res.content}

async def psychology_coach_node(state: AgentState):
    user_msg = f"매매 저널 데이터:\n{json.dumps(state['journals'], ensure_ascii=False)}"
    res = await llm.ainvoke([SystemMessage(content=PSYCHOLOGY_COACH_SYSTEM), HumanMessage(content=user_msg)])
    return {"psychology_analysis": res.content}

# 2. 취합 편집 노드 (Fan-In)
async def synthesizer_node(state: AgentState):
    expert_inputs = (
        f"[리스크 분석]\n{state['risk_analysis']}\n\n"
        f"[기술적 분석]\n{state['technical_analysis']}\n\n"
        f"[심리 코칭 분석]\n{state['psychology_analysis']}"
    )
    res = await llm.ainvoke([SystemMessage(content=SYNTHESIZER_SYSTEM), HumanMessage(content=expert_inputs)])
    return {"final_report": res.content}

# 3. LangGraph Workflow 구성
workflow = StateGraph(AgentState)
workflow.add_node("risk_manager", risk_manager_node)
workflow.add_node("technical_analyst", technical_analyst_node)
workflow.add_node("psychology_coach", psychology_coach_node)
workflow.add_node("synthesizer", synthesizer_node)

# Fan-Out
workflow.add_edge(START, "risk_manager")
workflow.add_edge(START, "technical_analyst")
workflow.add_edge(START, "psychology_coach")

# Fan-In
workflow.add_edge("risk_manager", "synthesizer")
workflow.add_edge("technical_analyst", "synthesizer")
workflow.add_edge("psychology_coach", "synthesizer")
workflow.add_edge("synthesizer", END)

agent_engine = workflow.compile()

async def analyze_trading_performance(request: PerformanceRequest) -> str:
    if not request.journals:
        return "분석할 매매 기록이 존재하지 않습니다."
    
    journals = [j.model_dump() for j in request.journals]
    initial_state: AgentState = {
        "journals": journals,
        "regime_data": calculate_hmm_regime(journals),
        "request_meta": {
            "historical_win_rate": request.historical_win_rate,
            "historical_avg_pnl": request.historical_avg_pnl,
            "batch_win_rate": request.batch_win_rate,
            "batch_avg_pnl": request.batch_avg_pnl
        },
        "risk_analysis": "", "technical_analysis": "", "psychology_analysis": "", "final_report": ""
    }
    
    output_state = await agent_engine.ainvoke(initial_state)
    return output_state["final_report"]