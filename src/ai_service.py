import json
from typing import List, Dict, Any
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from schemas import PerformanceRequest


# LangGraph State 구조 정의
class AgentState(TypedDict):
    journals: List[Dict[str, Any]]
    regime_data: Dict[str, Any]
    request_meta: Dict[str, Any]
    risk_analysis: str
    technical_analysis: str
    psychology_analysis: str
    final_report: str


# 전역 LLM 인스턴스
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)


# 계산 함수
def calc_win_rate(trades: List[Dict[str, Any]]):
    if not trades: return 0.0
    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    return round((wins / len(trades)) * 100, 2)

def calculate_hmm_regime(journals: List[Dict[str, Any]]):
    trend_trades = [j for j in journals if j.get("hmm_score", 50) > 60]
    chop_trades = [j for j in journals if j.get("hmm_score", 50) <= 60]
    return {
        "trend_market": {"trade_count": len(trend_trades), "win_rate": calc_win_rate(trend_trades)},
        "chop_market": {"trade_count": len(chop_trades), "win_rate": calc_win_rate(chop_trades)}
    }


# LangGraph 전문가 노드 정의
def risk_manager_node(state: AgentState):
    """수치적 리스크, 레버리지 및 자산 변동성 분석"""
    meta = state["request_meta"]
    
    system_prompt = f"""당신은 퀀트 펀드의 수석 리스크 관리자입니다.
제공되는 과거 지표와 현재 주기의 손익 데이터를 비교하여 트레이더의 리스크 관리 능력을 데이터 기반으로 서술하세요.

[과거 누적 성과] 승률: {meta['historical_win_rate']}%, 평균 수익금: {meta['historical_avg_pnl']} USDT
[현재 주기 성과] 승률: {meta['batch_win_rate']}%, 평균 수익금: {meta['batch_avg_pnl']} USDT

특히 매매 기록 데이터에 포함된 수익률 분포와 레버리지 배율을 밀접하게 분석하여, 무리한 고배율 사용 여부나 손익비 관리의 객관적 추이를 진단하세요."""

    user_content = f"현재 주기 리스크 분석 대상 데이터:\n{json.dumps(state['journals'], ensure_ascii=False)}"
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_content)])
    
    return {"risk_analysis": response.content}


def technical_analyst_node(state: AgentState):
    """종목, 포지션 방향성 및 시장 국면 기술적 분석"""
    regime = state["regime_data"]
    
    system_prompt = f"""당신은 수석 기술적 분석가입니다.
HMM 기반 시장 국면 통계와 매매 기록의 진입/청산 근거를 바탕으로 사용자의 기술적 매매 성향을 분석하세요.

[시장 국면(HMM) 통계]
- 추세장(HMM > 60): 매매 {regime['trend_market']['trade_count']}회, 승률: {regime['trend_market']['win_rate']}%
- 비추세장(HMM <= 60): 매매 {regime['chop_market']['trade_count']}회, 승률: {regime['chop_market']['win_rate']}%

종목별 특성과 매매 방향(position: LONG/SHORT)이 시장 국면(hmm_score)과 어떻게 맞아떨어졌는지 대조하고, 트레이더가 진입/청산 근거(SMC, ICT, 엘리어트 파동, 피보나치 등)의 기술적 원칙을 올바르게 고수했는지 철저히 분석하세요."""

    user_content = f"기술적 분석 대상 데이터:\n{json.dumps(state['journals'], ensure_ascii=False)}"
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_content)])
    
    return {"technical_analysis": response.content}


def psychology_coach_node(state: AgentState):
    """포지션 유지 시간과 감정의 상관관계 분석"""
    system_prompt = """당신은 프로 트레이더들을 케어하는 훈련된 트레이딩 심리 코치입니다.
매매 저널에 기록된 감정 상태(emotion)와 포지션 유지 시간(duration_seconds)을 종합적으로 매핑하여 심리적 취약성을 분석하세요.

특히 포지션 유지 시간이 극단적으로 짧은 손실 거래(공포로 인한 패닉 청산)나 과도하게 길어진 손실 거래(원칙을 무시한 버티기 및 기도 매매)를 찾아내어, 감정 태그와 행동 사이의 인과관계를 날카롭게 짚어내야 합니다."""

    user_content = f"심리 분석 대상 데이터:\n{json.dumps(state['journals'], ensure_ascii=False)}"
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_content)])
    
    return {"psychology_analysis": response.content}


def synthesizer_node(state: AgentState):
    """분석 원본을 가공해서 마크다운 포맷으로 컴파일"""
    meta = state["request_meta"]
    
    system_prompt = f"""당신은 리포트 편집을 담당하는 수석 편집자입니다.
앞서 리스크 관리자, 기술적 분석가, 심리 코치가 도출한 결과를 바탕으로 아래의 마크다운 양식을 100% 동일하게 유지하여 최종 리포트를 작성하세요. 전문가들의 개별 서술 내용이 누락 없이 각 섹션에 체계적으로 통합되어야 합니다.

[필수 출력 양식]
### 성과 비교 분석
(리스크 관리자의 분석을 바탕으로 과거 누적 승률 {meta['historical_win_rate']}%, 수익금 {meta['historical_avg_pnl']} USDT와 이번 주기 승률 {meta['batch_win_rate']}%, 수익금 {meta['batch_avg_pnl']} USDT를 비교하여 발전 및 하락 여부를 팩트 기반으로 분석)

### 시장 국면 기반 매매 성향
(기술적 분석가의 분석을 바탕으로 추세장/비추세장에서의 승률 차이와, 사용자가 어떤 장세나 종목, 포지션에서 강점/약점을 보이는지 분석)

### 행동 패턴 및 감정 분석
(심리 코치의 분석을 바탕으로 포지션 유지 시간, 진입/청산 근거, 감정 상태를 종합하여 뇌동매매, 탐욕, 공포 등의 패턴 분석)

### 트레이더 강점 및 개선점
- 강점: (전체 분석을 관통하는 객관적인 매매 장점 1~2줄 요약)
- 개선점: (전체 분석을 관통하는 객관적인 매매 단점 1~2줄 요약)"""

    expert_inputs = f"1. 리스크 분석:\n{state['risk_analysis']}\n\n2. 기술 분석:\n{state['technical_analysis']}\n\n3. 심리 분석:\n{state['psychology_analysis']}"
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=expert_inputs)])
    
    return {"final_report": response.content}


# LangGraph 워크플로우 그래프
workflow = StateGraph(AgentState)

workflow.add_node("risk_manager", risk_manager_node)
workflow.add_node("technical_analyst", technical_analyst_node)
workflow.add_node("psychology_coach", psychology_coach_node)
workflow.add_node("synthesizer", synthesizer_node)

workflow.add_edge(START, "risk_manager")
workflow.add_edge("risk_manager", "technical_analyst")
workflow.add_edge("technical_analyst", "psychology_coach")
workflow.add_edge("psychology_coach", "synthesizer")
workflow.add_edge("synthesizer", END)

agent_engine = workflow.compile()


# 진입점
def analyze_trading_performance(request: PerformanceRequest):
    journals = [j.model_dump() for j in request.journals]
    regime_stats = calculate_hmm_regime(journals)
    
    initial_state: AgentState = {
        "journals": journals,
        "regime_data": regime_stats,
        "request_meta": {
            "historical_win_rate": request.historical_win_rate,
            "historical_avg_pnl": request.historical_avg_pnl,
            "batch_win_rate": request.batch_win_rate,
            "batch_avg_pnl": request.batch_avg_pnl
        },
        "risk_analysis": "",
        "technical_analysis": "",
        "psychology_analysis": "",
        "final_report": ""
    }
    
    output_state = agent_engine.invoke(initial_state)
    
    return output_state["final_report"]