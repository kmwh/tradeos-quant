import json
from typing import Annotated
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from schemas import PerformanceRequest

# 도구 정의
@tool
def analyze_hmm_regime(journals_json: str) -> str:
    """
    사용자의 매매 기록을 입력받아 hmm_score를 기준으로 추세장과 비추세장에서의 매매 횟수와 승률을 계산하여 반환한다.
    """
    try:
        journals = json.loads(journals_json)
        trend_trades = [j for j in journals if j.get("hmm_score", 50) > 60]
        chop_trades = [j for j in journals if j.get("hmm_score", 50) <= 60]

        def calc_win_rate(trades):
            if not trades: return 0.0
            wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
            return round((wins / len(trades)) * 100, 2)

        return json.dumps({
            "trend_market": {
                "trade_count": len(trend_trades),
                "win_rate": calc_win_rate(trend_trades)
            },
            "chop_market": {
                "trade_count": len(chop_trades),
                "win_rate": calc_win_rate(chop_trades)
            }
        })
    except Exception as e:
        return f"Error analyzing regime: {str(e)}"

class State(TypedDict):
    messages: Annotated[list, add_messages]

def analyze_trading_performance(request: PerformanceRequest) -> str:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    tools = [analyze_hmm_regime]
    
    llm_with_tools = llm.bind_tools(tools)

    def chatbot_node(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    tool_node = ToolNode(tools=tools)

    graph_builder = StateGraph(State)
    
    # 노드 추가
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_node("tools", tool_node)
    
    # 그래프 연결
    # 시작 -> 챗봇
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")
    
    # 그래프 컴파일
    agent_engine = graph_builder.compile()

    system_instruction = f"""당신은 수석 퀀트 트레이딩 코치이자 분석 AI 에이전트입니다.
반드시 제공된 'analyze_hmm_regime' 툴을 사용하여 사용자의 추세/비추세장 승률 데이터를 먼저 확보하세요.
데이터 분석이 끝나면, **반드시 아래의 마크다운 양식을 100% 동일하게 유지**하여 최종 리포트를 작성해야 합니다.

[필수 출력 양식]
### 성과 비교 분석
(과거 누적 승률 {request.historical_win_rate}%, 수익금 {request.historical_avg_pnl} USDT와 이번 주기 승률 {request.batch_win_rate}%, 수익금 {request.batch_avg_pnl} USDT를 비교하여 발전 및 하락 여부를 팩트 기반으로 분석)

### 시장 국면(HMM) 기반 매매 성향
(툴의 결과를 바탕으로 추세장/비추세장에서의 승률 차이와, 사용자가 어떤 장세에서 강점/약점을 보이는지 분석)

### 행동 패턴 및 감정 분석
(매매 타점, 진입/청산 근거, 감정 상태를 종합하여 뇌동매매, 탐욕, 공포 등의 패턴 분석)

### 트레이더 강점 및 개선점
- **강점:** (객관적인 매매 장점 1~2줄 요약)
- **개선점:** (객관적인 매매 단점 1~2줄 요약)"""

    journals_str = json.dumps([j.model_dump() for j in request.journals])
    user_input = f"매매 기록 데이터(JSON):\n{journals_str}"
    
    initial_messages = [
        SystemMessage(content=system_instruction),
        HumanMessage(content=user_input)
    ]
    
    # 에이전트 엔진 실행
    response = agent_engine.invoke({"messages": initial_messages})
    
    return response["messages"][-1].content