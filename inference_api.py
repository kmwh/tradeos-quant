import uvicorn
from dotenv import load_dotenv
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

class AiAnalyzeRequest(BaseModel):
    position: str
    entry_reason: str
    exit_reason: str
    emotion_tag: str
    hmm_score: int

class HmmResponse(BaseModel):
    timestamp: str
    symbol: str
    trend_score: int

class AgentState(TypedDict):
    journal_data: Dict[str, Any]
    feedback: str

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "당신은 수학적 데이터 기반의 퀀트 트레이딩 코치입니다. HMM 추세 점수(0~100)는 0에 가까울수록 횡보, 100에 가까울수록 추세장입니다."),
    ("user", """
    아래 유저 매매 데이터를 분석하고 피드백을 주세요.
    - 포지션: {position}
    - 진입 사유: {entry_reason}
    - 종료 사유: {exit_reason}
    - 감정 상태: {emotion_tag}
    - HMM 점수: {hmm_score}
    
    분석은 1. 팩트 체크, 2. 행동 분석, 3. 전략 제안 순으로 작성하세요.
    """)
])

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
chain = PROMPT_TEMPLATE | llm

def generate_feedback_node(state: AgentState):
    journal = state["journal_data"]
    response = chain.invoke(journal)
    return {"feedback": response.content}

workflow = StateGraph(AgentState)
workflow.add_node("coach_agent", generate_feedback_node)
workflow.set_entry_point("coach_agent")
workflow.add_edge("coach_agent", END)
app_graph = workflow.compile()

app = FastAPI()

@app.post("/api/v1/ai/analyze")
def analyze_journal(request: AiAnalyzeRequest):
    initial_state = {
        "journal_data": request.model_dump(),
        "feedback": ""
    }
    
    result = app_graph.invoke(initial_state)
    return {"feedback_text": result["feedback"]}

@app.get("/api/v1/hmm/predict", response_model=HmmResponse)
def get_latest_hmm_score(symbol: str = "BTC/USDT"):
    # ------------------------------------------------------------
    # 추후 실제 로직이 들어갈 자리
    # ------------------------------------------------------------

    current_candle_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    
    return HmmResponse(
        timestamp=current_candle_time.isoformat(),
        symbol=symbol,
        trend_score=75
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)