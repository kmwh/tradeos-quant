import datetime
from typing import TypedDict, Dict, Any, Literal
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import uvicorn

load_dotenv()

# GOOGLE_API_KEY는 환경 변수에서 자동으로 읽어옴
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

app = FastAPI()

# ----- [DTO Models] -----
class HmmResponse(BaseModel):
    timestamp: str
    symbol: str
    trend_score: int

class AiAnalyzeRequest(BaseModel):
    position: str
    entry_reason: str
    exit_reason: str
    emotion_tag: str
    hmm_score: int
    roi: float

class TendencyRequest(BaseModel):
    period: str
    win_rate: float
    frequent_emotion: str
    total_trades: int

# ----- [LangGraph State & Nodes] -----
class AgentState(TypedDict):
    journal_data: Dict[str, Any]
    feedback: str
    expert_type: str

def router_node(state: AgentState) -> Literal["mental_coach", "quant_coach"]:
    emotion = state["journal_data"].get("emotion_tag", "")
    if emotion in ["FOMO", "불안", "분노", "뇌동매매"]:
        return "mental_coach"
    return "quant_coach"

def mental_coach_node(state: AgentState):
    journal = state["journal_data"]
    
    context = {
        "position": journal.get("position", "N/A"),
        "emotion_tag": journal.get("emotion_tag", "N/A"),
        "hmm_score": journal.get("hmm_score", 50),
        "entry_reason": journal.get("entry_reason", "N/A"),
        "exit_reason": journal.get("exit_reason", "N/A"),
        "roi": journal.get("roi", 0.0)
    }
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 시스템 트레이딩 및 멘탈 관리 전문가입니다. 트레이더가 기록한 감정 상태와 진입/종료 사유를 분석하여 심리적 통제력과 리스크 관리에 대한 객관적인 피드백을 제공하세요. 감정에 휘둘리지 않고 기계적인 매매 원칙을 준수할 수 있도록 조언하며, 답변은 반드시 3문장 이내로 핵심만 간결하게 작성하세요."),
        ("user", "포지션: {position}, 감정: {emotion_tag}, HMM점수: {hmm_score}, 진입/종료: {entry_reason} / {exit_reason}, 수익률: {roi}%")
    ])
    response = (prompt | llm).invoke(context)
    return {"feedback": response.content, "expert_type": "Mental Coach"}

def quant_coach_node(state: AgentState):
    journal = state["journal_data"]
    
    context = {
        "position": journal.get("position", "N/A"),
        "emotion_tag": journal.get("emotion_tag", "N/A"),
        "hmm_score": journal.get("hmm_score", 50),
        "entry_reason": journal.get("entry_reason", "N/A"),
        "exit_reason": journal.get("exit_reason", "N/A"),
        "roi": journal.get("roi", 0.0)
    }
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 스마트 머니 콘셉트(SMC)와 퀀트 분석에 능통한 수석 트레이더입니다. 유저의 HMM 추세 점수와 타점(진입/종료 사유)을 바탕으로 매매의 객관적인 타당성을 평가하세요. 기술적 분석의 정합성과 기계적 익/손절 여부를 중심으로 개선점을 지적하며, 답변은 반드시 3문장 이내로 팩트만 간결하게 전달하세요."),
        ("user", "포지션: {position}, 감정: {emotion_tag}, HMM점수: {hmm_score}, 진입/종료: {entry_reason} / {exit_reason}, 수익률: {roi}%")
    ])
    response = (prompt | llm).invoke(context)
    return {"feedback": response.content, "expert_type": "Quant Coach"}

# ----- [LangGraph 조립] -----
workflow = StateGraph(AgentState)
workflow.add_node("mental_coach", mental_coach_node)
workflow.add_node("quant_coach", quant_coach_node)

workflow.set_conditional_entry_point(
    router_node,
    {
        "mental_coach": "mental_coach",
        "quant_coach": "quant_coach"
    }
)

workflow.add_edge("mental_coach", END)
workflow.add_edge("quant_coach", END)
app_graph = workflow.compile()

# ----- [API Endpoints] -----
@app.get("/api/v1/hmm/predict", response_model=HmmResponse)
def get_latest_hmm_score(symbol: str = "BTC/USDT"):
    now = datetime.datetime.now()
    current_candle_time = now.replace(minute=0, second=0, microsecond=0)
    return HmmResponse(
        timestamp=current_candle_time.isoformat(),
        symbol=symbol,
        trend_score=75
    )

@app.post("/api/v1/ai/analyze")
def analyze_journal(request: AiAnalyzeRequest):
    initial_state = {
        "journal_data": request.model_dump(),
        "feedback": "",
        "expert_type": ""
    }
    result = app_graph.invoke(initial_state)
    final_text = f"[{result['expert_type']}의 피드백]\n\n" + result["feedback"]
    return {"feedback_text": final_text}

@app.post("/api/v1/ai/tendency")
def analyze_tendency(request: TendencyRequest):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 퀀트 트레이딩 종합 코치입니다. 유저의 주간/월간 데이터를 요약하여 팩트 기반의 개선점을 3문장 이내로 정리해주세요."),
        ("user", "{period} 리포트 - 승률: {win_rate}%, 자주 느낀 감정: {frequent_emotion}, 총 매매 수: {total_trades}")
    ])
    chain = prompt | llm
    
    response = chain.invoke({
        "period": request.period,
        "win_rate": request.win_rate,
        "frequent_emotion": request.frequent_emotion,
        "total_trades": request.total_trades
    })
    
    return {"summary_text": response.content}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)