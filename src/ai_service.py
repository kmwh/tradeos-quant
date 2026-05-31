from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from schemas import PerformanceRequest

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def analyze_trading_performance(request: PerformanceRequest) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 수석 퀀트 트레이딩 코치입니다.
사용자가 최근 완료한 {batch_size}번의 매매 기록을 바탕으로 퍼포먼스를 엄격히 평가해야 합니다.

[성과 요약]
- 과거 누적 평균 승률: {historical_win_rate}% / 과거 평균 수익금: {historical_avg_pnl} USDT
- 이번 주기 평균 승률: {batch_win_rate}% / 이번 주기 평균 수익금: {batch_avg_pnl} USDT

제출된 매매 사유와 감정 상태 리스트를 종합하여 다음 3가지를 팩트 기반으로 냉철하게 평가하세요:
1. 과거 성적과 비교하여 유의미한 발전이나 하락이 있었는가?
2. 매매 진입과 청산의 근거가 객관적이고 타당했는가?
3. 매매 중 감정 통제(뇌동매매, 탐욕, 공포 등)가 잘 이루어졌는가?

답변은 반드시 3~4문장 이내로 핵심만 간결하고 명확하게 작성하세요."""),
        ("user", "매매 기록 리스트: {journals}")
    ])
    
    chain = prompt | llm
    journal_dicts = [j.model_dump() for j in request.journals]
    
    response = chain.invoke({
        "batch_size": request.batch_size,
        "historical_win_rate": request.historical_win_rate,
        "historical_avg_pnl": request.historical_avg_pnl,
        "batch_win_rate": request.batch_win_rate,
        "batch_avg_pnl": request.batch_avg_pnl,
        "journals": journal_dicts
    })
    
    return response.content