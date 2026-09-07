import os
import sys
import time
import json
import asyncio
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# .env 및 src 경로 로드
root_dir = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=root_dir / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from schemas import PerformanceRequest
from mock_data import get_benchmark_dataset
from ai_service import calculate_hmm_regime, agent_engine, HMM_TREND_THRESHOLD
from prompts import SINGLE_PROMPT_SYSTEM

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.1)

# 병렬 노드들의 토큰 소비량을 실시간 합산하는 콜백 핸들러
class TokenUsageTracker(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.input_tokens = 0
        self.output_tokens = 0

    def on_llm_end(self, response, **kwargs):
        # 1. AIMessage usage_metadata 탐색
        for gen_list in getattr(response, "generations", []):
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg and hasattr(msg, "usage_metadata") and msg.usage_metadata:
                    self.input_tokens += msg.usage_metadata.get("input_tokens", 0)
                    self.output_tokens += msg.usage_metadata.get("output_tokens", 0)
                    return

        # 2. llm_output fallback 탐색
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {}) or {}
            self.input_tokens += usage.get("prompt_tokens", 0)
            self.output_tokens += usage.get("completion_tokens", 0)

def extract_tokens(res) -> tuple[int, int]:
    meta = getattr(res, "usage_metadata", {}) or {}
    return meta.get("input_tokens", 0), meta.get("output_tokens", 0)

def to_clean_text(content) -> str:
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

# 1. 단일 거대 프롬프트 시나리오
async def test_single(data: PerformanceRequest):
    t0 = time.perf_counter()
    journals = [j.model_dump() for j in data.journals]
    regime = calculate_hmm_regime(journals)
    
    sys_prompt = SINGLE_PROMPT_SYSTEM.format(
        hist_win=data.historical_win_rate, hist_pnl=data.historical_avg_pnl,
        batch_win=data.batch_win_rate, batch_pnl=data.batch_avg_pnl,
        hmm_threshold=HMM_TREND_THRESHOLD,
        trend_cnt=regime["trend_market"]["trade_count"], trend_win=regime["trend_market"]["win_rate"],
        chop_cnt=regime["chop_market"]["trade_count"], chop_win=regime["chop_market"]["win_rate"]
    )
    user_msg = f"매매 기록 데이터:\n{json.dumps(journals, ensure_ascii=False)}"
    res = await llm.ainvoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_msg)])
    lat = time.perf_counter() - t0
    in_t, out_t = extract_tokens(res)
    
    return {
        "scenario": "1. Single Prompt",
        "latency": lat,
        "in_tok": in_t,
        "out_tok": out_t,
        "total_tok": in_t + out_t,
        "report_text": to_clean_text(res.content)
    }

# 2. 멀티 에이전트 Fan-Out 병렬 실행 시나리오 (LangGraph DAG)
async def test_parallel(data: PerformanceRequest):
    tracker = TokenUsageTracker()
    t0 = time.perf_counter()
    journals = [j.model_dump() for j in data.journals]
    
    state = {
        "journals": journals,
        "regime_data": calculate_hmm_regime(journals),
        "request_meta": {
            "historical_win_rate": data.historical_win_rate,
            "historical_avg_pnl": data.historical_avg_pnl,
            "batch_win_rate": data.batch_win_rate,
            "batch_avg_pnl": data.batch_avg_pnl
        },
        "risk_analysis": "", "technical_analysis": "", "psychology_analysis": "", "final_report": ""
    }
    
    # LangGraph 실행 시 콜백 핸들러를 주입하여 모든 분기 노드의 토큰 합산 추적
    final_state = await agent_engine.ainvoke(state, config={"callbacks": [tracker]})
    lat = time.perf_counter() - t0
    
    return {
        "scenario": "2. Multi-Agent Parallel",
        "latency": lat,
        "in_tok": tracker.input_tokens,
        "out_tok": tracker.output_tokens,
        "total_tok": tracker.input_tokens + tracker.output_tokens,
        "report_text": to_clean_text(final_state["final_report"])
    }

async def main():
    # 100건 매매 저널 데이터 주입
    DATASET_SIZE = 100
    dataset = get_benchmark_dataset(count=DATASET_SIZE)
    
    print(f"🚀 [TradeOS AI 파이프라인 벤치마크 시작 (모델: {MODEL_NAME} | 데이터: {DATASET_SIZE}건)]\n")

    print("1. Single Prompt 실행 중...")
    s1 = await test_single(dataset)
    print(f"   -> 지연시간: {s1['latency']:.2f}초 | 총 토큰: {s1['total_tok']}")

    # API 호출 간격 보호
    await asyncio.sleep(3)

    print("2. Multi-Agent Parallel 실행 중 (LangGraph 비동기 DAG)...")
    s2 = await test_parallel(dataset)
    print(f"   -> 지연시간: {s2['latency']:.2f}초 | 총 토큰: {s2['total_tok']}\n")

    # 결과 마크다운 리포트 저장
    output_dir = root_dir / "benchmark_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "report_single_prompt_100.md").write_text(s1["report_text"], encoding="utf-8")
    (output_dir / "report_multi_agent_100.md").write_text(s2["report_text"], encoding="utf-8")

    print(f"📄 리포트 마크다운 파일 저장 완료:")
    print(f"   - 단일 프롬프트: {output_dir / 'report_single_prompt_100.md'}")
    print(f"   - Multi-Agent : {output_dir / 'report_multi_agent_100.md'}\n")

    # 요약 통계 출력
    df = pd.DataFrame([
        {k: v for k, v in s.items() if k != "report_text"}
        for s in [s1, s2]
    ])

    print("=" * 70)
    print(f"📊 [TradeOS AI 벤치마크 프로파일링 최종 결과 (N={DATASET_SIZE})]")
    print("=" * 70)
    header = f"{'시나리오':<25} | {'지연시간(초)':<10} | {'입력 토큰':<9} | {'출력 토큰':<9} | {'총 토큰':<8}"
    print(header)
    print("-" * 70)
    for _, row in df.iterrows():
        print(f"{row['scenario']:<25} | {row['latency']:<10.2f} | {int(row['in_tok']):<9} | {int(row['out_tok']):<9} | {int(row['total_tok']):<8}")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())