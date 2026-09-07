"""
src/mock_data.py
- 대규모 벤치마크 테스트를 위한 매매 저널 및 DTO 생성 모듈 (100건 이상)
"""
import random
from typing import List
from schemas import PerformanceRequest, JournalData

def get_benchmark_dataset(count: int = 100) -> PerformanceRequest:
    """
    현실적인 매매 분포를 반영한 저널 데이터셋 생성 (결과 일관성을 위해 고정 시드 적용)
    - 정상 매매: 긴 보유 시간, 5~12배 레버리지, HMM 추세 국면(>60), CALM
    - 뇌동 매매: 짧은 보유 시간, 20~25배 레버리지, HMM 비추세 국면(<=60), FEARFUL / GREEDY
    """
    random.seed(42)
    tickers = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    # 1. 시나리오별 프로토타입 템플릿
    normal_setups = [
        ("LONG", (5.0, 12.0), "4H 골든크로스 돌파 지지", "직전 저항선 도달 분할 익절", "CALM", (150.0, 450.0), (8.0, 22.0), (14400, 86400), (65, 88)),
        ("LONG", (5.0, 10.0), "추세 지지선 리테스트 확인 진입", "추세 둔화 시그널 전량 청산", "CALM", (100.0, 320.0), (5.0, 16.0), (18000, 72000), (62, 85)),
        ("LONG", (3.0, 5.0), "박스권 하단 지지선 분할 매수", "박스권 상단 목표 도달", "CALM", (50.0, 180.0), (3.0, 9.0), (28800, 90000), (51, 58))
    ]

    emotional_setups = [
        ("SHORT", (20.0, 25.0), "급락 후 반등 뇌동 숏 추격", "스탑로스 강제 터치 손절", "FEARFUL", (-350.0, -150.0), (-60.0, -35.0), (300, 900), (38, 49)),
        ("SHORT", (18.0, 25.0), "FOMO 고점 숏 배팅", "공포 심리 조기 패닉 청산", "GREEDY", (-300.0, -120.0), (-55.0, -25.0), (600, 1500), (42, 50)),
        ("LONG", (20.0, 25.0), "급등 구간 추격 롱 매수", "지지선 붕괴 급격한 패닉셀", "GREEDY", (-280.0, -100.0), (-50.0, -20.0), (450, 1200), (40, 52))
    ]

    journals: List[JournalData] = []

    # 2. 6:4 비율로 정상 매매와 감정 매매 분배 생성
    normal_count = int(count * 0.6)
    emotional_count = count - normal_count

    # 정상 매매 생성
    for _ in range(normal_count):
        pos, lev_r, ent, ex, emo, pnl_r, roi_r, dur_r, hmm_r = random.choice(normal_setups)
        journals.append(JournalData(
            ticker=random.choice(tickers),
            position=pos,
            leverage=round(random.uniform(*lev_r), 1),
            entry_reason=ent,
            exit_reason=ex,
            emotion=emo,
            pnl=round(random.uniform(*pnl_r), 2),
            roi=round(random.uniform(*roi_r), 2),
            duration_seconds=random.randint(*dur_r),
            hmm_score=random.randint(*hmm_r)
        ))

    # 감정 매매 생성
    for _ in range(emotional_count):
        pos, lev_r, ent, ex, emo, pnl_r, roi_r, dur_r, hmm_r = random.choice(emotional_setups)
        journals.append(JournalData(
            ticker=random.choice(tickers),
            position=pos,
            leverage=round(random.uniform(*lev_r), 1),
            entry_reason=ent,
            exit_reason=ex,
            emotion=emo,
            pnl=round(random.uniform(*pnl_r), 2),
            roi=round(random.uniform(*roi_r), 2),
            duration_seconds=random.randint(*dur_r),
            hmm_score=random.randint(*hmm_r)
        ))

    # 데이터 순서 무작위 혼합
    random.shuffle(journals)

    # 3. 100건 데이터 기반 실제 성과 수치 동적 산출
    wins = [j for j in journals if j.pnl > 0]
    calc_batch_win_rate = round((len(wins) / count) * 100, 2)
    calc_batch_avg_pnl = round(sum(j.pnl for j in journals) / count, 2)

    return PerformanceRequest(
        batch_size=count,
        historical_win_rate=54.2,
        historical_avg_pnl=65.0,
        batch_win_rate=calc_batch_win_rate,
        batch_avg_pnl=calc_batch_avg_pnl,
        journals=journals
    )