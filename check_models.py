# check_models.py
import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# .env 로드
load_dotenv(Path(__file__).resolve().parent / ".env")
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

print(f"=== 발급된 API Key로 사용 가능한 모델 목록 ===")
for model in client.models.list():
    # 텍스트 생성(generateContent)을 지원하는 모델만 출력
    actions = getattr(model, "supported_actions", None) or getattr(model, "supported_generation_methods", [])
    if "generateContent" in actions:
        # 'models/' 접두사를 뗀 순수 모델 식별자 출력
        clean_name = model.name.replace("models/", "")
        print(f"- {clean_name}")