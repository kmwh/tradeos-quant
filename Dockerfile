# 1. Base 이미지 설정
FROM python:3.12-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 의존성 파일 복사 및 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 소스 코드 복사 (src 폴더 내의 코드를 컨테이너로 복사)
COPY src/ ./src/

ENV PYTHONPATH=/app/src

# 5. 외부에 노출할 포트
EXPOSE 8000

# 6. 컨테이너가 켜질 때 실행할 명령어 (main.py 위치에 맞게 모듈 경로 지정)
# 예: src 폴더 안에 main.py가 있다면 src.main:app
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]