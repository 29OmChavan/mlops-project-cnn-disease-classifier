FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt setup.py README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY config/ ./config/
COPY params.yaml dvc.yaml main.py app.py ./
COPY templates/ ./templates/
COPY artifacts/ ./artifacts/
COPY scores.json ./scores.json

# If DVC artifacts were not checked into Git, generate them automatically during build
RUN if [ ! -f artifacts/training/model.pt ]; then DISABLE_REMOTE_MLFLOW=1 python main.py; fi

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
