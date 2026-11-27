FROM python:3.11-slim

WORKDIR /app
COPY api/requirements.txt api/requirements.txt
COPY parser/requirements.txt parser/requirements.txt
RUN python -m pip install --no-cache-dir -r api/requirements.txt

COPY api/ api/
COPY search/ search/
COPY parser/ parser/
COPY catalog/ catalog/
COPY frontend/__init__.py frontend/fake_api.py frontend/fixtures.py frontend/
COPY data/cards.csv data/cards.csv
COPY artifacts/parser/model.joblib artifacts/parser/model.joblib

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
