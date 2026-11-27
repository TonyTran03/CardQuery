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
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s \
  CMD ["python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/ready', timeout=2).close()"]
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
