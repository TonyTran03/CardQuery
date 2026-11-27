FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CARDQUERY_PARSER_BACKEND=real
ENV PREFECT_SERVER_ANALYTICS_ENABLED=false

COPY training/requirements.txt training/requirements.txt
COPY parser/requirements.txt parser/requirements.txt
COPY parser/requirements-semantic.txt parser/requirements-semantic.txt

RUN pip install --no-cache-dir -r training/requirements.txt

COPY . /app

CMD ["python", "-m", "training.bootstrap"]
