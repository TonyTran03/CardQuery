FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CARDQUERY_PARSER_BACKEND=real
ENV PREFECT_SERVER_ANALYTICS_ENABLED=false

COPY training/requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app

CMD ["python", "-m", "training.bootstrap"]