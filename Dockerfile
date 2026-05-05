FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_RELEASE_TAG=eng-sp-tranlate \
    TRANSLATOR_AUTO_DOWNLOAD_MODEL=1 \
    TRANSLATOR_HOST=0.0.0.0 \
    TRANSLATOR_PORT=8000

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt && \
    adduser --disabled-password --gecos "" appuser

COPY . .

RUN chmod +x scripts/docker-entrypoint.sh && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

ENTRYPOINT ["scripts/docker-entrypoint.sh"]
