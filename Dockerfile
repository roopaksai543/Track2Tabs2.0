FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Demucs and soundfile need these native audio libraries at runtime.
RUN apt-get update \
    && apt-get install --no-install-recommends -y ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio \
    && pip install -r backend/requirements.txt

# Cache the pretrained separator in the image instead of downloading it on the
# first user request.
RUN python -c "from demucs.pretrained import get_model; get_model('htdemucs')"

COPY backend backend
COPY ml/artifacts ml/artifacts

CMD ["sh", "-c", "uvicorn app:app --app-dir backend --host 0.0.0.0 --port ${PORT:-8000}"]
