FROM python:3.12-slim

# System deps: ffmpeg for audio, libsndfile for librosa
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install core Python deps (always needed)
COPY requirements.txt .
RUN pip install --no-cache-dir flask gunicorn yt-dlp librosa numpy scipy

# Try to install essentia (may fail on some platforms — that's OK, librosa fallback works)
RUN pip install --no-cache-dir essentia-tensorflow || echo "WARNING: essentia not available, using librosa fallback"

# Copy app
COPY . .

# If essentia installed, create symlink so subprocess finds python
RUN mkdir -p .venv312/bin && ln -s $(which python) .venv312/bin/python

# Writable dirs for cache and audio
RUN mkdir -p static/audio

EXPOSE 10000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--timeout", "600", "--workers", "1"]
