FROM python:3.12-slim

# System deps: ffmpeg for audio, build tools for librosa/essentia
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Essentia runs directly (no subprocess venv needed — we're on 3.12)
# Create a symlink so the subprocess path resolves
RUN mkdir -p .venv312/bin && ln -s /usr/local/bin/python .venv312/bin/python

# Writable dirs for cache and audio
RUN mkdir -p static/audio

EXPOSE 10000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--timeout", "600", "--workers", "2"]
