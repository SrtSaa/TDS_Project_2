FROM python:3.12-slim

# --- System deps required by Playwright browsers AND Tesseract ---
RUN apt-get update && apt-get install -y \
    wget gnupg ca-certificates curl unzip \
    # Playwright dependencies
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libxkbcommon0 \
    libgtk-3-0 libgbm1 libasound2 libxcomposite1 libxdamage1 libxrandr2 \
    libxfixes3 libpango-1.0-0 libcairo2 \
    # Tesseract OCR engine
    tesseract-ocr \
    # FFmpeg for audio processing (pydub)
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# --- Install uv package manager ---
RUN pip install uv

# --- Copy app to container ---
WORKDIR /app
COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

# --- 1. Install project dependencies (creates .venv) ---
RUN uv sync --frozen

# --- 2. Activate uv environment & Install Browsers ---
# We use 'uv run' so it uses the EXACT playwright version installed above
# We skip '--with-deps' because you installed system deps manually above
RUN uv run playwright install chromium

# HuggingFace Spaces exposes port 7860
EXPOSE 7860

# --- Run your FastAPI app ---
CMD ["uv", "run", "main.py"]