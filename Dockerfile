FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Google Chrome (required by selenium / undetected-chromedriver for the twitter scrapper)
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget gnupg ca-certificates curl \
    && curl -fsSL https://dl.google.com/linux/linux_signing_key.pub \
        | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" \
        > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update && apt-get install -y --no-install-recommends google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first to keep this layer cached when only code changes
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code (heavy runtime assets are mounted via docker-compose volumes;
# everything else excluded by .dockerignore)
COPY api/ ./api/
COPY MultiagentSystem/ ./MultiagentSystem/
COPY configs/ ./configs/
COPY Logs/LoggingSystem/ ./Logs/LoggingSystem/

EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
