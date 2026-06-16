FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Chrome for Testing — pinned, never-auto-updating browser + matching driver, baked
# into the image so the version never drifts out of sync with chromedriver (the
# SessionNotCreatedException class of failures). Replaces apt google-chrome-stable.
# Keep CFT_VERSION in sync with PINNED_VERSION in setup_chrome_for_testing.py.
ARG CFT_VERSION=148.0.7778.178
ENV CHROME_FOR_TESTING_DIR=/opt/chrome-for-testing

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget unzip ca-certificates \
        fonts-liberation libasound2 libatk-bridge2.0-0 libatk1.0-0 libatspi2.0-0 \
        libcups2 libdbus-1-3 libdrm2 libgbm1 libgtk-3-0 libnspr4 libnss3 \
        libpango-1.0-0 libx11-6 libxcb1 libxcomposite1 libxdamage1 libxext6 \
        libxfixes3 libxkbcommon0 libxrandr2 xdg-utils \
    && mkdir -p "$CHROME_FOR_TESTING_DIR" \
    && wget -q "https://storage.googleapis.com/chrome-for-testing-public/${CFT_VERSION}/linux64/chrome-linux64.zip" -O /tmp/c.zip \
    && wget -q "https://storage.googleapis.com/chrome-for-testing-public/${CFT_VERSION}/linux64/chromedriver-linux64.zip" -O /tmp/d.zip \
    && unzip -q /tmp/c.zip -d "$CHROME_FOR_TESTING_DIR" \
    && unzip -q /tmp/d.zip -d "$CHROME_FOR_TESTING_DIR" \
    && chmod +x "$CHROME_FOR_TESTING_DIR/chrome-linux64/chrome" \
                "$CHROME_FOR_TESTING_DIR/chromedriver-linux64/chromedriver" \
    && echo "${CFT_VERSION}" > "$CHROME_FOR_TESTING_DIR/version.txt" \
    && rm -f /tmp/c.zip /tmp/d.zip \
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
# Predictions-cache package: ships the Python module only — the .db itself is
# excluded by .dockerignore (**/*.db) and bind-mounted from next to docker-compose.yml.
COPY Database_of_cached_results_for_predictions/ ./Database_of_cached_results_for_predictions/

EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
