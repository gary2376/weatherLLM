FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY code/ ./code/
COPY collector/ ./collector/
COPY .streamlit/ ./.streamlit/
COPY warmup_fastembed.py ./

# Download both local inference models into the image so model initialization
# does not need Hugging Face access at runtime.
RUN python warmup_fastembed.py

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).read()"]

CMD ["streamlit", "run", "code/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
