FROM python:3.12-slim

# Proxies entreprise
ENV HTTP_PROXY=''
ENV HTTPS_PROXY=''
ENV NO_PROXY=''
ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTPS_PROXY
ENV no_proxy=$NO_PROXY

# === STATELESS DEPLOYMENT ===
# Disable pickle cache, force cleanup, redirect experiments to /tmp.
# All app state vanishes at container restart — only the model_storage volume
# (mounted read-only) persists across runs.
ENV APP_ENV=production
# Logs go to stdout for `docker logs` (no logs.txt file path expected on disk)
ENV APP_LOG_TO_CONSOLE=true
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Dépendances système minimales pour audio (ffmpeg + libsndfile pour pydub/soundfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code applicatif. Le .dockerignore exclut le contenu lourd
# (model_storage, experiments, audio data, IDE temp files…).
COPY . .

# Streamlit healthcheck: surface failures to docker swarm/k8s.
# Use python (already available) instead of pulling curl — keeps image smaller.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health',timeout=5).status==200 else 1)" || exit 1

EXPOSE 8501
EXPOSE 8000

# Default = Streamlit. Override at run time:
#   docker run ... <image> python run.py --serve --port 8000   (FastAPI)
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
