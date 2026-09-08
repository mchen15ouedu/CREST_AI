# CREST_updater — Docker Space that refreshes the CREST_demo data feeds.
# Needs libeccodes (cfgrib reads MRMS grib2) and libexpat1 (the rasterio/GDAL
# wheel dlopens the system libexpat.so.1, which python:3.11-slim stopped
# shipping in 2026-08 — without it `import rasterio` fails and every PET
# produce() errors). Everything else is pip wheels.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libeccodes0 libexpat1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# writable caches for the non-root HF runtime user
ENV HF_HOME=/tmp/hf_home \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    HF_HUB_DISABLE_PROGRESS_BARS=1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
