# CREST_demo — HF Space (Docker SDK). Builds the mchen15ouedu/EF5 fork with
# Apache Arrow (PQF/Parquet forcing) + CRESTPHYS, then runs the Gradio app.
# Based on AQUAH's working Dockerfile; changes: fork clone URL, Arrow libs,
# ./configure --with-arrow.
FROM ubuntu:22.04

LABEL name="CREST_demo"
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---- System dependencies (EF5 build + geo stack + report tooling) ----
# apt hardening (2026-08-18: HF's builder lost archive.ubuntu.com for an hour
# — three consecutive BUILD_ERRORs at this step): retries + a second mirror
# tried before the primary, so one dead host does not fail the build.
RUN echo 'Acquire::Retries "5"; Acquire::http::Timeout "45";' \
        > /etc/apt/apt.conf.d/80-retries \
 && sed -i 's|http://archive.ubuntu.com/ubuntu|http://us.archive.ubuntu.com/ubuntu|g' \
        /etc/apt/sources.list \
 && (apt-get update || (sleep 20 && apt-get update)) \
 && apt-get install -y --no-install-recommends \
    git gcc g++ build-essential make \
    autoconf automake libtool dh-autoreconf autotools-dev pkg-config \
    libgeotiff-dev libtiff-dev zlib1g-dev \
    python3 python3-dev python3-pip python-is-python3 \
    wget ca-certificates lsb-release gnupg \
    pandoc texlive-xetex lmodern texlive-fonts-recommended texlive-latex-recommended \
    libgeos-dev libproj-dev libgdal-dev \
 && rm -rf /var/lib/apt/lists/*

# ---- Apache Arrow C++ + Parquet (required for the EF5 fork's PQF reader) ----
RUN wget -q https://apache.jfrog.io/artifactory/arrow/ubuntu/apache-arrow-apt-source-latest-jammy.deb \
 && apt-get install -y -V ./apache-arrow-apt-source-latest-jammy.deb \
 && apt-get update \
 && apt-get install -y -V libarrow-dev libparquet-dev \
 && rm -rf /var/lib/apt/lists/* apache-arrow-apt-source-latest-jammy.deb

# ---- Build EF5 fork (CRESTPHYS + lake + native Parquet forcing) ----
WORKDIR /EF5
RUN git clone https://github.com/mchen15ouedu/EF5.git . \
 && git checkout 5a26a86 \
 && autoreconf --force --install \
 && ./configure --with-arrow CXXFLAGS="-std=c++20 -Wall -O2" CFLAGS="-Wall -O2" \
 && sed -i 's/-Werror//g' Makefile \
 && make -j"$(nproc)" \
 && test -x bin/ef5

# ---- Python environment ----
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip \
 && pip3 install --no-cache-dir -r requirements.txt

# ---- CREST-iMAP v2 (V25 event inundation): CPU torch + crestimap ----
# torch from the CPU wheel index FIRST so crestimap's "torch" dep is already
# satisfied (PyPI default would pull multi-GB CUDA wheels). The fork is ~2 GB
# (v1 case data), so clone blobless+sparse: only crestimap/ is materialized.
# CRESTIMAP_REF pins the fork commit: bump it to deploy a new engine (a bare
# "origin/v2" checkout sits in a cached layer and silently keeps the old code)
ARG CRESTIMAP_REF=678cbf17
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && git clone --filter=blob:none --no-checkout https://github.com/mchen15ouedu/CREST-iMAP.git /opt/crest-imap \
 && git -C /opt/crest-imap checkout ${CRESTIMAP_REF} -- crestimap \
 && test -f /opt/crest-imap/crestimap/__init__.py \
 && cp -r /opt/crest-imap/crestimap "$(python3 -c 'import site; print(site.getsitepackages()[0])')/" \
 && rm -rf /opt/crest-imap \
 && python3 -c "import crestimap; print('crestimap', crestimap.__version__)"

# ---- App code + EF5 binary on the expected path (AQUAH uses ./EF5/bin/ef5) ----
COPY . .
RUN ln -sf /EF5 /app/EF5 && ln -sf /EF5/bin/ef5 /usr/local/bin/ef5

# HF Spaces (Docker SDK) serves on 7860
EXPOSE 7860
CMD ["python", "server.py"]
