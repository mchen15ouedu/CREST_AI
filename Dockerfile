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
RUN apt-get update && apt-get install -y --no-install-recommends \
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

# ---- App code + EF5 binary on the expected path (AQUAH uses ./EF5/bin/ef5) ----
COPY . .
RUN ln -sf /EF5 /app/EF5 && ln -sf /EF5/bin/ef5 /usr/local/bin/ef5

# HF Spaces (Docker SDK) serves on 7860
EXPOSE 7860
CMD ["python", "server.py"]
