# CREST_fleet_runner — HF Space (Docker SDK). Same EF5 build as CREST_demo;
# at boot it clones the current CREST_AI sources and runs the virtual-user
# fleet (fleet/fleet_run.py) continuously, serving a status page on 7860.
FROM ubuntu:22.04

LABEL name="CREST_fleet_runner"
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git gcc g++ build-essential make \
    autoconf automake libtool dh-autoreconf autotools-dev pkg-config \
    libgeotiff-dev libtiff-dev zlib1g-dev \
    python3 python3-dev python3-pip python-is-python3 \
    wget ca-certificates lsb-release gnupg \
    libgeos-dev libproj-dev libgdal-dev \
 && rm -rf /var/lib/apt/lists/*

# Apache Arrow C++ + Parquet (EF5 fork's PQF reader)
RUN wget -q https://apache.jfrog.io/artifactory/arrow/ubuntu/apache-arrow-apt-source-latest-jammy.deb \
 && apt-get install -y -V ./apache-arrow-apt-source-latest-jammy.deb \
 && apt-get update \
 && apt-get install -y -V libarrow-dev libparquet-dev \
 && rm -rf /var/lib/apt/lists/* apache-arrow-apt-source-latest-jammy.deb

# EF5 fork (CRESTPHYS + lake + native Parquet forcing)
WORKDIR /EF5
RUN git clone https://github.com/mchen15ouedu/EF5.git . \
 && git checkout 5a26a86 \
 && autoreconf --force --install \
 && ./configure --with-arrow CXXFLAGS="-std=c++20 -Wall -O2" CFLAGS="-Wall -O2" \
 && sed -i 's/-Werror//g' Makefile \
 && make -j"$(nproc)" \
 && test -x bin/ef5

# Python deps: same set the dashboard uses (fleet imports hf_data.*)
WORKDIR /app
RUN wget -q -O requirements.txt \
      https://raw.githubusercontent.com/mchen15ouedu/CREST_AI/main/requirements.txt \
 && pip3 install --no-cache-dir --upgrade pip \
 && pip3 install --no-cache-dir -r requirements.txt

COPY runner.py .
EXPOSE 7860
CMD ["python3", "runner.py"]
