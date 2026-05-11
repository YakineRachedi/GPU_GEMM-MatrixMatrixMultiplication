FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /usr/local/workspace

RUN apt-get update && apt-get install -y \
    build-essential \
    make \
    libopenblas-dev \
    python3 \
    python3-pip \
    vim \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install pandas matplotlib numpy

COPY . .
CMD ["/bin/bash"]