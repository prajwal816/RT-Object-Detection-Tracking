# ═══════════════════════════════════════════════════════════════════════════════
# Real-Time Object Detection & Tracking System — Dockerfile
# ═══════════════════════════════════════════════════════════════════════════════
#
# Build:
#   docker build -t rt-pipeline .
#
# Run (webcam):
#   docker run --rm -it --device /dev/video0 rt-pipeline --source 0
#
# Run (video file):
#   docker run --rm -it -v $(pwd)/data:/app/data rt-pipeline \
#       --source data/video.mp4 --no-show --save
#
# Run with GPU:
#   docker run --rm -it --gpus all rt-pipeline --source 0 --device cuda
# ═══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: Build C++ pipeline ──────────────────────────────────────────────
FROM ubuntu:22.04 AS cpp-builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY CMakeLists.txt .
COPY cpp/ cpp/

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -j$(nproc)

# ── Stage 2: Python runtime ─────────────────────────────────────────────────
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System dependencies (OpenCV runtime, video codecs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libopencv-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies (install first for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Copy C++ binary from builder
COPY --from=cpp-builder /build/build/rt_pipeline /usr/local/bin/rt_pipeline

# Create output directories
RUN mkdir -p data/output models benchmarks

# Default: run the Python pipeline
ENTRYPOINT ["python", "scripts/run_pipeline.py"]
CMD ["--source", "0"]
