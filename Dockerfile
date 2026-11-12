# syntax=docker/dockerfile:1
FROM python:3.10-slim AS build

ENV CONDA_PKGS_DIRS=/root/.conda/pkgs \
    MINIFORGE_CACHE_DIR=/root/.cache/miniforge \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        git \
        bzip2 \
        ca-certificates \
        libdbus-1-3 \
        libglib2.0-0 \
        libxkbcommon-x11-0 \
        libxrender1

# Copy local build context and overwrite the cloned version
ADD setup.py \
    README.rst \
    MANIFEST.in \
    requirements.txt \
    install_sct \
    LICENSE \
    /opt/sct/
ADD spinalcordtoolbox /opt/sct/spinalcordtoolbox/
ADD data/ /opt/sct/data/
ADD contrib/ /opt/sct/contrib/

# Install SCT and dependencies
WORKDIR /opt/sct
RUN --mount=type=cache,target=/root/.cache/miniforge,sharing=locked \
    --mount=type=cache,target=/root/.conda/pkgs,sharing=locked \
    --mount=type=cache,target=/root/.mamba/pkgs,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    bash install_sct -y -c -g

FROM debian:bookworm-slim

LABEL org.opencontainers.image.source="https://github.com/spinalcordtoolbox/spinalcordtoolbox"
LABEL org.opencontainers.image.description="Comprehensive and open-source library of analysis tools for MRI of the spinal cord."
LABEL org.opencontainers.image.licenses="LGPL-3.0"

ARG USERNAME=sct
ARG USER_UID=1000
ARG USER_GID=1000
ARG DEEPSEG_TASKS

ENV DEEPSEG_TASKS=${DEEPSEG_TASKS:-""}

# Setup time and locales
RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime

# Install runtime dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        libdbus-1-3 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libxkbcommon-x11-0 \
        libxrender1

# Create non-root user for development
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Switch to non-root user
USER $USERNAME

# Copy SCT conda environment from build stage
COPY --from=build --chmod=ugo=rwX /opt/sct/ /opt/sct/
ENV PATH="/opt/sct/bin:${PATH}"
ENV SCT_DIR=/opt/sct

# Set bash shell as default
SHELL ["/opt/sct/python/bin/conda", "run", "-n", "venv_sct", "/bin/bash", "-c"]

# Install listed models if any (comma separated list, call sct_deepseg <model> -install for each)
RUN if [ -n "$DEEPSEG_TASKS" ]; then \
        echo "Installing deepseg models: $DEEPSEG_TASKS"; \
        set -euo pipefail; \
        printf '%s\n' "$DEEPSEG_TASKS" | tr ',' '\n' | while read -r MODEL; do \
            echo "Installing model: $MODEL"; \
            sct_deepseg "$MODEL" -install; \
            # Sleep for 5 seconds to avoid potential rate limits
            sleep 5; \
        done; \
    fi
