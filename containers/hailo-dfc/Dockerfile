# Hailo Dataflow Compiler 3.x container (x86-64 Linux).
#
# Used by `frigate-learn deploy --real` when the host is macOS: the DFC only
# runs on x86-64 Linux, so the three compile stages execute inside this image
# (see docs/hailo-deploy.md). Building requires the DFC wheel, which you get
# from the Hailo developer zone (Hailo Software Suite / Dataflow Compiler
# download), e.g. `hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl`.
#
# Build from the repo root:
#
#   scripts/build-hailo-dfc.sh [wheel] [tag]
#
# Runtime: `frigate-learn` invokes `docker run --rm --platform linux/amd64
# -v <modeldir>:/work -w /work <image> hailo <stage> ...`; the image only needs
# the `hailo` CLI on PATH.

ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

ARG DFC_WHEEL=hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip python3-venv python3-dev build-essential pciutils && \
    rm -rf /var/lib/apt/lists/*

COPY ${DFC_WHEEL} /tmp/${DFC_WHEEL}
RUN python3 -m venv /opt/hailo && \
    /opt/hailo/bin/pip install --no-cache-dir /tmp/${DFC_WHEEL} psutil && \
    rm /tmp/${DFC_WHEEL}
ENV PATH="/opt/hailo/bin:${PATH}"