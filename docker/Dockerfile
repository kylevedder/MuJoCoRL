# Requires host NVIDIA driver + nvidia-container-runtime
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    git curl ca-certificates build-essential \
    libegl1 libgl1 libglvnd0 libgles2 libxext6 libx11-6 \
    libosmesa6 patchelf ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV MUJOCO_GL=egl
RUN python3 -m pip install --no-cache-dir uv==0.4.*

# Option A: two-step install
RUN python3 -m pip install --no-cache-dir torch==2.4.* --index-url https://download.pytorch.org/whl/cu124 
RUN python3 -m pip install --no-cache-dir \
  "gymnasium>=1.2,<2" \
  "mujoco>=3.0" \
  "gymnasium-robotics>=1.4" \
  numpy tqdm matplotlib


# before: cd /opt/cleanrl && uv pip install .
ENV UV_SYSTEM_PYTHON=1
RUN git clone https://github.com/vwxyzjn/cleanrl.git /opt/cleanrl && \
    cd /opt/cleanrl && git checkout 004f8a0 && uv pip install --system --no-deps .

RUN ln -s /usr/bin/python3 /usr/local/bin/python


COPY bashrc /root/.bashrc

WORKDIR /project
CMD ["bash"]
