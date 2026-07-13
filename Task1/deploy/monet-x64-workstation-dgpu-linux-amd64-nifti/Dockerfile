
ARG GPU_TYPE=dgpu




FROM nvcr.io/nvidia/cuda:12.6.0-runtime-ubuntu22.04 AS base

RUN apt-get update \
    && apt-get install -y --no-install-recommends --no-install-suggests \
        curl \
        jq \
        git \
    && rm -rf /var/lib/apt/lists/*




# FROM base AS mofed-installer
# ARG MOFED_VERSION=23.10-2.1.3.1

# # In a container, we only need to install the user space libraries, though the drivers are still
# # needed on the host.
# # Note: MOFED's installation is not easily portable, so we can't copy the output of this stage
# # to our final stage, but must inherit from it. For that reason, we keep track of the build/install
# # only dependencies in the `MOFED_DEPS` variable (parsing the output of `--check-deps-only`) to
# # remove them in that same layer, to ensure they are not propagated in the final image.
# WORKDIR /opt/nvidia/mofed
# ARG MOFED_INSTALL_FLAGS="--dpdk --with-mft --user-space-only --force --without-fw-update"
# RUN UBUNTU_VERSION=$(cat /etc/lsb-release | grep DISTRIB_RELEASE | cut -d= -f2) \
#     && OFED_PACKAGE="MLNX_OFED_LINUX-${MOFED_VERSION}-ubuntu${UBUNTU_VERSION}-$(uname -m)" \
#     && curl -S -# -o ${OFED_PACKAGE}.tgz -L \
#         https://www.mellanox.com/downloads/ofed/MLNX_OFED-${MOFED_VERSION}/${OFED_PACKAGE}.tgz \
#     && tar xf ${OFED_PACKAGE}.tgz \
#     && MOFED_INSTALLER=$(find . -name mlnxofedinstall -type f -executable -print) \
#     && MOFED_DEPS=$(${MOFED_INSTALLER} ${MOFED_INSTALL_FLAGS} --check-deps-only 2>/dev/null | tail -n1 |  cut -d' ' -f3-) \
#     && apt-get update \
#     && apt-get install --no-install-recommends -y ${MOFED_DEPS} \
#     && ${MOFED_INSTALLER} ${MOFED_INSTALL_FLAGS} \
#     && rm -r * \
#     && apt-get remove -y ${MOFED_DEPS} && apt-get autoremove -y \
#     && rm -rf /var/lib/apt/lists/*

FROM base AS release
ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=xterm-256color

ARG GPU_TYPE
ARG UNAME
ARG UID
ARG GID

RUN mkdir -p /etc/holoscan/ \
        && mkdir -p /opt/holoscan/ \
        && mkdir -p /var/holoscan \
        && mkdir -p /opt/holoscan/app \
        && mkdir -p /var/holoscan/input \
        && mkdir -p /var/holoscan/output

LABEL base="nvcr.io/nvidia/cuda:12.6.0-runtime-ubuntu22.04"
LABEL tag="monet:1.0"
LABEL org.opencontainers.image.title="MONAI Deploy App Package - NIFTI nnUNet Seg Inference"
LABEL org.opencontainers.image.version="1.0"
LABEL org.nvidia.holoscan="3.1.0"

LABEL org.monai.deploy.app-sdk="2.0.0"

ENV HOLOSCAN_INPUT_PATH=/var/holoscan/input
ENV HOLOSCAN_OUTPUT_PATH=/var/holoscan/output
ENV HOLOSCAN_WORKDIR=/var/holoscan
ENV HOLOSCAN_APPLICATION=/opt/holoscan/app
ENV HOLOSCAN_TIMEOUT=0
ENV HOLOSCAN_MODEL_PATH=/opt/holoscan/models
ENV HOLOSCAN_DOCS_PATH=/opt/holoscan/docs
ENV HOLOSCAN_CONFIG_PATH=/var/holoscan/app.yaml
ENV HOLOSCAN_APP_MANIFEST_PATH=/etc/holoscan/app.json
ENV HOLOSCAN_PKG_MANIFEST_PATH=/etc/holoscan/pkg.json
ENV HOLOSCAN_LOGS_PATH=/var/holoscan/logs
ENV HOLOSCAN_VERSION=3.1.0









# If torch is installed, we can skip installing Python
ENV PYTHON_VERSION=3.10.6-1~22.04
ENV PYTHON_PIP_VERSION=22.0.2+dfsg-*

RUN apt update \
    && apt-get install -y --no-install-recommends --no-install-suggests \
        python3-minimal=${PYTHON_VERSION} \
        libpython3-stdlib=${PYTHON_VERSION} \
        python3=${PYTHON_VERSION} \
        python3-venv=${PYTHON_VERSION} \
        python3-pip=${PYTHON_PIP_VERSION} \
        python-is-python3 \
    && rm -rf /var/lib/apt/lists/*







RUN groupadd -f -g $GID $UNAME
RUN useradd -rm -d /home/$UNAME -s /bin/bash -g $GID -G sudo -u $UID $UNAME
RUN chown -R holoscan /var/holoscan && \
    chown -R holoscan /var/holoscan/input && \
    chown -R holoscan /var/holoscan/output

# Set the working directory
WORKDIR /var/holoscan

# Copy HAP/MAP tool script
COPY ./tools /var/holoscan/tools
RUN chmod +x /var/holoscan/tools

# Set the working directory
WORKDIR /var/holoscan
RUN mkdir /opt/holoscan/monailabel
RUN chmod -R 777 /opt/holoscan/monailabel
USER $UNAME

ENV PATH=/home/${UNAME}/.local/bin:/opt/nvidia/holoscan/bin:$PATH
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/${UNAME}/.local/lib/python3.10/site-packages/holoscan/lib

COPY ./pip/requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --user -r /tmp/requirements.txt


# Install MONAI Deploy App SDK

# Install MONAI Deploy from PyPI org
# RUN pip install monai-deploy-app-sdk==2.0.0
RUN pip install git+https://github.com/SimoneBendazzoli93/monai-deploy-app-sdk.git@8571947ed5bb9004f18c81eb94f2e017288d1d8d

ARG CACHE_BREAKER=1
RUN echo "Cache breaker: $CACHE_BREAKER"

RUN pip install --upgrade pip setuptools wheel

USER root
RUN apt-get install python3-distutils
RUN apt-get install python3-setuptools

USER $UNAME
COPY ./models  /opt/holoscan/models


COPY ./map/app.json /etc/holoscan/app.json
COPY ./app.config /var/holoscan/app.yaml
COPY ./map/pkg.json /etc/holoscan/pkg.json

COPY ./app /opt/holoscan/app

COPY Segmentation_Task_BraTS-MET-nnDet.yaml /etc/holoscan/Segmentation_Task.yaml
## Temporary fix for the issue with Holoscan
RUN sed -i 's/from holoscan.graphs import \*/from holoscan.core import \*/g' /home/${UNAME}/.local/lib/python3.10/site-packages/monai/deploy/graphs/__init__.py
# Fix Conditions
RUN sed -i 's/from holoscan.conditions import \*/from holoscan.core import \*/g' /home/${UNAME}/.local/lib/python3.10/site-packages/monai/deploy/conditions/__init__.py

# Fix Resources
RUN sed -i 's/from holoscan.resources import \*/from holoscan.core import \*/g' /home/${UNAME}/.local/lib/python3.10/site-packages/monai/deploy/resources/__init__.py

RUN pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --force-reinstall 
RUN pip install --upgrade nvidia-cuda-runtime-cu12
ENV LD_LIBRARY_PATH=/home/${UNAME}/.local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-venv \
    python3.8-dev \
    git \
    cmake \
    make \
    wget \
    gnupg \
    build-essential \
    software-properties-common \
    gdb \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

ARG env_det_num_threads=6
ARG env_det_verbose=1

# Setup environment variables
ENV det_data=/opt/data det_models=/opt/models det_num_threads=$env_det_num_threads det_verbose=$env_det_verbose OMP_NUM_THREADS=1
# 2. Create the venv specifically using the Python 3.8 binary
RUN /usr/bin/python3.8 -m venv /opt/venv/nndet

RUN /opt/venv/nndet/bin/pip install --upgrade pip setuptools wheel
RUN /opt/venv/nndet/bin/pip install numpy 
RUN /opt/venv/nndet/bin/pip install --upgrade requests urllib3

COPY ./requirements.txt ./nndet_requirements.txt
RUN mkdir ${det_data} \
  && mkdir ${det_models} \
  && mkdir -p /opt/code/nndet \
  && /opt/venv/nndet/bin/pip install -r nndet_requirements.txt  \
  && /opt/venv/nndet/bin/pip install hydra-core --upgrade --pre \
  && /opt/venv/nndet/bin/pip install git+https://github.com/mibaumgartner/pytorch_model_summary.git

WORKDIR /opt/code/
RUN git clone https://github.com/minnelab/nnDetection
WORKDIR /opt/code/nnDetection


RUN /opt/venv/nndet/bin/pip install pymaia-learn
RUN /opt/venv/nndet/bin/pip install monai
RUN /opt/venv/nndet/bin/pip install torch==1.11.0+cu113 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113 --force-reinstall

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common wget gnupg \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    # Fetch the NVIDIA key for Ubuntu 20.04
    && wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub | apt-key add - \
    # Add the Ubuntu 20.04 CUDA repository to sources
    && echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" > /etc/apt/sources.list.d/cuda-ubuntu2004.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    cmake \
    make \
    gdb \
    ninja-build \
    gcc-10 \
    g++-10 \
    cuda-toolkit-11-3 \
    && rm -rf /var/lib/apt/lists/*

# Force the symlink so setup.py finds the exact 11.3 compiler
RUN ln -sfn /usr/local/cuda-11.3 /usr/local/cuda

RUN CC=gcc-10 CXX=g++-10 TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" FORCE_CUDA=1 /opt/venv/nndet/bin/pip install -v -e .
RUN /opt/venv/nndet/bin/pip install typing-extensions==4.13.2
COPY ./nnDet_models /opt/models
RUN mkdir -p /opt/venv/nndet/lib/python3.8/site-packages/src/
COPY ./networks.py /opt/venv/nndet/lib/python3.8/site-packages/src/
RUN mkdir -p /opt/data/Task101_BraTS-MET-2026
RUN chmod -R 777 /opt/data/Task101_BraTS-MET-2026
RUN chmod -R 777 /opt/models/Task101_BraTS-MET-2026/RetinaUNetV001_D3V001_3d/consolidated
COPY ./MONet_concatenate_modalities.py /opt/code/nnDetection/MONet_concatenate_modalities.py
COPY ./start.sh /var/holoscan/start.sh
RUN chmod +x /var/holoscan/start.sh
COPY ./Segmentation_Task_BraTS-MET-nnDet.yaml /etc/holoscan/Segmentation_Task.yaml
RUN mkdir -p /opt/data/Task101_BraTS-MET-2026/raw_splitted/imagesTs
RUN chmod -R 777 /opt/data/Task101_BraTS-MET-2026/raw_splitted/imagesTs
USER $UNAME
RUN mkdir -p /home/holoscan/.config/matplotlib

#ENTRYPOINT ["/var/holoscan/tools"]
#ENTRYPOINT ["sleep", "infinity"]       
ENTRYPOINT ["/var/holoscan/start.sh"]
#/opt/venv/nndet/bin/nndet_predict 101 RetinaUNetV001_D3V001_3d --fold -1