ARG PYTORCH_VERSION="2.1.2"
ARG CUDA_VERSION="12.1"
ARG CUDNN_VERSION="8"

FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel

ARG TORCHVISION_VERSION="0.16.2"
ARG LIGHTNING_VERSION="2.1.3"
ARG TIMM_VERSION="0.9.12"
ARG TENSOBOARD_VERSION="2.13.0"
ARG TENSORFLOW_VERSION="2.13.1"

RUN useradd pretrain && \
    chown -R pretrain:pretrain /home

WORKDIR /home/pretrain
RUN mkdir -p /home/pretrain/output_dir

# Temporary: deal with broken GPG key rotation - remove repos
# Until resolution of https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN rm /etc/apt/sources.list.d/cuda.list

# Install torchvision
RUN CUDA_VERSION_TAG=$(python -c "print('cu' + ''.join('${CUDA_VERSION}'.split('.')[:2]) if '${CUDA_VERSION}' else 'cpu')") && \
    pip install \
        --no-cache-dir \
        --find-links https://download.pytorch.org/whl/torch_stable.html \
        torchvision==${TORCHVISION_VERSION}+${CUDA_VERSION_TAG}


# Install other packages
RUN pip install \
    --no-cache-dir \
    lightning==${LIGHTNING_VERSION} \
    timm==${TIMM_VERSION} \
    tensorboard==${TENSOBOARD_VERSION} \
    tensorflow==${TENSORFLOW_VERSION} \
    onnx

# Disable tensorflow warnings
ENV TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0

# Set entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Install lightly and requirements
COPY lightly /home/pretrain/lightly/lightly/
COPY requirements /home/pretrain/lightly/requirements/
COPY pyproject.toml README.md setup.py /home/pretrain/lightly/
RUN pip install \
    --no-cache-dir \
    -e /home/pretrain/lightly
