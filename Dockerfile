# Use the NVIDIA CUDA runtime base image
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PATH="/usr/local/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Update and install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    git \
    libopenmpi-dev \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libopenblas-dev \
    libblas-dev \
    liblapack-dev \
    libhdf5-serial-dev \
    libcurl4-openssl-dev \
    cmake \
    make \
    g++ \
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-perftools-dev \
    python3 \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools

RUN python3 -m pip install --no-cache-dir \
    torch==1.13.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117 && \
    python3 -m pip install --no-cache-dir \
    numpy==1.22.3 \
    scipy==1.7.3 \
    pandas==1.5.3 \
    scikit-learn==1.2.0 \
    matplotlib==3.6.3 \
    jupyter==1.0.0 \
    ipython==8.9.0 \
    ipykernel==6.19.2 \
    tensorflow==2.9.1 \
    tensorboard==2.9.0 \
    transformers==4.26.1 \
    datasets==2.9.0 \
    evaluate==0.4.0 \
    tokenizers==0.13.2 \
    huggingface-hub==0.12.0 \
    pyarrow==11.0.0 \
    pillow==9.4.0 \
    tqdm==4.64.1 \
    requests==2.28.1 \
    protobuf==3.19.0 \
    pyyaml==6.0 \
    typing-extensions==4.4.0 \
    packaging==22.0 \
    filelock==3.9.0 \
    regex==2022.10.31 \
    ninja==1.11.1 \
    psutil==5.9.0 \
    py-cpuinfo==9.0.0 \
    pyparsing==3.0.9 \
    pydantic==1.10.5 \
    typeguard==2.13.3 \
    xxhash==3.2.0

RUN git clone https://github.com/google/sentencepiece.git /sentencepiece && \
    cd /sentencepiece && \
    mkdir build && cd build && \
    cmake .. -DSPM_ENABLE_SHARED=OFF -DCMAKE_INSTALL_PREFIX=./root && \
    make install && \
    cd ../python && \
    python3 setup.py bdist_wheel && \
    python3 -m pip install dist/sentencepiece*.whl && \
    rm -rf /sentencepiece

# Clean up pip cache
RUN python3 -m pip cache purge

# Default command to run Python scripts
CMD ["python3"]
