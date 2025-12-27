# Test Dockerfile for semantic-chunking-server
# This builds the server and runs tests against it

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    curl \
    ca-certificates \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Go
RUN wget -q https://go.dev/dl/go1.21.0.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go1.21.0.linux-amd64.tar.gz && \
    rm go1.21.0.linux-amd64.tar.gz

ENV PATH=/usr/local/go/bin:$PATH

# Install ONNX Runtime C/C++ library (for GPU)
RUN mkdir -p /tmp/onnxrt && \
    cd /tmp/onnxrt && \
    wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-gpu-1.23.2.tgz && \
    tar -xzf onnxruntime-linux-x64-gpu-1.23.2.tgz && \
    cd onnxruntime-linux-x64-gpu-1.23.2 && \
    cp -r include/* /usr/local/include/ && \
    cp -r lib/* /usr/local/lib/ && \
    ldconfig && \
    cd / && \
    rm -rf /tmp/onnxrt

# Make sure loader can find libonnxruntime.so
ENV LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

ENV CGO_ENABLED=1

WORKDIR /app

# Download the ONNX model
RUN wget -q https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5/resolve/main/onnx/model.onnx

# Download vocab.txt (required by tokenizer)
RUN wget -q https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5/resolve/main/vocab.txt

# Copy the modified tokenizer from the repo
COPY ../tokenizer.json ./

# Copy Go source files
COPY ../go.mod ../go.sum ./
RUN go mod download
COPY ../*.go ./

# Build the server
RUN go build -o semantic-chunking-server .

# Install Python dependencies for tests
RUN pip3 install requests

# Copy test files
COPY test/run_tests.py ./
COPY test/run_tests.sh ./
COPY test/sample.txt ./

# Server configuration
ENV READ_TIMEOUT_SECONDS=180
ENV WRITE_TIMEOUT_SECONDS=180
ENV MAX_BATCH_TOKENS=6000
ENV PORT=8080

CMD ["/app/run_tests.sh"]
