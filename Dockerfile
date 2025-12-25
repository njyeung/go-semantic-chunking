# For CUDA GPUs, however will fall back to cpu

                    # CUDA UBUNTU
                    # for cpu only, you can use a
                    # regular ubuntu image

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04


RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    curl \
    ca-certificates \
    nano \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Go
RUN wget -q https://go.dev/dl/go1.21.0.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go1.21.0.linux-amd64.tar.gz && \
    rm go1.21.0.linux-amd64.tar.gz

ENV PATH=/usr/local/go/bin:$PATH


                    # REQUIRES AN ONNX RUNTIME
                    # You may bring your own runtime if you're
                    # planning on running this project locally

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


                    # YOU MAY BRING YOUR OWN EMBEDDING MODEL
                    # AND TOKENIZER
                    # here, we are using gte-large-en-v1.5

# Download the ONNX model
RUN wget -q https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5/resolve/main/onnx/model.onnx

# Download vocab.txt (required by tokenizer)
RUN wget -q https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5/resolve/main/vocab.txt

# Copy the modified tokenizer from the repo
COPY tokenizer.json ./

COPY go.mod go.sum ./
RUN go mod download
COPY *.go ./
RUN go build -o semantic-chunking-server .


                    # CONFIGURE SERVER USING THESE ENVIRONMENT VARIABLES

# SERVER RESPONSE TIMEOUTS
# (if you plan to do large batches, increase) 
ENV READ_TIMEOUT_SECONDS=120
ENV WRITE_TIMEOUT_SECONDS=120

# CHANGE BASED ON THE AMOUNT OF MEMORY YOU HAVE 
# (6000 is comfortable on an 8gb GPU)
ENV MAX_BATCH_TOKENS=6000

# Change server port
ENV PORT=8080
EXPOSE 8080


CMD ["./semantic-chunking-server"]
