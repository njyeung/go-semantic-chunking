# Cloud Embeddings Version
# Uses Gemini API instead of local GPU inference

FROM golang:1.21-alpine

RUN apk add --no-cache ca-certificates

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY *.go ./
RUN go build -o semantic-chunking-server .


                    # CONFIGURE SERVER USING THESE ENVIRONMENT VARIABLES

# SERVER RESPONSE TIMEOUTS
# (if you plan to do large batches, increase)
ENV READ_TIMEOUT_SECONDS=120
ENV WRITE_TIMEOUT_SECONDS=120

# Gemini API Configuration
# GEMINI_API_KEY must be set at runtime
ENV GEMINI_MODEL=gemini-embedding-001
ENV EMBEDDING_DIMENSIONS=1024

# PUT YOUR GEMINI KEY HERE
ENV GEMINI_API_KEY=

# Change server port
ENV PORT=8080
EXPOSE 8080


CMD ["./semantic-chunking-server"]
