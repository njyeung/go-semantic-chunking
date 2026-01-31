# Go Semantic Chunking (Cloud Edition)

A semantic chunking server written in Go, using **Gemini cloud embeddings** for fast, GPU-free deployment.

> This branch uses the Gemini API for embeddings. For the local GPU version using ONNX Runtime, see the `main` branch.

## Quick Start

#### Using Docker

```bash
docker build -t semantic-chunking-server .

docker run -d --name semantic-server -p 8080:8080 \
  -e GEMINI_API_KEY=your-gemini-api-key \
  -e API_KEY=your-secret-api-key \
  semantic-chunking-server
```

#### Deploy to Google Cloud Run

```bash
# Enable required APIs (first time only)
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

# Deploy
gcloud run deploy semantic-chunking \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=your-gemini-key,API_KEY=your-secret-api-key
```

## Configuration

All configuration is done via environment variables:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | Yes | - | Your Gemini API key ([get one here](https://ai.google.dev/gemini-api/docs/api-key)) |
| `API_KEY` | No | - | Secret key to protect your endpoint |
| `GEMINI_MODEL` | No | `gemini-embedding-001` | Gemini embedding model to use |
| `EMBEDDING_DIMENSIONS` | No | `768` | Output embedding dimensions (768, 1536, or 3072) |
| `PORT` | No | `8080` | Server port |
| `READ_TIMEOUT_SECONDS` | No | `120` | HTTP read timeout |
| `WRITE_TIMEOUT_SECONDS` | No | `120` | HTTP write timeout |

## API Usage

### Authentication

If `API_KEY` is set, all requests to `/embed` must include the key via:
- **Header**: `X-API-Key: your-secret-key`
- **Query param**: `?api_key=your-secret-key`

The `/health` endpoint remains unauthenticated for health checks.

### Request Format

```bash
curl -X POST http://localhost:8080/embed \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-api-key" \
  -d '{
    "documents": [
      {
        "id": "doc1",
        "text": "First document text. These are separated by delimiters such as. Or? And!"
      },
      {
        "id": "doc2",
        "text": "Second document text.",
        "chunking_config": {
          "optimal_size": 300,
          "max_size": 400,
          "lambda_size": 1.5,
          "chunk_penalty": 2.0
        }
      }
    ]
  }'
```

### Response Format

```json
{
  "documents": [
    {
      "id": "doc1",
      "chunks": [
        {
          "text": "Chunk text here...",
          "embedding": [0.123, -0.456, ...],
          "num_sentences": 4,
          "token_count": 45,
          "chunk_index": 0
        }
      ],
      "error": ""
    },
    {
      "id": "doc2",
      "chunks": [...]
    }
  ]
}
```

### Health Check

```bash
curl http://localhost:8080/health
# Returns: ok
```

## Chunking Configuration

Each document can specify custom chunking parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `optimal_size` | 470 | Target chunk size in tokens, no penalty below this |
| `max_size` | 512 | Hard limit on chunk size in tokens |
| `lambda_size` | 2.0 | Maximum penalty at max_size |
| `chunk_penalty` | 1.0 | Per-chunk penalty to discourage over-splitting |

## Embedding Model

This server uses [Gemini's embedding API](https://ai.google.dev/gemini-api/docs/embeddings) with the `gemini-embedding-001` model.

**Key features:**
- 768, 1536, or 3072 dimensional embeddings
- Optimized for semantic similarity tasks
- No GPU required - runs anywhere

## Local Development

```bash
# Install dependencies
go mod download

# Set environment variables
export GEMINI_API_KEY=your-gemini-api-key
export API_KEY=your-secret-api-key  # optional

# Run the server
go run .
```

## Differences from GPU Version (main branch)

| Feature | Cloud (this branch) | GPU (main branch) |
|---------|---------------------|-------------------|
| Embedding | Gemini API | Local ONNX Runtime |
| Docker image | ~50MB (Alpine) | ~5GB (CUDA) |
| GPU required | No | Yes (or slow CPU fallback) |
| Cost | Pay per Gemini API call | Free after setup |
| Latency | Network dependent | Local inference |
| Setup | Just API key | CUDA + model download |

---

## How Semantic Chunking Works

The semantic chunking algorithm converts raw text into ~500 token chunks optimized for retrieval augmented generation (RAG). The algorithm balances three competing objectives:
1. **Semantic coherence**: Keep similar sentences together
2. **Chunk size**: Stay close to optimal token count
3. **Minimal fragmentation**: Avoid creating too many tiny chunks

### Preprocessing

Raw text is first segmented into sentences using standard delimiters (`.`, `?`, `!`).

**Token Limit Enforcement**: To ensure compatibility with downstream models, we enforce a hard maximum token limit (`max_size`) on all segments. In rare cases where a single sentence exceeds `max_size`, we greedily split it into consecutive chunks. After this step, all sentences satisfy `TokenCount <= max_size`.

### Dynamic Programming Algorithm

Given a sequence of embedded sentences, we use dynamic programming to find the optimal partition:

**DP Definition**:
```
dp[j] = best achievable score when chunking sentences [0, j)
dp[0] = 0  (zero sentences -> score of 0)
```

**Recurrence Relation**:
```
dp[j] = max{ i < j } (dp[i] + reward(i,j) - sizePenalty(i,j) - chunk_penalty)
```

Where:
- **`reward(i, j)`**: Sum of cosine similarities between adjacent sentences (favors semantic coherence)
- **`sizePenalty(i, j)`**: Smooth penalty as token count approaches `max_size`
- **`chunk_penalty`**: Constant penalty per chunk to discourage over-fragmentation

### Size Penalty Function

```
if tokenCount <= optimal_size:
    penalty = 0
else if tokenCount > max_size:
    penalty = infinity (illegal chunk)
else:
    normalized = (tokenCount - optimal_size) / (max_size - optimal_size)
    penalty = lambda_size * normalized
```

### Tuning Parameters

**For balanced RAG** (default):
```json
{
  "optimal_size": 470,
  "max_size": 512,
  "lambda_size": 2.0,
  "chunk_penalty": 1.0
}
```

- Small `chunk_penalty`: Allows isolated short chunks
- Large `optimal_size` + `max_size`: Longer, coherent chunks
- Large `chunk_penalty`: More evenly distributed chunk sizes
- `max_size` >> `optimal_size` with light `lambda_size`: Prioritizes semantic integrity

The best way to find optimal parameters is to test on documents from your specific use case.
