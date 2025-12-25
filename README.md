# Go Semantic Chunking

A semantic chunking algorithm written (mostly) in Go, with a quickly deployable server.

## Quick Start

#### Using Docker (Recommended)

```bash
docker build -t semantic-chunking-server .

# Run with GPU support
docker run -d --name semantic-server --gpus all -p 8080:8080 semantic-chunking-server

# or run CPU only
docker run -d --name semantic-server -p 8080:8080 semantic-chunking-server
```

#### Configuration

All configuration is done via environment variables. See the `Dockerfile` for detailed documentation on available options including:
- Server timeouts
- Batch token limits
- Port configuration

Override defaults when running:
```bash
docker run -d --gpus all -p 8080:8080 \
  -e MAX_BATCH_TOKENS=12000 \
  -e READ_TIMEOUT_SECONDS=180 \
  -e WRITE_TIMEOUT_SECONDS=180 \
  semantic-chunking-server
```

> The Dockerfile also contains useful information for local setup, such as installing an ONNX runtime library and downloading an embedding model

## API Usage

The API supports batch processing by default. Each document is a string of text that is processed independently. Documents are processed sequentially and have no effect on one another.

#### Request Format

```
curl -X POST http://localhost:8080/embed \
  -H "Content-Type: application/json" \
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

#### Response Format

```
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

## Chunking Configuration

Each document can specify custom chunking parameters:

- **optimal_size** (default: 470): Target chunk size in tokens, no penalty below this
- **max_size** (default: 512): Hard limit on chunk size in tokens
- **lambda_size** (default: 5.0): Maximum penalty at max_size
- **chunk_penalty** (default: 1.0): Per-chunk penalty to discourage over-splitting

> More information on how parameters affect chunking below

## Embedding Model

This server uses the [gte-large-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5) model from Alibaba-NLP. The ONNX model and vocab.txt are automatically downloaded during the Docker build, while a custom tokenizer.json is included in the repository.

#### Bring Your Own Embedding Model
1. Replace the model download commands in the Dockerfile with your model URL
2. Ensure your model has the same input/output format (input_ids, attention_mask, token_type_ids and last_hidden_state)
3. Provide compatible tokenizer files (tokenizer.json and vocab.txt)

## Local Setup (without Docker)

Requirements:
- Go 1.21+
- ONNX Runtime (GPU or CPU version)
- CUDA (for GPU support)

```bash
# Install dependencies
go mod download

# Download model and vocab
wget https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5/resolve/main/onnx/model.onnx
wget https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5/resolve/main/vocab.txt
# Note: tokenizer.json is included in the repository

# Set ONNX Runtime library path
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Run the server
go run .
```

---

## How Semantic Chunking Works

This section provides a detailed explanation of the semantic chunking algorithm for readers interested in understanding how the chunking parameters affect output.

#### Overview

The semantic chunking algorithm converts raw text into ~500 token chunks optimized for retrieval augmented generation (RAG). The algorithm balances three competing objectives:
1. **Semantic coherence**: Keep similar sentences together
2. **Chunk size**: Stay close to optimal token count
3. **Minimal fragmentation**: Avoid creating too many tiny chunks

#### Preprocessing

Raw text is first segmented into sentences using standard delimiters (`.`, `?`, `!`).

**Token Limit Enforcement**: To ensure compatibility with downstream models, we enforce a hard maximum token limit (`max_size`) on all segments. In rare cases where a single sentence exceeds `max_size`, we greedily split it into consecutive chunks, with the final chunk containing remaining tokens. These chunks may end mid-sentence, but this is acceptable for RAG applications. After this step, all sentences satisfy `TokenCount ≤ max_size`.

#### Problem Formulation

Given a sequence of embedded sentences:
```
s0, s1, s2, ..., s{n-1}
```

The goal is to partition them into contiguous chunks such that:
- Semantically coherent sentences are grouped together (cosine similarity)
- Chunk sizes remain close to `optimal_size`
- No chunk exceeds `max_size`
- Excessive fragmentation is discouraged (`chunk_penalty`)

We solve this segmentation problem via dynamic programming.

#### Sentence Similarity Precomputation

1. Each sentence is embedded exactly once using the ONNX model
2. For each adjacent pair `(sᵢ, sᵢ₊₁)`, we compute cosine similarity:
   ```
   sim[i] = cos(embed(s_i), embed(s_{i+1}))
   ```
3. All similarities are **min-max normalized** to `[0, 1]` to ensure:
   - Rewards are always non-negative
   - There's always reward for merging sentences
   - Higher similarity always increases merge desirability

#### Optimization for O(1) Scoring

To enable efficient DP transitions, we precompute two prefix arrays:

**Prefix Similarity Array** (`prefix_sim`):
```
prefix_sim[k] = sim[0] + sim[1] + ... + sim[k-1]
```
This allows computing the total similarity within any segment `[i, j)` in O(1):
```
segment_similarity(i, j) = prefix_sim[j-1] - prefix_sim[i]
```

**Prefix Token Array** (`prefix_tokens`):
```
prefix_tokens[k] = tokens[0] + tokens[1] + ... + tokens[k-1]
```
This allows computing total tokens in segment `[i, j)` in O(1):
```
segment_tokens(i, j) = prefix_tokens[j] - prefix_tokens[i]
```

#### Dynamic Programming Algorithm

**DP Definition**:
```
dp[j] = best achievable score when chunking sentences [0, j)
dp[0] = 0  (zero sentences → score of 0)
```

**Recurrence Relation**:
```
dp[j] = max{ i < j } (dp[i] + reward(i,j) - sizePenalty(i,j) - chunk_penalty)
```

Where:
- **`reward(i, j)`**: Sum of cosine similarities between adjacent sentences in `[i, j)` (always positive, favors semantic coherence)
- **`sizePenalty(i, j)`**: Smooth penalty as token count approaches `max_size`, becomes infinite if exceeded
- **`chunk_penalty`**: Constant penalty per chunk to discourage over-fragmentation

**Legal Segments**: Only segments with total token count ≤ `max_size` are considered.

**Reconstruction**: A `start[]` array tracks the optimal starting index for each position, allowing backtracking from `dp[n]` to `dp[0]` to reconstruct the optimal chunking.

#### Size Penalty Function

The `sizePenalty` is a hinge-like function parameterized by:
- **`optimal_size`**: No penalty below this threshold
- **`max_size`**: Hard upper bound (infinite penalty if exceeded)
- **`lambda_size`**: Maximum penalty applied at `max_size`

**Formula**:
```go
if tokenCount <= optimal_size:.
    penalty = 0
else if tokenCount > max_size:
    penalty = ∞ (illegal chunk)
else:
    normalized = (tokenCount - optimal_size) / (max_size - optimal_size)
    penalty = lambda_size × normalized
```

This encourages chunks near `optimal_size` while allowing flexibility when semantic coherence warrants larger chunks.

#### Reconstruction

Once `dp[n]` is computed, we reconstruct the optimal segmentation by backtracking through `start[]` from `n` to `0`. Chunks are built in reverse order, then reversed to restore the original sequence.

Each chunk aggregates:
- Sentence text (concatenated)
- Sentence embeddings
- Total token count
- Chunk index

Finally, each chunk is embedded one final time to produce the chunk-level embedding returned in the response.

---

#### Tuning Parameters for Your Use Case

Small/No `chunk_penalty` encourages isolated chunks, phrases such as "Alright" and "Okay". This may be useful if you add a processing step afterwards to weed chunks smaller than 30 tokens.   

For longer, coherent chunks, a large `optimal_size` and `max_size` may be desireable.

For evenly distrbuted chunks of about the same token length, a large `chunk_penalty` would be helpful.

If varying token lengths is acceptable and strong semantic integrity within chunks is a priority, then having `max_size` >> `optimal_size` and a light `lambda_size` allows for chunks to grow against the penalty as long as the sentences within them are very similar.

**For balanced RAG** (default):
```json
{
  "optimal_size": 470,
  "max_size": 512,
  "lambda_size": 2.0,
  "chunk_penalty": 1.0
}
```

Ultimately, the best way to find optimal parameters is to test on documents from your specific use case and visually validate the output.