# Semantic Chunking Server - Test Suite

This directory contains comprehensive tests for the semantic chunking server.

## Test Overview

The test suite includes three tests that verify different chunking behaviors:

### Test 1: Basic Test (No Config)
- **Purpose**: Verify the server responds and handles multiple documents
- **Setup**: 3 documents with 3 short sentences each
- **Config**: Uses default parameters (optimal_size: 470, max_size: 512)
- **Expected**: Server responds successfully, creates chunks with default behavior

### Test 2: Small Chunks
- **Purpose**: Test chunking with aggressive size limits
- **Setup**: Single sample.txt (Declaration of Independence)
- **Config**: optimal_size=70, max_size=100, lambda_size=3.0, chunk_penalty=0.5
- **Expected**: Many small chunks (70-100 tokens each)

### Test 3: Huge Chunk + Medium Chunks
- **Purpose**: Test extreme parameter ranges with multiple documents
- **Setup**: Two copies of sample.txt with different configs
- **Config 1**: optimal_size=99999, max_size=99999, chunk_penalty=99999
  - Creates 1 huge chunk (or very few chunks) due to extreme penalty for creating new chunks
- **Config 2**: optimal_size=200, max_size=300, chunk_penalty=1.5
  - Creates medium-sized chunks (200-300 tokens)
- **Expected**: Doc 1 has 1-2 chunks, Doc 2 has multiple medium chunks

## Files

- `run_tests.py` - Python test script that sends HTTP requests and prints statistics
- `sample.txt` - Test data (Declaration of Independence)
- `Dockerfile` - Container environment for running tests with ONNX runtime
- `README.md` - This file

## Running the Tests

### Option 1: Using Docker (Recommended)

This is the recommended approach since it includes the ONNX runtime required by the server.

```bash
# From the test directory
cd test

# Build the test image
docker build -t semantic-chunking-test .

# Run tests with GPU support
docker run --rm --gpus all semantic-chunking-test

# Or run tests CPU only
docker run --rm semantic-chunking-test
```

The Dockerfile will:
1. Build the semantic-chunking-server
2. Start the server in the background
3. Run all tests
4. Print statistics about chunk formation
5. Stop the server
6. Exit with success/failure code

### Option 2: Running Tests Manually

If you already have the server running locally:

```bash
# Make sure the server is running on port 8080
# In another terminal, run:
cd test
python3 run_tests.py
```

**Requirements for manual testing:**
- Python 3 with `requests` library (`pip3 install requests`)
- Server running on http://localhost:8080

## Test Output

Each test prints:
- Number of chunks created
- Total tokens across all chunks
- Total sentences across all chunks
- Average, min, and max tokens per chunk
- Breakdown of each chunk (sentence count, token count, text preview)

Example output:
```
================================================================================
TEST 1: Basic test with 3 short documents (no config)
================================================================================
• Sending request with 3 documents...
• Each document has 3 short sentences
• Using default config (optimal_size: 470, max_size: 512)
✓ Request successful!

Statistics for doc1:
  Number of chunks: 1
  Total tokens: 42
  Total sentences: 3
  Average tokens/chunk: 42.0
  Min tokens: 42
  Max tokens: 42

Chunk breakdown:
  Chunk 0: 3 sentences, 42 tokens
    Preview: The quick brown fox jumps over the lazy dog. This is a test sentence. Another sentence here.
```

## Verifying Results

Since we cannot easily predict embedding model behavior, tests focus on:
1. **Server responsiveness**: All requests complete without errors
2. **Chunk statistics**: Number of chunks aligns with config parameters
3. **Visual inspection**: Review chunk sizes and text to verify semantic coherence

Expected behaviors to look for:
- **Test 1**: Few chunks per document (sentences are short and similar)
- **Test 2**: Many small chunks (70-100 tokens each)
- **Test 3 Doc 1**: 1-2 very large chunks (extreme chunk_penalty discourages splitting)
- **Test 3 Doc 2**: Several medium chunks (200-300 tokens)

## Troubleshooting

### Server not responding
- Increase timeout values in the Dockerfile: `READ_TIMEOUT_SECONDS` and `WRITE_TIMEOUT_SECONDS`
- Check if server started successfully by looking at Docker logs

### Out of memory errors
- Reduce `MAX_BATCH_TOKENS` in the Dockerfile
- Use CPU-only mode instead of GPU

### Tests fail but server responds
- Check the error messages in test output
- Verify sample.txt is present in the test directory
- Ensure test parameters are valid (max_size >= optimal_size, etc.)

## Adding New Tests

To add a new test:

1. Create a new function in `run_tests.py`:
```python
def run_test_4():
    """Test 4: Your test description."""
    print_section("TEST 4: Your test description")

    payload = {
        "documents": [
            {
                "id": "test_doc",
                "text": "Your test text...",
                "chunking_config": {
                    "optimal_size": 100,
                    "max_size": 200,
                    "lambda_size": 2.0,
                    "chunk_penalty": 1.0
                }
            }
        ]
    }

    # ... rest of test implementation
```

2. Add to the `main()` function:
```python
results.append(("Test 4: Description", run_test_4()))
```

## Parameter Reference

From [config.go](../config.go):

- **optimal_size** (default: 470): Target chunk size in tokens, no penalty below this
- **max_size** (default: 512): Hard limit on chunk size in tokens
- **lambda_size** (default: 2.0): Maximum penalty at max_size
- **chunk_penalty** (default: 1.0): Per-chunk penalty to discourage over-splitting

See the main [README.md](../README.md) for detailed information on how these parameters affect chunking behavior.
