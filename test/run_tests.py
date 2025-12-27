#!/usr/bin/env python3
"""
Comprehensive test suite for the semantic chunking server.

Tests:
1. Basic test with 3 short documents (3 sentences each), no config
2. Single sample.txt with custom config (small chunks)
3. Two sample.txt documents with custom config (one huge chunk)
"""

import requests
import json
import os
import sys
import time

# Server endpoint
URL = "http://localhost:8080/embed"

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"{BOLD}{BLUE}{title}{RESET}")
    print("=" * 80)

def print_success(msg):
    """Print a success message."""
    print(f"{GREEN}✓ {msg}{RESET}")

def print_error(msg):
    """Print an error message."""
    print(f"{RED}✗ {msg}{RESET}")

def print_info(msg):
    """Print an info message."""
    print(f"{YELLOW}• {msg}{RESET}")

def print_stats(chunks, doc_id):
    """Print statistics about chunks."""
    num_chunks = len(chunks)
    total_tokens = sum(c['token_count'] for c in chunks)
    total_sentences = sum(c['num_sentences'] for c in chunks)
    avg_tokens = total_tokens / num_chunks if num_chunks > 0 else 0
    min_tokens = min(c['token_count'] for c in chunks) if num_chunks > 0 else 0
    max_tokens = max(c['token_count'] for c in chunks) if num_chunks > 0 else 0

    print(f"\n{BOLD}Statistics for {doc_id}:{RESET}")
    print(f"  Number of chunks: {num_chunks}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Total sentences: {total_sentences}")
    print(f"  Average tokens/chunk: {avg_tokens:.1f}")
    print(f"  Min tokens: {min_tokens}")
    print(f"  Max tokens: {max_tokens}")

    print(f"\n{BOLD}Chunk breakdown:{RESET}")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {chunk['num_sentences']} sentences, {chunk['token_count']} tokens")
        # Show first 100 chars of text
        text_preview = chunk['text'][:100] + "..." if len(chunk['text']) > 100 else chunk['text']
        print(f"    Preview: {text_preview}")

def wait_for_server(max_retries=30, delay=2):
    """Wait for server to be ready."""
    print_info("Waiting for server to be ready...")
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8080", timeout=1)
            print_success("Server is ready!")
            return True
        except requests.exceptions.RequestException:
            if i < max_retries - 1:
                time.sleep(delay)
            else:
                print_error(f"Server did not become ready after {max_retries * delay} seconds")
                return False
    return False

def run_test_1():
    """Test 1: Basic test with 3 short documents (3 sentences each), no config."""
    print_section("TEST 1: Basic test with 3 short documents (no config)")

    payload = {
        "documents": [
            {
                "id": "doc1",
                "text": "The quick brown fox jumps over the lazy dog. This is a test sentence. Another sentence here."
            },
            {
                "id": "doc2",
                "text": "Python is a great programming language. It has many useful libraries. The syntax is very clean."
            },
            {
                "id": "doc3",
                "text": "Machine learning is fascinating. Neural networks can learn complex patterns. Deep learning has revolutionized AI."
            }
        ]
    }

    print_info(f"Sending request with {len(payload['documents'])} documents...")
    print_info("Each document has 3 short sentences")
    print_info("Using default config (optimal_size: 470, max_size: 512)")

    try:
        response = requests.post(URL, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()

        print_success(f"Request successful!")

        for doc in result["documents"]:
            if doc.get("error"):
                print_error(f"Error for {doc['id']}: {doc['error']}")
            else:
                print_stats(doc["chunks"], doc["id"])

        return True

    except Exception as e:
        print_error(f"Test 1 failed: {e}")
        return False

def run_test_2():
    """Test 2: Single sample.txt with custom config (small chunks)."""
    print_section("TEST 2: Single sample.txt with custom config (small chunks)")

    sample_file = "sample.txt"
    if not os.path.exists(sample_file):
        print_error(f"{sample_file} not found!")
        return False

    with open(sample_file, 'r') as f:
        text = f.read()

    # Config for small chunks: low max_size and optimal_size
    payload = {
        "documents": [
            {
                "id": "sample_small_chunks",
                "text": text,
                "chunking_config": {
                    "optimal_size": 70,
                    "max_size": 100,
                    "lambda_size": 3.0,
                    "chunk_penalty": 0.5
                }
            }
        ]
    }

    print_info(f"Sending request with sample.txt ({len(text)} characters)")
    print_info("Config: optimal_size=70, max_size=100, lambda_size=3.0, chunk_penalty=0.5")
    print_info("This should create many small chunks")

    try:
        response = requests.post(URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()

        print_success(f"Request successful!")

        for doc in result["documents"]:
            if doc.get("error"):
                print_error(f"Error for {doc['id']}: {doc['error']}")
            else:
                print_stats(doc["chunks"], doc["id"])

        return True

    except Exception as e:
        print_error(f"Test 2 failed: {e}")
        return False

def run_test_3():
    """Test 3: Two sample.txt documents with custom config (one huge chunk)."""
    print_section("TEST 3: Two sample.txt documents with different configs")

    sample_file = "sample.txt"
    if not os.path.exists(sample_file):
        print_error(f"{sample_file} not found!")
        return False

    with open(sample_file, 'r') as f:
        text = f.read()

    # Config for huge chunks: very high max_size and optimal_size, high chunk_penalty
    payload = {
        "documents": [
            {
                "id": "sample_huge_chunk",
                "text": text,
                "chunking_config": {
                    "optimal_size": 99999,
                    "max_size": 99999,
                    "lambda_size": 1.0,
                    "chunk_penalty": 99999.0
                }
            },
            {
                "id": "sample_medium_chunks",
                "text": text,
                "chunking_config": {
                    "optimal_size": 200,
                    "max_size": 300,
                    "lambda_size": 2.0,
                    "chunk_penalty": 1.5
                }
            }
        ]
    }

    print_info(f"Sending request with 2 copies of sample.txt ({len(text)} characters each)")
    print_info("Doc 1 config: optimal_size=99999, max_size=99999, chunk_penalty=99999")
    print_info("  -> This should create 1 huge chunk (or very few chunks)")
    print_info("Doc 2 config: optimal_size=200, max_size=300, chunk_penalty=1.5")
    print_info("  -> This should create medium-sized chunks")

    try:
        response = requests.post(URL, json=payload, timeout=180)
        response.raise_for_status()
        result = response.json()

        print_success(f"Request successful!")

        for doc in result["documents"]:
            if doc.get("error"):
                print_error(f"Error for {doc['id']}: {doc['error']}")
            else:
                print_stats(doc["chunks"], doc["id"])

        return True

    except Exception as e:
        print_error(f"Test 3 failed: {e}")
        return False

def main():
    """Run all tests."""
    print(f"{BOLD}Semantic Chunking Server - Test Suite{RESET}")

    # Wait for server to be ready
    if not wait_for_server():
        print_error("Server is not ready. Exiting.")
        sys.exit(1)

    # Run all tests
    results = []

    results.append(("Test 1: Basic 3 documents", run_test_1()))
    time.sleep(1)  # Brief pause between tests

    results.append(("Test 2: Small chunks", run_test_2()))
    time.sleep(1)

    results.append(("Test 3: Huge chunk + medium chunks", run_test_3()))

    # Print summary
    print_section("TEST SUMMARY")
    all_passed = True
    for test_name, passed in results:
        if passed:
            print_success(f"{test_name}: PASSED")
        else:
            print_error(f"{test_name}: FAILED")
            all_passed = False

    if all_passed:
        print(f"\n{GREEN}{BOLD}All tests passed!{RESET}")
        sys.exit(0)
    else:
        print(f"\n{RED}{BOLD}Some tests failed!{RESET}")
        sys.exit(1)

if __name__ == "__main__":
    main()
