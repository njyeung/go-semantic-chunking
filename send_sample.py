#!/usr/bin/env python3
"""
Send sample.txt to the semantic chunking server with default parameters.
"""

import requests
import json
import os

# Server endpoint
URL = "http://localhost:8080/embed"

# Read sample.txt
SAMPLE_FILE = "sample.txt"

if not os.path.exists(SAMPLE_FILE):
    print(f"❌ Error: {SAMPLE_FILE} not found!")
    exit(1)

with open(SAMPLE_FILE, 'r') as f:
    text = f.read()

if not text.strip():
    print(f"❌ Error: {SAMPLE_FILE} is empty!")
    exit(1)

print(f"Read {len(text)} characters from {SAMPLE_FILE}")
print("-" * 60)

# Prepare request with DEFAULT parameters (no chunking_config)
payload = {
    "documents": [
        {
            "id": "sample_doc",
            "text": text
            # No chunking_config = uses server defaults
        }
    ]
}

try:
    print("Sending request to server...")
    response = requests.post(URL, json=payload, timeout=60)
    response.raise_for_status()

    result = response.json()

    # Extract document response
    doc = result["documents"][0]

    if doc.get("error"):
        print(f"❌ Error from server: {doc['error']}")
        exit(1)

    chunks = doc["chunks"]
    num_chunks = len(chunks)

    print(f"✅ Success! Received {num_chunks} chunk(s)")
    print("=" * 60)

    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i}:")
        print(f"  Number of sentences: {chunk['num_sentences']}")
        print(f"  Token count: {chunk['token_count']}")
        print(f"  Embedding dimensions: {len(chunk['embedding'])}")
        print(f"  Text: {chunk['text']}")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total chunks: {num_chunks}")
    print(f"Total tokens: {sum(c['token_count'] for c in chunks)}")
    print(f"Total sentences: {sum(c['num_sentences'] for c in chunks)}")

except requests.exceptions.ConnectionError:
    print(f"❌ Error: Could not connect to server at {URL}")
    print("   Make sure the server is running on port 8080")
    exit(1)
except requests.exceptions.Timeout:
    print("❌ Error: Request timed out (took more than 60 seconds)")
    exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)
