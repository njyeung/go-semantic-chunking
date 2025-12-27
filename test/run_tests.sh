#!/bin/bash
set -e

echo "Starting semantic-chunking-server..."
./semantic-chunking-server &
SERVER_PID=$!

echo "Server started with PID $SERVER_PID"
echo "Waiting for server to be ready..."

# Run the tests
python3 run_tests.py
TEST_EXIT_CODE=$?

# Kill the server
echo "Stopping server..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null || true

echo "Tests completed with exit code $TEST_EXIT_CODE"
exit $TEST_EXIT_CODE
