package main

import (
	"os"
	"strconv"
	"time"
)

// ServerConfig holds HTTP server configuration
type ServerConfig struct {
	Port         string
	ReadTimeout  time.Duration
	WriteTimeout time.Duration
}

// ChunkingConfig holds all tunable parameters for the semantic chunking algorithm
type ChunkingConfig struct {
	OptimalSize  int     `json:"optimal_size"`  // optimal chunk size, no penalty below this (default: 470)
	MaxSize      int     `json:"max_size"`      // chunk size hard limit, infinite penalty at or above (default: 512)
	LambdaSize   float32 `json:"lambda_size"`   // Max penalty in "edge units" at MaxSize (default: 3.0)
	ChunkPenalty float32 `json:"chunk_penalty"` // Initial penalty per chunk to discourage small chunks (default: 1.0)
}

// EmbeddingConfig holds embedding model configuration
type EmbeddingConfig struct {
	MaxBatchTokens int // Max total tokens per batch (controls GPU memory usage)
}

// LoadServerConfig loads server configuration from environment variables
// Falls back to defaults if not set
func LoadServerConfig() ServerConfig {
	port := "8080"
	readTimeout := 120 * time.Second
	writeTimeout := 120 * time.Second

	if envVal := os.Getenv("PORT"); envVal != "" {
		port = envVal
	}

	if envVal := os.Getenv("READ_TIMEOUT_SECONDS"); envVal != "" {
		if val, err := strconv.Atoi(envVal); err == nil && val > 0 {
			readTimeout = time.Duration(val) * time.Second
		}
	}

	if envVal := os.Getenv("WRITE_TIMEOUT_SECONDS"); envVal != "" {
		if val, err := strconv.Atoi(envVal); err == nil && val > 0 {
			writeTimeout = time.Duration(val) * time.Second
		}
	}

	return ServerConfig{
		Port:         port,
		ReadTimeout:  readTimeout,
		WriteTimeout: writeTimeout,
	}
}

// LoadEmbeddingConfig loads embedding configuration from environment variables
// Falls back to defaults if not set
func LoadEmbeddingConfig() EmbeddingConfig {
	maxBatchTokens := 6000 // default

	if envVal := os.Getenv("MAX_BATCH_TOKENS"); envVal != "" {
		if val, err := strconv.Atoi(envVal); err == nil && val > 0 {
			maxBatchTokens = val
		}
	}

	return EmbeddingConfig{
		MaxBatchTokens: maxBatchTokens,
	}
}

// DefaultEmbeddingConfig returns sensible defaults for embedding
func DefaultEmbeddingConfig() EmbeddingConfig {
	return EmbeddingConfig{
		// 12000 tokens is about:
		// 240 short sentences (50 tokens each) in one batch
		// 24 medium chunks (500 tokens each) in one batch
		// 12 large chunks (1000 tokens each) in one batch
		MaxBatchTokens: 6000,
	}
}

// DefaultChunkingConfig returns sensible defaults
func DefaultChunkingConfig() ChunkingConfig {
	return ChunkingConfig{
		OptimalSize:  470,
		MaxSize:      512,
		LambdaSize:   2.0,
		ChunkPenalty: 1.0,
	}
}
