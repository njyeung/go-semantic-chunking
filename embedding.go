package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	"google.golang.org/genai"
)

// EmbeddingModel manages the Gemini embedding client
type EmbeddingModel struct {
	client *genai.Client
	config EmbeddingConfig
	ctx    context.Context
}

// InitEmbeddingModel initializes the Gemini embedding client
func InitEmbeddingModel(config EmbeddingConfig) (*EmbeddingModel, error) {
	ctx := context.Background()

	// Get API key from environment
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("GEMINI_API_KEY environment variable is required")
	}

	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create Gemini client: %w", err)
	}

	fmt.Println("Gemini embedding client initialized successfully")
	fmt.Printf("Using model: %s, dimensions: %d\n", config.Model, config.OutputDimensionality)

	return &EmbeddingModel{
		client: client,
		config: config,
		ctx:    ctx,
	}, nil
}

// EmbedSentences embeds a slice of Sentence structs
func (em *EmbeddingModel) EmbedSentences(sentences []*Sentence) error {
	if len(sentences) == 0 {
		return nil
	}

	texts := make([]string, len(sentences))
	for i, s := range sentences {
		texts[i] = s.Text
	}

	embeddings, err := em.embed(texts)
	if err != nil {
		return err
	}

	for i, emb := range embeddings {
		sentences[i].Embedding = emb
	}
	return nil
}

// EmbedChunks embeds a slice of Chunk structs (updates Embedding field in place)
func (em *EmbeddingModel) EmbedChunks(chunks []*Chunk) error {
	if len(chunks) == 0 {
		return nil
	}

	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Text
	}

	embeddings, err := em.embed(texts)
	if err != nil {
		return err
	}

	for i, emb := range embeddings {
		chunks[i].Embedding = emb
	}
	return nil
}

// embed processes texts using Gemini API
func (em *EmbeddingModel) embed(texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return [][]float32{}, nil
	}

	// Build contents for batch embedding
	contents := make([]*genai.Content, len(texts))
	for i, text := range texts {
		contents[i] = genai.NewUserContentFromText(text)
	}

	// Configure embedding request
	outputDim := int32(em.config.OutputDimensionality)
	config := &genai.EmbedContentConfig{
		TaskType:             "SEMANTIC_SIMILARITY",
		OutputDimensionality: &outputDim,
	}

	result, err := em.client.Models.EmbedContent(em.ctx, em.config.Model, contents, config)
	if err != nil {
		return nil, fmt.Errorf("embedding request failed: %w", err)
	}

	if len(result.Embeddings) != len(texts) {
		return nil, fmt.Errorf("expected %d embeddings, got %d", len(texts), len(result.Embeddings))
	}

	// Convert to [][]float32 and normalize
	embeddings := make([][]float32, len(result.Embeddings))
	for i, emb := range result.Embeddings {
		embeddings[i] = normalizeEmbedding(emb.Values)
	}

	return embeddings, nil
}

// normalizeEmbedding normalizes an embedding vector to unit length
// This is important for non-3072 dimensions as per Gemini docs
func normalizeEmbedding(embedding []float32) []float32 {
	var sumSquares float64
	for _, v := range embedding {
		sumSquares += float64(v) * float64(v)
	}

	if sumSquares == 0 {
		return embedding
	}

	norm := float32(1.0 / sqrt(sumSquares))
	normalized := make([]float32, len(embedding))
	for i, v := range embedding {
		normalized[i] = v * norm
	}
	return normalized
}

// sqrt computes square root (avoiding math import for just this)
func sqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	z := x
	for i := 0; i < 10; i++ {
		z = z - (z*z-x)/(2*z)
	}
	return z
}

// CountTokens estimates token count for a text
// This is a rough approximation since we don't have the exact tokenizer
// Gemini uses ~4 chars per token on average for English text
func CountTokens(text string) int {
	// Simple heuristic: count words and add ~30% for subword tokenization
	words := len(strings.Fields(text))
	chars := len(text)

	// Use character-based estimate, averaging ~4 chars per token
	charBasedEstimate := (chars + 3) / 4

	// Use word-based estimate with 1.3x multiplier for subwords
	wordBasedEstimate := int(float64(words) * 1.3)

	// Return the larger estimate to be conservative
	if charBasedEstimate > wordBasedEstimate {
		return charBasedEstimate
	}
	return wordBasedEstimate
}

// Close releases resources
func (em *EmbeddingModel) Close() error {
	// The genai.Client doesn't have a Close method in this version
	return nil
}
