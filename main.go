package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
)

// EmbedRequest represents a single document to embed
type EmbedRequest struct {
	ID             string          `json:"id,omitempty"`
	Text           string          `json:"text"`
	ChunkingConfig *ChunkingConfig `json:"chunking_config,omitempty"`
}

// BatchEmbedRequest represents the HTTP request body supporting both single and batch
type BatchEmbedRequest struct {
	Documents []EmbedRequest `json:"documents"`
}

// ChunkResponse represents a chunk in the HTTP response
type ChunkResponse struct {
	Text         string    `json:"text"`
	StartTime    string    `json:"start_time,omitempty"`
	Embedding    []float32 `json:"embedding"`
	NumSentences int       `json:"num_sentences"`
	TokenCount   int       `json:"token_count"`
	ChunkIndex   int       `json:"chunk_index"`
}

// DocumentResponse represents the response for a single document
type DocumentResponse struct {
	ID     string          `json:"id,omitempty"`
	Chunks []ChunkResponse `json:"chunks"`
	Error  string          `json:"error,omitempty"`
}

// BatchEmbedResponse represents the HTTP response body
type BatchEmbedResponse struct {
	Documents []DocumentResponse `json:"documents"`
}

func main() {
	// Load server and embedding configurations from environment variables
	serverConfig := LoadServerConfig()
	embeddingConfig := LoadEmbeddingConfig()

	fmt.Printf("Server config: Port=%s, ReadTimeout=%v, WriteTimeout=%v\n", serverConfig.Port, serverConfig.ReadTimeout, serverConfig.WriteTimeout)
	fmt.Printf("Embedding config: MaxBatchTokens=%d\n", embeddingConfig.MaxBatchTokens)

	// Load embedding model
	fmt.Println("Loading embedding model...")
	embeddingModel, err := InitEmbeddingModel(embeddingConfig)
	if err != nil {
		log.Fatalf("Failed to load embedding model: %v", err)
	}
	defer embeddingModel.Close()
	fmt.Println("Embedding model loaded successfully")

	// Create and start HTTP server
	mux := http.NewServeMux()
	mux.HandleFunc("/embed", func(w http.ResponseWriter, r *http.Request) {
		handleEmbed(w, r, embeddingModel)
	})
	server := &http.Server{
		Addr:         ":" + serverConfig.Port,
		Handler:      mux,
		ReadTimeout:  serverConfig.ReadTimeout,
		WriteTimeout: serverConfig.WriteTimeout,
	}
	go func() {
		fmt.Printf("Starting HTTP server on port %s...\n", serverConfig.Port)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server failed: %v", err)
		}
	}()

	// Wait for interrupt
	sigchan := make(chan os.Signal, 1)
	signal.Notify(sigchan, syscall.SIGINT, syscall.SIGTERM)
	sig := <-sigchan
	fmt.Printf("\nReceived signal %v: shutting down...\n", sig)
}

// handleEmbed processes a BatchEmbedRequest
func handleEmbed(w http.ResponseWriter, r *http.Request, embeddingModel *EmbeddingModel) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var batchReq BatchEmbedRequest
	if err := json.NewDecoder(r.Body).Decode(&batchReq); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if len(batchReq.Documents) == 0 {
		http.Error(w, "At least one document is required", http.StatusBadRequest)
		return
	}

	// Process each document
	response := BatchEmbedResponse{
		Documents: make([]DocumentResponse, len(batchReq.Documents)),
	}
	for i, doc := range batchReq.Documents {
		docResp := processDocument(embeddingModel, &doc)
		response.Documents[i] = docResp
	}

	// Send response
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("Failed to encode response: %v", err)
	}
}

// processDocument processes a single document and returns its response
func processDocument(embeddingModel *EmbeddingModel, doc *EmbedRequest) DocumentResponse {
	resp := DocumentResponse{
		ID: doc.ID,
	}

	// Validate text
	if doc.Text == "" {
		resp.Error = "Text field is required"
		return resp
	}

	// Use default configs if not provided
	chunkingConfig := DefaultChunkingConfig()
	if doc.ChunkingConfig != nil {
		chunkingConfig = *doc.ChunkingConfig
	}

	// Process the text
	chunks, err := processText(embeddingModel, doc.Text, chunkingConfig)
	if err != nil {
		resp.Error = fmt.Sprintf("Processing failed: %v", err)
		return resp
	}

	// Convert chunks to response format
	resp.Chunks = make([]ChunkResponse, len(chunks))
	for i, chunk := range chunks {
		resp.Chunks[i] = ChunkResponse{
			Text:         chunk.Text,
			StartTime:    chunk.StartTime,
			Embedding:    chunk.Embedding,
			NumSentences: chunk.NumSentences,
			TokenCount:   chunk.TokenCount,
			ChunkIndex:   chunk.ChunkIndex,
		}
	}

	return resp
}

// processText takes raw text and returns semantic chunks with embeddings
func processText(embeddingModel *EmbeddingModel, text string, chunkingConfig ChunkingConfig) ([]*Chunk, error) {
	log.Printf("Processing text (%d characters)", len(text))

	// Create a single "frame" from the input text
	frames := []Frame{{Text: text, StartTime: "", EndTime: ""}}

	// Extract sentences from the text
	sentences := embeddingModel.ExtractSentencesFromFrames(frames)
	log.Printf("Extracted %d sentences", len(sentences))

	if len(sentences) == 0 {
		return []*Chunk{}, nil
	}

	// Embed sentences
	if err := embeddingModel.EmbedSentences(sentences); err != nil {
		return nil, fmt.Errorf("failed to embed sentences: %w", err)
	}
	log.Printf("Embedded %d sentences", len(sentences))

	// Perform semantic chunking with provided config
	chunks, err := chunkingConfig.ExtractChunksFromSentences(sentences)
	if err != nil {
		return nil, fmt.Errorf("failed to extract chunks: %w", err)
	}
	log.Printf("Created %d chunks", len(chunks))

	// Embed chunks
	if err := embeddingModel.EmbedChunks(chunks); err != nil {
		return nil, fmt.Errorf("failed to embed chunks: %w", err)
	}
	log.Printf("Embedded %d chunks", len(chunks))

	return chunks, nil
}
