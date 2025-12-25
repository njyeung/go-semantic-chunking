package main

// Sentence: a single complete sentence
type Sentence struct {
	Text       string
	StartTime  string // Optional: can be empty for text-only input
	Embedding  []float32
	TokenCount int
}

// Chunk: semantically grouped sentences, formed by merging sentences based on embedding similarity
type Chunk struct {
	Text               string
	StartTime          string
	Embedding          []float32
	NumSentences       int
	TokenCount         int
	ChunkIndex         int
	SentenceEmbeddings [][]float32 // Individual sentence embeddings
}

