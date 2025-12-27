package main

import (
	"strings"

	tokenizer "github.com/sugarme/tokenizer"
)

// ExtractSentencesFromText splits text into sentences based on sentence boundaries
// A sentence is text ending with . or ? or !
func (em *EmbeddingModel) ExtractSentencesFromText(text string, maxSize int) []*Sentence {
	if text == "" {
		return []*Sentence{}
	}

	var sentences []*Sentence
	var currentSentence strings.Builder

	// Split text into words and punctuation
	words := strings.Fields(text)

	for _, word := range words {
		if currentSentence.Len() > 0 {
			currentSentence.WriteString(" ")
		}
		currentSentence.WriteString(word)

		// Check if this word ends with . or ? or !
		trimmed := strings.TrimSpace(word)
		if strings.HasSuffix(trimmed, ".") || strings.HasSuffix(trimmed, "!") || strings.HasSuffix(trimmed, "?") {
			sentenceText := currentSentence.String()

			sentences = append(sentences, &Sentence{
				Text:       sentenceText,
				StartTime:  "",  // Not applicable for text input
				Embedding:  nil, // Will be populated by embedding function
				TokenCount: CountTokens(em.Tokenizer, sentenceText),
			})

			currentSentence.Reset()
		}
	}

	// Add any remaining text as a sentence
	if currentSentence.Len() > 0 {
		sentenceText := currentSentence.String()
		sentences = append(sentences, &Sentence{
			Text:       sentenceText,
			StartTime:  "",
			Embedding:  nil,
			TokenCount: CountTokens(em.Tokenizer, sentenceText),
		})
	}

	// Post-process: split any oversized sentences (>512 tokens) into smaller chunks
	// This prevents the DP algorithm from failing when individual sentences are too large
	finalSentences := make([]*Sentence, 0, len(sentences))

	for _, sent := range sentences {
		if sent.TokenCount <= maxSize {
			finalSentences = append(finalSentences, sent)
			continue
		}

		// Split oversized sentence by words
		words := strings.Fields(sent.Text)
		if len(words) == 0 {
			finalSentences = append(finalSentences, sent)
			continue
		}

		// Binary search to find how many words fit in maxTokens
		var currentChunk strings.Builder
		for len(words) > 0 {
			// Start with first word
			currentChunk.Reset()
			currentChunk.WriteString(words[0])
			wordCount := 1

			// Add words until we hit token limit
			for wordCount < len(words) {
				testText := currentChunk.String() + " " + words[wordCount]
				tokens := CountTokens(em.Tokenizer, testText)

				if tokens > maxSize {
					break
				}

				currentChunk.WriteString(" ")
				currentChunk.WriteString(words[wordCount])
				wordCount++
			}

			// Create sub-sentence
			chunkText := currentChunk.String()
			finalSentences = append(finalSentences, &Sentence{
				Text:       chunkText,
				StartTime:  sent.StartTime,
				Embedding:  nil,
				TokenCount: CountTokens(em.Tokenizer, chunkText),
			})

			words = words[wordCount:]
		}
	}

	return finalSentences
}

func CountTokens(tok *tokenizer.Tokenizer, text string) int {
	encoding, err := tok.EncodeSingle(text)
	if err != nil {
		return 0
	}

	return len(encoding.GetIds())
}
