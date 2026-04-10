package programaware

import (
	"bufio"
	"encoding/json"
	"os"
	"sync"
	"time"
)

// PickLogEntry is one JSONL line written per Pick() invocation.
type PickLogEntry struct {
	Timestamp  time.Time          `json:"timestamp"`
	Strategy   string             `json:"strategy"`
	WinnerID      string             `json:"winnerId"`
	PickLatencyUs int64              `json:"pickLatencyUs"`
	Candidates    []PickLogCandidate `json:"candidates"`
}

// PickLogCandidate holds the scoring data for a single non-empty queue.
type PickLogCandidate struct {
	ProgramID        string         `json:"programId"`
	QueueDepth       int            `json:"queueDepth"`
	RawValues        []float64      `json:"rawValues"`
	NormalizedValues []float64      `json:"normalizedValues"`
	Score            float64        `json:"score"`
	Metrics          PickLogMetrics `json:"metrics"`
}

// PickLogMetrics is a snapshot of ProgramMetrics at Pick() time.
type PickLogMetrics struct {
	AttainedService   float64 `json:"attainedService"`
	AverageWaitTime   float64 `json:"averageWaitTime"`
	TotalRequests     int64   `json:"totalRequests"`
	DispatchedCount   int64   `json:"dispatchedCount"`
	TotalInputTokens  int64   `json:"totalInputTokens"`
	TotalOutputTokens int64   `json:"totalOutputTokens"`
	DeficitTokens     int64   `json:"deficitTokens"`
	ServiceRate       float64 `json:"serviceRate"`
}

// PickLogger writes JSONL pick-decision entries to a file.
// All methods are goroutine-safe.
type PickLogger struct {
	mu     sync.Mutex
	file   *os.File
	writer *bufio.Writer
	enc    *json.Encoder
}

// NewPickLogger opens the given path for append-only JSONL writing.
func NewPickLogger(path string) (*PickLogger, error) {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return nil, err
	}
	w := bufio.NewWriterSize(f, 64*1024)
	return &PickLogger{
		file:   f,
		writer: w,
		enc:    json.NewEncoder(w),
	}, nil
}

// Log writes a single JSONL entry and flushes the buffer.
func (l *PickLogger) Log(entry PickLogEntry) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if err := l.enc.Encode(entry); err != nil {
		return err
	}
	return l.writer.Flush()
}

// Close flushes any buffered data and closes the underlying file.
// It is safe to call Close multiple times; subsequent calls are no-ops.
func (l *PickLogger) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.file == nil {
		return nil
	}
	err := l.writer.Flush()
	if closeErr := l.file.Close(); closeErr != nil && err == nil {
		err = closeErr
	}
	l.file = nil
	return err
}