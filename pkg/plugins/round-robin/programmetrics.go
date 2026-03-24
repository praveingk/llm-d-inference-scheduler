package roundrobin

import (
	"sync"
	"sync/atomic"
)

const ewmaAlpha = 0.2

// ProgramMetrics holds aggregated metrics for a single program (identified by its fairness ID).
// All methods are goroutine-safe.
type ProgramMetrics struct {
	mu                sync.Mutex
	averageWaitTime   float64 // EWMA in milliseconds
	accumulatedWaitMs float64 // total accumulated wait time in milliseconds
	waitCount         int64   // number of wait time observations
	hasWaitData       bool

	totalRequests     atomic.Int64
	dispatchedCount   atomic.Int64
	totalInputTokens  atomic.Int64
	totalOutputTokens atomic.Int64
}

// IncrementRequests atomically increments the total request counter.
func (m *ProgramMetrics) IncrementRequests() {
	m.totalRequests.Add(1)
}

// IncrementDispatched atomically increments the dispatched counter.
func (m *ProgramMetrics) IncrementDispatched() {
	m.dispatchedCount.Add(1)
}

// RecordWaitTime updates the EWMA of wait time with a new observation
// and accumulates the total wait time for lifetime average computation.
func (m *ProgramMetrics) RecordWaitTime(waitMs float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.hasWaitData {
		m.averageWaitTime = waitMs
		m.hasWaitData = true
	} else {
		m.averageWaitTime = ewmaAlpha*waitMs + (1-ewmaAlpha)*m.averageWaitTime
	}
	m.accumulatedWaitMs += waitMs
	m.waitCount++
}

// HasWaitData returns true if at least one wait time observation has been recorded.
func (m *ProgramMetrics) HasWaitData() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.hasWaitData
}

// AverageWaitTime returns the current EWMA of wait time in milliseconds.
func (m *ProgramMetrics) AverageWaitTime() float64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.averageWaitTime
}

// TotalAverageWaitTime returns the lifetime average wait time in milliseconds
// (accumulated wait time / total observations). Returns 0 if no data.
func (m *ProgramMetrics) TotalAverageWaitTime() float64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.waitCount == 0 {
		return 0
	}
	return m.accumulatedWaitMs / float64(m.waitCount)
}

// RecordTokens adds token counts from a completed request.
func (m *ProgramMetrics) RecordTokens(input, output int64) {
	m.totalInputTokens.Add(input)
	m.totalOutputTokens.Add(output)
}

// TotalRequests returns the total number of requests seen for this program.
func (m *ProgramMetrics) TotalRequests() int64 {
	return m.totalRequests.Load()
}

// DispatchedCount returns the total number of dispatched requests for this program.
func (m *ProgramMetrics) DispatchedCount() int64 {
	return m.dispatchedCount.Load()
}

// TotalInputTokens returns the total number of input tokens across all requests.
func (m *ProgramMetrics) TotalInputTokens() int64 {
	return m.totalInputTokens.Load()
}

// TotalOutputTokens returns the total number of output tokens across all requests.
func (m *ProgramMetrics) TotalOutputTokens() int64 {
	return m.totalOutputTokens.Load()
}

