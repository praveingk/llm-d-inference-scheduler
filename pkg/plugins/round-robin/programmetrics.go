package roundrobin

import (
	"sync"
	"sync/atomic"
	"time"
)

const ewmaAlpha = 0.5

// Token cost weights: output tokens are ~2× more expensive than input tokens
const (
	weightInputToken  = 1
	weightOutputToken = 2
)

// ProgramMetrics holds aggregated metrics for a single program (identified by its fairness ID).
// All methods are goroutine-safe.
type ProgramMetrics struct {
	mu                sync.Mutex
	averageWaitTime   float64 // EWMA in milliseconds
	accumulatedWaitMs float64 // total accumulated wait time in milliseconds
	waitCount         int64   // number of wait time observations
	hasWaitData       bool

	// Per-request throughput tracking: tokens/sec per completed request.
	throughputSum   float64 // sum of per-request tokens/sec values
	throughputCount int64   // number of throughput observations

	// Service rate: EWMA of weighted tokens per second, updated on each completion.
	// Used for Jain's fairness index (equalizing rates across programs = fair).
	serviceRate        float64
	lastCompletionTime time.Time

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

// RecordThroughput records a per-request throughput observation (tokens/sec).
func (m *ProgramMetrics) RecordThroughput(tokensPerSec float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.throughputSum += tokensPerSec
	m.throughputCount++
}

// AverageThroughput returns the mean per-request throughput (tokens/sec)
// across all completed requests. Returns 0 if no data.
func (m *ProgramMetrics) AverageThroughput() float64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.throughputCount == 0 {
		return 0
	}
	return m.throughputSum / float64(m.throughputCount)
}

// RecordServiceRate updates the EWMA of service rate (weighted tokens/sec)
// using the elapsed time since the last completion.
func (m *ProgramMetrics) RecordServiceRate(weightedTokens float64, now time.Time) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.lastCompletionTime.IsZero() {
		m.lastCompletionTime = now
		return // no rate from a single point
	}
	elapsed := now.Sub(m.lastCompletionTime).Seconds()
	if elapsed <= 0 {
		return
	}
	instantRate := weightedTokens / elapsed
	if m.serviceRate == 0 {
		m.serviceRate = instantRate
	} else {
		m.serviceRate = ewmaAlpha*instantRate + (1-ewmaAlpha)*m.serviceRate
	}
	m.lastCompletionTime = now
}

// ServiceRate returns the EWMA of weighted tokens per second.
func (m *ProgramMetrics) ServiceRate() float64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.serviceRate
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

