package roundrobin

import (
	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"
	metricsutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/metrics"
)

const roundRobinSubsystem = "round_robin"

// Package-level metrics that the plugin records to directly.
// These are registered at startup via GetCollectors() before the plugin is instantiated.
var (
	requestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: roundRobinSubsystem,
			Name:      "requests_total",
			Help:      metricsutil.HelpMsgWithStability("Total requests received per program", compbasemetrics.ALPHA),
		},
		[]string{"program_id"},
	)

	dispatchedTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: roundRobinSubsystem,
			Name:      "dispatched_total",
			Help:      metricsutil.HelpMsgWithStability("Total requests dispatched per program", compbasemetrics.ALPHA),
		},
		[]string{"program_id"},
	)

	waitTimeMs = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: roundRobinSubsystem,
			Name:      "wait_time_milliseconds",
			Help:      metricsutil.HelpMsgWithStability("Flow control queue wait time per program in milliseconds", compbasemetrics.ALPHA),
			Buckets:   []float64{1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000},
		},
		[]string{"program_id"},
	)

	inputTokensTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: roundRobinSubsystem,
			Name:      "input_tokens_total",
			Help:      metricsutil.HelpMsgWithStability("Total input (prompt) tokens consumed per program", compbasemetrics.ALPHA),
		},
		[]string{"program_id"},
	)

	outputTokensTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: roundRobinSubsystem,
			Name:      "output_tokens_total",
			Help:      metricsutil.HelpMsgWithStability("Total output (completion) tokens produced per program", compbasemetrics.ALPHA),
		},
		[]string{"program_id"},
	)

	pickLatencyUs = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: roundRobinSubsystem,
			Name:      "pick_latency_microseconds",
			Help:      metricsutil.HelpMsgWithStability("Latency of the Pick() call in microseconds", compbasemetrics.ALPHA),
			Buckets:   []float64{1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 5000},
		},
	)

	fairnessIndex = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Subsystem: roundRobinSubsystem,
			Name:      "jains_fairness_index",
			Help:      metricsutil.HelpMsgWithStability("Jain's fairness index over average throughput (tokens/sec) across active programs (1.0 = perfectly fair)", compbasemetrics.ALPHA),
		},
	)

	throughputTokensPerSec = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: roundRobinSubsystem,
			Name:      "throughput_tokens_per_second",
			Help:      metricsutil.HelpMsgWithStability("Average per-request throughput (tokens/sec) per program", compbasemetrics.ALPHA),
		},
		[]string{"program_id"},
	)

	ewmaWaitTimeMs = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: roundRobinSubsystem,
			Name:      "ewma_wait_time_milliseconds",
			Help:      metricsutil.HelpMsgWithStability("Exponentially weighted moving average of flow control queue wait time per program in milliseconds", compbasemetrics.ALPHA),
		},
		[]string{"program_id"},
	)
)

// GetCollectors returns all Prometheus collectors for the round-robin plugin.
// Called from pkg/metrics/metrics.go to register with the runner at startup.
func GetCollectors() []prometheus.Collector {
	return []prometheus.Collector{
		requestsTotal,
		dispatchedTotal,
		waitTimeMs,
		inputTokensTotal,
		outputTokensTotal,
		pickLatencyUs,
		fairnessIndex,
		throughputTokensPerSec,
		ewmaWaitTimeMs,
	}
}
