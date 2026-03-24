// Package programaware implements a flow-control fairness policy that schedules
// programs using their accumulated metrics using scoring strategies (EWMA or DRR).
package programaware

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"time"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	requestcontrol "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
)

const (
	// ProgramAwarePluginType is the registered type name for this plugin.
	ProgramAwarePluginType = "program-aware-fairness"

	// fairnessIDHeader is the standard header used to identify the program.
	fairnessIDHeader = "x-gateway-inference-fairness-id"

	// defaultFairnessID is the flow key assigned by the upstream framework when
	// no x-gateway-inference-fairness-id header is present on the request.
	// Matches the constant in sigs.k8s.io/gateway-api-inference-extension/pkg/epp/handlers/request.go.
	defaultFairnessID = "default-flow"

	// defaultDefaultShareLimit is the fraction of Pick() calls reserved for the
	// default queue when slotted dispatch is enabled.
	defaultDefaultShareLimit = 0.2
)

// Config holds the JSON-decoded configuration for the plugin.
type Config struct {
	// Strategy selects the fairness scoring algorithm used by Pick().
	// Valid values: "ewma" (default), "drr".
	//
	//   "ewma" — head-of-queue age + EWMA historical wait + dispatch-count penalty.
	//            Practical heuristic; strong starvation prevention.
	//
	//   "drr"  — Deficit Round Robin adapted for tokens [Shreedhar & Varghese 1995].
	//            Each round every active queue earns a token quantum; actual token
	//            usage is deducted at response completion. Provides provably
	//            proportional fairness independent of request rate or size.
	Strategy string `json:"strategy"`

	// SlottedDispatch enables reserved-slot scheduling for the default queue.
	// When true, every Nth Pick() call is reserved for the default (unlabeled)
	// queue, preventing starvation under heavy labeled-program traffic.
	SlottedDispatch bool `json:"slottedDispatch"`

	// DefaultShareLimit is the fraction of Pick() calls reserved for the default
	// queue (range (0,1)). Only used when SlottedDispatch is true. Default: 0.2.
	DefaultShareLimit float64 `json:"defaultShareLimit"`
}

// Compile-time interface assertions.
var (
	_ flowcontrol.FairnessPolicy       = &ProgramAwarePlugin{}
	_ requestcontrol.PrepareDataPlugin = &ProgramAwarePlugin{}
	_ requestcontrol.PreRequest        = &ProgramAwarePlugin{}
	_ requestcontrol.ResponseComplete  = &ProgramAwarePlugin{}
)

// ProgramAwarePluginFactory creates a new ProgramAwarePlugin from JSON config.
// Example config: {"strategy": "drr", "slottedDispatch": true, "defaultShareLimit": 0.2}
//
//nolint:revive
func ProgramAwarePluginFactory(name string, rawCfg json.RawMessage, _ plugin.Handle) (plugin.Plugin, error) {
	cfg := Config{Strategy: "ewma"}
	if len(rawCfg) > 0 {
		if err := json.Unmarshal(rawCfg, &cfg); err != nil {
			return nil, fmt.Errorf("invalid config for %s plugin %q: %w", ProgramAwarePluginType, name, err)
		}
	}
	strategy, err := newStrategy(cfg.Strategy)
	if err != nil {
		return nil, fmt.Errorf("%s plugin %q: %w", ProgramAwarePluginType, name, err)
	}

	p := &ProgramAwarePlugin{
		name:     name,
		strategy: strategy,
	}

	if cfg.SlottedDispatch {
		share := cfg.DefaultShareLimit
		if share == 0 {
			share = defaultDefaultShareLimit
		}
		if share <= 0 || share >= 1 {
			return nil, fmt.Errorf("invalid config for %s plugin %q: defaultShareLimit must be in (0, 1), got %v", ProgramAwarePluginType, name, cfg.DefaultShareLimit)
		}
		p.slottedDispatch = true
		p.cycleLength = int(math.Round(1.0 / share))
		if p.cycleLength < 1 {
			p.cycleLength = 1
		}
	}

	return p, nil
}

// ProgramAwarePlugin implements a FairnessPolicy that selects which program's
// queue to service next, and request lifecycle hooks that track per-program metrics.
//
// Fairness behaviour is determined by the configured ScoringStrategy (default: EWMA).
// Program identity comes from the x-gateway-inference-fairness-id request header.
//
//nolint:revive
type ProgramAwarePlugin struct {
	name     string
	strategy ScoringStrategy

	// Slotted dispatch: reserves every Nth Pick() call for the default queue.
	slottedDispatch bool
	cycleLength     int
	pickCounter     atomic.Uint64

	// programMetrics stores aggregated metrics per program.
	// Key: program ID (string), Value: *ProgramMetrics.
	programMetrics sync.Map

	// requestTimestamps tracks when Pick() dispatched each request,
	// used to compute flow-control queue wait time in PreRequest.
	// Key: request ID (string), Value: time.Time.
	requestTimestamps sync.Map
}

// TypedName returns the plugin type and instance name.
func (p *ProgramAwarePlugin) TypedName() plugin.TypedName {
	return plugin.TypedName{
		Type: ProgramAwarePluginType,
		Name: p.name,
	}
}

// getStrategy returns the configured strategy, falling back to EWMA for zero-value
// plugin instances constructed directly in tests.
func (p *ProgramAwarePlugin) getStrategy() ScoringStrategy {
	if p.strategy == nil {
		return &EWMAStrategy{}
	}
	return p.strategy
}

// --- FairnessPolicy interface ---

// NewState creates per-PriorityBand state. This plugin uses its own sync.Map
// for all state, so no per-band state is needed.
func (p *ProgramAwarePlugin) NewState(_ context.Context) any {
	return nil
}

// isDefaultQueue returns true if the queue belongs to the default (unlabeled) flow.
func isDefaultQueue(queue flowcontrol.FlowQueueAccessor) bool {
	id := queue.FlowKey().ID
	return id == defaultFairnessID || id == ""
}

// Pick selects which program queue to service next.
//
// For each queue in the band, the configured ScoringStrategy is given a chance
// to update its per-program state (OnPickStart), then the queue with the highest
// score is selected for dispatch.
//
// When slotted dispatch is enabled, every Nth call (where N = cycleLength) is
// reserved for the default queue. On other calls only labeled queues are scored.
// Both paths are work-conserving: if the designated queue type is empty, the
// other type is tried as a fallback.
func (p *ProgramAwarePlugin) Pick(_ context.Context, band flowcontrol.PriorityBandAccessor) (flowcontrol.FlowQueueAccessor, error) {
	start := time.Now()
	defer func() {
		pickLatencyUs.Observe(float64(time.Since(start).Microseconds()))
	}()

	if band == nil {
		return nil, nil //nolint:nilnil
	}

	var bestQueue flowcontrol.FlowQueueAccessor

	if p.slottedDispatch {
		bestQueue = p.pickSlotted(band)
	} else {
		bestQueue = p.pickScored(band, false)
	}

	// Record the selected item's enqueue time so PreRequest can compute
	// the actual flow-control queue wait time (enqueue → dispatch).
	if bestQueue != nil {
		if head := bestQueue.PeekHead(); head != nil {
			p.requestTimestamps.Store(head.OriginalRequest().ID(), head.EnqueueTime())
		}
	}

	fairnessIndex.Set(p.computeFairnessIndex())

	return bestQueue, nil
}

// pickSlotted implements the slotted dispatch algorithm.
// Every cycleLength-th call reserves the pick for the default queue.
func (p *ProgramAwarePlugin) pickSlotted(band flowcontrol.PriorityBandAccessor) flowcontrol.FlowQueueAccessor {
	counter := p.pickCounter.Add(1)
	isDefaultSlot := (counter % uint64(p.cycleLength)) == 0

	if isDefaultSlot {
		// Default slot: try default queue first, fall back to best labeled queue.
		if dq := band.Queue(defaultFairnessID); dq != nil && dq.Len() > 0 {
			return dq
		}
		return p.pickScored(band, true)
	}

	// Labeled slot: score labeled queues only, fall back to default if none available.
	if best := p.pickScored(band, true); best != nil {
		return best
	}
	if dq := band.Queue(defaultFairnessID); dq != nil && dq.Len() > 0 {
		return dq
	}
	return nil
}

// pickScored iterates queues and returns the highest-scoring non-empty one.
// It also runs OnPickStart for every queue so DRR quantum bookkeeping stays correct.
// When skipDefault is true, the default queue is excluded from scoring.
func (p *ProgramAwarePlugin) pickScored(band flowcontrol.PriorityBandAccessor, skipDefault bool) flowcontrol.FlowQueueAccessor {
	var bestQueue flowcontrol.FlowQueueAccessor
	bestScore := -1.0
	strategy := p.getStrategy()

	band.IterateQueues(func(queue flowcontrol.FlowQueueAccessor) (keepIterating bool) {
		if queue == nil {
			return true
		}

		queueLen := queue.Len()
		metrics := p.getOrCreateMetrics(queue.FlowKey().ID)
		strategy.OnPickStart(queue.FlowKey().ID, queueLen, metrics)

		if queueLen == 0 {
			return true
		}
		if skipDefault && isDefaultQueue(queue) {
			return true
		}

		score := p.scoreQueue(queue)
		queueScore.WithLabelValues(queue.FlowKey().ID).Set(score)
		if score > bestScore {
			bestScore = score
			bestQueue = queue
		}
		return true
	})

	return bestQueue
}

// scoreQueue delegates to the configured ScoringStrategy.
func (p *ProgramAwarePlugin) scoreQueue(queue flowcontrol.FlowQueueAccessor) float64 {
	var metrics *ProgramMetrics
	if metricsRaw, ok := p.programMetrics.Load(queue.FlowKey().ID); ok {
		metrics = metricsRaw.(*ProgramMetrics)
	}
	return p.getStrategy().ScoreQueue(queue, metrics)
}

// getOrCreateMetrics returns the ProgramMetrics for the given program ID, creating if needed.
func (p *ProgramAwarePlugin) getOrCreateMetrics(programID string) *ProgramMetrics {
	if metricsRaw, ok := p.programMetrics.Load(programID); ok {
		return metricsRaw.(*ProgramMetrics)
	}
	m := &ProgramMetrics{}
	actual, _ := p.programMetrics.LoadOrStore(programID, m)
	return actual.(*ProgramMetrics)
}

// computeFairnessIndex returns Jain's Fairness Index over the EWMA average wait
// Returns 1.0 when fewer than 2 programs have wait data.
func (p *ProgramAwarePlugin) computeFairnessIndex() float64 {
	var sum, sumSq float64
	var n float64
	p.programMetrics.Range(func(_, value any) bool {
		m := value.(*ProgramMetrics)
		if !m.HasWaitData() {
			return true
		}
		x := m.AverageWaitTime()
		sum += x
		sumSq += x * x
		n++
		return true
	})
	if n <= 1 || sumSq == 0 {
		return 1.0
	}
	return (sum * sum) / (n * sumSq)
}

// normalize clamps v/cap to [0, 1].
func normalize(v, cap float64) float64 {
	if cap <= 0 {
		return 0
	}
	return math.Min(math.Max(v/cap, 0), 1)
}
