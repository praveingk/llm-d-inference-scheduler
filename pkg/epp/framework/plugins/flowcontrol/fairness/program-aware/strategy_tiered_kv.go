package programaware

import (
	"math"
	"sort"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// TieredKVStrategy is the evolved "three-tier cascade + KV-pressure aware"
// scheduler (AdaEvolve run llm_d_las_0622_0904, Solution 3). Instead of one
// blended score it enforces a strict priority ladder:
//
//	Tier 1 (headWait > 5s):  pick max  headWaitMs^2 * (1 + kvLoad)   // hard starvation ceiling
//	Tier 2 (idle > 10s):     pick max  idleTimeMs  * (1 + 0.5*kvLoad)// progress gate
//	Tier 3 (everyone else):  score = (smoothedWait+1)^2 / (normService^1.5 + 1) * urgency
//	                         urgency = max(0.1, 1 - exp(-lastServedAgo / (5*(1+kvLoad))))
//
// kvLoad is a per-program EWMA of the completion/prompt token ratio (a proxy for
// KV-cache pressure — the real GPU bottleneck). Cost weights are fleet-adaptive:
// w_prompt = 2-sigma, w_completion = 1+sigma with sigma = fleetEWMA/(1+fleetEWMA),
// and service is resource-normalized by the program's own average request size.
type TieredKVStrategy struct {
	weightService   float64
	weightHeadWait  float64
	decayFactor     float64
	halfLifeSeconds float64

	state       sync.Map // key: program ID (string), value: *tieredKVState
	evolveState sync.Map // *tieredKVEvolveState per ID, plus id+"_avg" -> *float64 request-size EWMA

	// muAvg guards the read-modify-write of the id+"_avg" request-size EWMA
	// pointers stored in evolveState; concurrent OnCompleted for the same
	// program would otherwise race on the shared *float64.
	muAvg sync.Mutex

	muFleet              sync.Mutex
	fleetEWMA            float64
	fleetSampleCount     int64
	initializedFleetEWMA bool
}

var _ Strategy = &TieredKVStrategy{}

// tieredKVState is the primary per-program service/KV state.
type tieredKVState struct {
	mu                sync.Mutex
	rawService        float64 // cumulative raw weighted token cost
	normalizedService float64 // cumulative resource-normalized cost
	kvLoad            float64 // EWMA of completionTokens / (promptTokens+1)
	lastDecay         time.Time
	lastServed        time.Time
	servedCount       int64
}

// tieredKVEvolveState holds the scratch signals the scoring cascade needs.
type tieredKVEvolveState struct {
	mu              sync.Mutex
	lastServiceTime time.Time
	smoothedWait    float64 // EWMA of headWaitMs with alpha = 0.7
}

func (s *tieredKVState) Service() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.normalizedService
}

func (s *tieredKVState) KVLoad() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.kvLoad
}

// AddService accumulates both service counters, folds kvRatio into the kvLoad
// EWMA (alpha = 0.3), and marks the program as served.
func (s *tieredKVState) AddService(normalizedCost, rawCost, kvRatio float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	const kvAlpha = 0.3
	s.normalizedService += normalizedCost
	s.rawService += rawCost
	if s.servedCount == 0 {
		s.kvLoad = kvRatio
	} else {
		s.kvLoad = kvAlpha*kvRatio + (1-kvAlpha)*s.kvLoad
	}
	s.servedCount++
	now := time.Now()
	s.lastServed = now
	s.lastDecay = now
}

// Decay applies idle-only decay to both service counters and kvLoad at a fixed
// 30s half-life (the evolved algorithm hardcodes this, ignoring config).
func (s *tieredKVState) Decay(now time.Time, _, _ float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	const halfLifeSeconds = 30.0
	if s.lastDecay.IsZero() {
		s.lastDecay = now
		return
	}
	elapsed := now.Sub(s.lastDecay).Seconds()
	if elapsed <= 0 {
		return
	}
	f := math.Pow(0.5, elapsed/halfLifeSeconds)
	s.normalizedService *= f
	s.rawService *= f
	s.kvLoad *= f
	s.lastDecay = now
}

var tieredKVAttainedServiceTokens = prometheus.NewGaugeVec(
	prometheus.GaugeOpts{
		Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:      "program_aware_tiered_kv_attained_service_tokens",
		Help:      metricsutil.HelpMsgWithStability("Time-decayed resource-normalized attained service per program, tiered-kv strategy.", compbasemetrics.ALPHA),
	},
	[]string{"program_id"},
)

func (s *TieredKVStrategy) getOrCreateState(id string) *tieredKVState {
	if a, ok := s.state.Load(id); ok {
		if st, ok := a.(*tieredKVState); ok {
			return st
		}
	}
	fresh := &tieredKVState{}
	actual, _ := s.state.LoadOrStore(id, fresh)
	if st, ok := actual.(*tieredKVState); ok {
		return st
	}
	s.state.Store(id, fresh)
	return fresh
}

func (s *TieredKVStrategy) getOrCreateEvolveState(id string) *tieredKVEvolveState {
	if a, ok := s.evolveState.Load(id); ok {
		if st, ok := a.(*tieredKVEvolveState); ok {
			return st
		}
	}
	fresh := &tieredKVEvolveState{}
	actual, _ := s.evolveState.LoadOrStore(id, fresh)
	if st, ok := actual.(*tieredKVEvolveState); ok {
		return st
	}
	s.evolveState.Store(id, fresh)
	return fresh
}

// getFleetCompletionRatio returns sigma = fleetEWMA/(1+fleetEWMA), used to drift
// the cost weights with the workload. Defaults to 0.5 before any samples.
func (s *TieredKVStrategy) getFleetCompletionRatio() float64 {
	s.muFleet.Lock()
	defer s.muFleet.Unlock()
	if !s.initializedFleetEWMA {
		return 0.5
	}
	return s.fleetEWMA / (1.0 + s.fleetEWMA)
}

// updateFleetCompletionRatio folds a completion/prompt ratio into a very slow
// fleet-wide EWMA (alpha = 0.01).
func (s *TieredKVStrategy) updateFleetCompletionRatio(completionTokens, promptTokens int64) {
	ratio := float64(completionTokens) / math.Max(1.0, float64(promptTokens))
	s.muFleet.Lock()
	defer s.muFleet.Unlock()
	const alpha = 0.01
	if !s.initializedFleetEWMA {
		s.fleetEWMA = ratio
		s.initializedFleetEWMA = true
	} else {
		s.fleetEWMA = alpha*ratio + (1-alpha)*s.fleetEWMA
	}
	s.fleetSampleCount++
}

func (s *TieredKVStrategy) Name() string { return "tiered-kv" }

func (s *TieredKVStrategy) Pick(_ int, queues map[string]QueueInfo) flowcontrol.FlowQueueAccessor {
	const starveThresholdMs = 5000.0
	const progressGateMs = 10000.0 // 10 seconds
	const ewmaAlpha = 0.7
	const beta = 1.5
	const baseUrgencyTC = 5.0

	now := time.Now()

	type candidate struct {
		queue         flowcontrol.FlowQueueAccessor
		headWaitMs    float64
		smoothedWait  float64
		normService   float64
		kvLoad        float64
		waitLimitHit  bool
		progressHit   bool
		idleTimeMs    float64
		lastServedAgo float64 // seconds since last completion
	}

	candidates := make(map[string]candidate)
	anyWaitLimitHit := false
	anyProgressHit := false

	for id, qi := range queues {
		if qi.Metrics == nil {
			continue
		}

		st := s.getOrCreateState(id)

		// Idle-only decay (no requests waiting and nothing in flight).
		if qi.Len == 0 && qi.Metrics.InFlight() == 0 {
			st.Decay(now, s.halfLifeSeconds, s.decayFactor)
			continue
		}

		if qi.Len == 0 {
			continue
		}

		ev := s.getOrCreateEvolveState(id)
		normService := st.Service()
		kvLoad := st.KVLoad()

		var headWaitMs float64
		if head := qi.Queue.Peek(); head != nil {
			headWaitMs = float64(now.Sub(head.EnqueueTime()).Milliseconds())
		}

		// EWMA-smoothed wait for stable fairness decisions.
		ev.mu.Lock()
		var smoothedWait float64
		if ev.smoothedWait > 0 {
			smoothedWait = ewmaAlpha*headWaitMs + (1-ewmaAlpha)*ev.smoothedWait
		} else {
			smoothedWait = headWaitMs
		}
		ev.smoothedWait = smoothedWait
		lastServiceTime := ev.lastServiceTime
		ev.mu.Unlock()

		var idleTimeMs float64
		if !lastServiceTime.IsZero() {
			idleTimeMs = float64(now.Sub(lastServiceTime).Milliseconds())
		}

		waitHit := headWaitMs > starveThresholdMs
		progressHit := idleTimeMs > progressGateMs

		if waitHit {
			anyWaitLimitHit = true
		}
		if progressHit {
			anyProgressHit = true
		}

		// Seconds since last completion for urgency.
		var lastServedAgo float64
		if !lastServiceTime.IsZero() {
			lastServedAgo = now.Sub(lastServiceTime).Seconds()
		} else {
			lastServedAgo = 60.0 // New programs get a boost.
		}

		candidates[id] = candidate{
			queue:         qi.Queue,
			headWaitMs:    headWaitMs,
			smoothedWait:  smoothedWait,
			normService:   normService,
			kvLoad:        kvLoad,
			waitLimitHit:  waitHit,
			progressHit:   progressHit,
			idleTimeMs:    idleTimeMs,
			lastServedAgo: lastServedAgo,
		}
	}

	if len(candidates) == 0 {
		return nil
	}

	// Sort IDs for deterministic iteration.
	ids := make([]string, 0, len(candidates))
	for id := range candidates {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	// Tier 1: Starvation Prevention — longest head wait exceeding the threshold,
	// with a quadratic score for stronger differentiation at high waits.
	if anyWaitLimitHit {
		var best flowcontrol.FlowQueueAccessor
		bestScore := math.Inf(-1)
		for _, id := range ids {
			c := candidates[id]
			if !c.waitLimitHit {
				continue
			}
			// KV-pressure-aware starvation: prioritize high-KV programs.
			score := c.headWaitMs * c.headWaitMs * (1.0 + c.kvLoad)
			if score > bestScore {
				bestScore = score
				best = c.queue
			}
		}
		if best != nil {
			return best
		}
	}

	// Tier 2: Progress Gating — program not served in > 10 seconds.
	if anyProgressHit {
		var best flowcontrol.FlowQueueAccessor
		bestIdleMs := 0.0
		for _, id := range ids {
			c := candidates[id]
			if !c.progressHit {
				continue
			}
			// KV-pressure-aware progress gate.
			adjustedIdleMs := c.idleTimeMs * (1.0 + 0.5*c.kvLoad)
			if adjustedIdleMs > bestIdleMs {
				bestIdleMs = adjustedIdleMs
				best = c.queue
			}
		}
		if best != nil {
			return best
		}
	}

	// Tier 3: score = (smoothedWait+1)^2 / (normService^1.5 + 1) * urgency,
	// urgency = max(0.1, 1 - exp(-lastServedAgo / (baseUrgencyTC*(1+kvLoad)))).
	// High KV pressure => shorter effective time constant => faster urgency ramp.
	var best flowcontrol.FlowQueueAccessor
	bestScore := math.Inf(-1)
	for _, id := range ids {
		c := candidates[id]
		score := (c.smoothedWait + 1.0) * (c.smoothedWait + 1.0) / (math.Pow(c.normService, beta) + 1.0)

		effectiveTC := baseUrgencyTC * (1.0 + c.kvLoad)
		urgency := 1.0 - math.Exp(-c.lastServedAgo/effectiveTC)
		if urgency < 0.1 {
			urgency = 0.1
		}
		score *= urgency

		if score > bestScore {
			bestScore = score
			best = c.queue
		}
	}

	return best
}

func (s *TieredKVStrategy) OnPreRequest(_ *ProgramMetrics, _ *fwksched.InferenceRequest) {}

// avgKeyFor returns the evolveState map key holding a program's request-size EWMA.
func avgKeyFor(id string) string { return id + "_avg" }

func (s *TieredKVStrategy) OnCompleted(_ *ProgramMetrics, request *fwksched.InferenceRequest, response *fwkrc.Response) {
	if request == nil || response == nil {
		return
	}
	promptTokens := int64(response.Usage.PromptTokens)
	completionTokens := int64(response.Usage.CompletionTokens)
	totalTokens := float64(promptTokens + completionTokens)
	id := programIDFor(request)

	// Mark last service time for the urgency/progress signals.
	ev := s.getOrCreateEvolveState(id)
	ev.mu.Lock()
	ev.lastServiceTime = time.Now()
	ev.mu.Unlock()

	// Fleet-adaptive cost weights: sigma drifts with the workload.
	s.updateFleetCompletionRatio(completionTokens, promptTokens)
	sigma := s.getFleetCompletionRatio()
	dynWeightInput := 2.0 - sigma
	dynWeightOutput := 1.0 + sigma
	rawCost := dynWeightInput*float64(promptTokens) + dynWeightOutput*float64(completionTokens)

	// Resource-normalize by a per-program EWMA of request size (alpha = 0.3),
	// stored under id+"_avg" in the evolveState map.
	const avgAlpha = 0.3
	avgKey := avgKeyFor(id)
	var avg float64
	s.muAvg.Lock()
	if a, ok := s.evolveState.Load(avgKey); ok {
		if p, ok := a.(*float64); ok {
			avg = avgAlpha*totalTokens + (1-avgAlpha)*(*p)
			*p = avg
		}
	} else {
		avg = totalTokens
		v := avg
		s.evolveState.Store(avgKey, &v)
	}
	s.muAvg.Unlock()
	normalizedCost := rawCost / math.Max(1.0, avg)

	// KV ratio: completion/prompt (or raw completion tokens if no prompt).
	var kvRatio float64
	if promptTokens > 0 {
		kvRatio = float64(completionTokens) / float64(promptTokens)
	} else {
		kvRatio = float64(completionTokens)
	}

	st := s.getOrCreateState(id)
	st.AddService(normalizedCost, rawCost, kvRatio)
	tieredKVAttainedServiceTokens.WithLabelValues(id).Set(st.Service())
}

func (s *TieredKVStrategy) EvictProgram(id string) {
	s.state.Delete(id)
	s.evolveState.Delete(id)
	s.evolveState.Delete(avgKeyFor(id))
	tieredKVAttainedServiceTokens.DeleteLabelValues(id)
}

func (s *TieredKVStrategy) Collectors() []prometheus.Collector {
	return []prometheus.Collector{tieredKVAttainedServiceTokens}
}
