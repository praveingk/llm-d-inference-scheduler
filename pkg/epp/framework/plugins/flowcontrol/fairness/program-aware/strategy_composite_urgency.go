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

// CompositeUrgencyStrategy is the evolved "composite urgency" scheduler
// (AdaEvolve run llm_d_las_0622_2242, Solution 10). It scores each program by
// summing three deliberately-simple, individually-bounded terms:
//
//	score = waitScore + serviceScore + starveBoost
//	  waitScore    = headWaitMs * (1 + sqrt(qLen)/sqrt(maxSeenDepth))  // immediate p99 driver
//	  serviceScore = cumWaitMs / (1 + 0.01*attainedService)           // cumulative-wait fairness
//	  starveBoost  = starveFactor * headWaitMs * 0.005                 // gentle, capped 6x ramp
//
// It deliberately avoids EMAs, pending-cost estimation, and dual-timescale
// debt: every term is bounded and self-resetting, which is why it is the
// safest of the evolved policies to operate.
type CompositeUrgencyStrategy struct {
	weightService   float64
	weightHeadWait  float64
	decayFactor     float64
	halfLifeSeconds float64

	state sync.Map // key: program ID (string), value: *compositeState
}

var _ Strategy = &CompositeUrgencyStrategy{}

// compositeState is the per-program state for CompositeUrgencyStrategy.
type compositeState struct {
	mu              sync.Mutex
	attainedService float64
	// decayAnchor is the wall-clock anchor for time-based decay.
	decayAnchor time.Time
	// cumWaitMs accumulates every ms this program goes unserved and resets to 0 on service.
	cumWaitMs float64
	// lastServiceTime is the last time this program was served (used to accrue cumWaitMs).
	lastServiceTime time.Time
	// starveFactor is a bounded escalation multiplier in [1, 6]: *1.4 on skip, /2 on service.
	starveFactor float64
	// maxSeenDepth is the running max queue depth for sqrt-normalization.
	maxSeenDepth float64
	hasHistory   bool
}

func (s *compositeState) Service() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.attainedService
}

// AccrueWait adds the elapsed ms since the last service to cumWaitMs and returns it.
func (s *compositeState) AccrueWait(now time.Time) float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.hasHistory && !s.lastServiceTime.IsZero() {
		elapsed := float64(now.Sub(s.lastServiceTime).Milliseconds())
		if elapsed > 0 {
			s.cumWaitMs += elapsed
		}
	}
	s.lastServiceTime = now
	s.hasHistory = true
	return s.cumWaitMs
}

// AddService charges the completion cost, resets the cumulative-wait meter, and
// relaxes the starve factor (halved, floored at 1).
func (s *compositeState) AddService(cost float64) float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.attainedService += cost
	s.decayAnchor = time.Now()
	s.cumWaitMs = 0
	if s.starveFactor <= 0 {
		s.starveFactor = 1.0
	}
	s.starveFactor = math.Max(s.starveFactor/2.0, 1.0)
	s.lastServiceTime = time.Now()
	s.hasHistory = true
	return s.attainedService
}

// RecordSkip escalates the starve factor (*1.4, capped at 6) for a program that
// lost the Pick, and tracks the running max queue depth.
func (s *compositeState) RecordSkip(currentQueueLen int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.starveFactor < 1.0 {
		s.starveFactor = 1.0
	}
	s.starveFactor = math.Min(s.starveFactor*1.4, 6.0)
	if d := float64(currentQueueLen); d > s.maxSeenDepth {
		s.maxSeenDepth = d
	}
}

func (s *compositeState) GetStarveFactor() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.starveFactor < 1.0 {
		return 1.0
	}
	return s.starveFactor
}

func (s *compositeState) GetMaxDepth() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.maxSeenDepth
}

// Decay applies idle-only time-based decay: attainedService at the configured
// half-life, and cumWaitMs at a 4x-longer half-life (slower to forget neglect).
func (s *compositeState) Decay(now time.Time, halfLifeSeconds, factor float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if halfLifeSeconds > 0 {
		if s.decayAnchor.IsZero() {
			s.decayAnchor = now
			return
		}
		elapsed := now.Sub(s.decayAnchor).Seconds()
		if elapsed <= 0 {
			return
		}
		s.attainedService *= math.Pow(0.5, elapsed/halfLifeSeconds)
		s.cumWaitMs *= math.Pow(0.5, elapsed/(4*halfLifeSeconds))
		s.decayAnchor = now
		return
	}
	s.attainedService *= factor
	s.cumWaitMs *= factor
}

var compositeUrgencyAttainedServiceTokens = prometheus.NewGaugeVec(
	prometheus.GaugeOpts{
		Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:      "program_aware_composite_urgency_attained_service_tokens",
		Help:      metricsutil.HelpMsgWithStability("Time-decayed attained service (weighted tokens consumed) per program, composite-urgency strategy.", compbasemetrics.ALPHA),
	},
	[]string{"program_id"},
)

func (s *CompositeUrgencyStrategy) getOrCreateState(id string) *compositeState {
	if a, ok := s.state.Load(id); ok {
		if st, ok := a.(*compositeState); ok {
			return st
		}
	}
	fresh := &compositeState{starveFactor: 1.0}
	actual, _ := s.state.LoadOrStore(id, fresh)
	if st, ok := actual.(*compositeState); ok {
		return st
	}
	s.state.Store(id, fresh)
	return fresh
}

func (s *CompositeUrgencyStrategy) Name() string { return "composite-urgency" }

func (s *CompositeUrgencyStrategy) Pick(_ int, queues map[string]QueueInfo) flowcontrol.FlowQueueAccessor {
	now := time.Now()

	// Collect active IDs with deterministic ordering.
	activeIDs := make([]string, 0)
	for id, qi := range queues {
		if qi.Metrics == nil {
			continue
		}
		if qi.Len == 0 {
			// Idle-only decay: skip while a request is in flight.
			if qi.Metrics.InFlight() == 0 {
				st := s.getOrCreateState(id)
				st.Decay(now, s.halfLifeSeconds, s.decayFactor)
			}
			continue
		}
		activeIDs = append(activeIDs, id)
	}
	sort.Strings(activeIDs)

	if len(activeIDs) == 0 {
		return nil
	}

	// Compute composite urgency score for each active program.
	var bestID string
	bestScore := math.Inf(-1)

	for _, id := range activeIDs {
		qi := queues[id]
		st := s.getOrCreateState(id)
		attained := st.Service()

		// Peek the head before touching the wait meter: a queue reported with
		// Len>0 may drain before Pick reaches it, and accruing cumulative wait
		// for an empty queue would let the meter run away.
		head := qi.Queue.Peek()
		if head == nil {
			continue
		}
		headWaitMs := float64(now.Sub(head.EnqueueTime()).Milliseconds())

		// Accrue cumulative wait since last service (only for a genuinely
		// pending program).
		cumWait := st.AccrueWait(now)

		// Queue depth normalization: sqrt depth / sqrt max depth.
		normDepth := float64(qi.Len)
		maxDepth := st.GetMaxDepth()
		if maxDepth > 0 {
			normDepth = math.Sqrt(float64(qi.Len)) / math.Sqrt(maxDepth)
		}
		queueMult := 1.0 + normDepth

		// Wait component: head wait * queue depth multiplier (immediate p99 driver).
		waitScore := headWaitMs * queueMult

		// Service component: cumulative wait / (1 + 0.01*attainedService).
		// Links waiting time with compute consumed for fairness.
		serviceScore := cumWait / (1.0 + 0.01*attained)

		// Starve escalation: gentle boost for repeatedly skipped programs.
		starve := st.GetStarveFactor()
		starveBoost := starve * headWaitMs * 0.005

		score := waitScore + serviceScore + starveBoost

		if score > bestScore {
			bestScore = score
			bestID = id
		}
	}

	// Record skip for non-selected programs.
	for _, id := range activeIDs {
		if id != bestID {
			st := s.getOrCreateState(id)
			st.RecordSkip(queues[id].Len)
		}
	}

	if bestID != "" {
		return queues[bestID].Queue
	}
	return nil
}

func (s *CompositeUrgencyStrategy) OnPreRequest(_ *ProgramMetrics, _ *fwksched.InferenceRequest) {
	// Deliberate no-op: pending-cost estimation creates harmful feedback loops
	// on a saturated backend.
}

func (s *CompositeUrgencyStrategy) OnCompleted(_ *ProgramMetrics, request *fwksched.InferenceRequest, response *fwkrc.Response) {
	if request == nil || response == nil {
		return
	}
	promptTokens := int64(response.Usage.PromptTokens)
	completionTokens := int64(response.Usage.CompletionTokens)
	// Moderate output weighting: 1x prompt + 3x completion (heavier than the seed's 2x).
	cost := float64(1*promptTokens + 3*completionTokens)
	id := programIDFor(request)
	service := s.getOrCreateState(id).AddService(cost)
	compositeUrgencyAttainedServiceTokens.WithLabelValues(id).Set(service)
}

func (s *CompositeUrgencyStrategy) EvictProgram(id string) {
	s.state.Delete(id)
	compositeUrgencyAttainedServiceTokens.DeleteLabelValues(id)
}

func (s *CompositeUrgencyStrategy) Collectors() []prometheus.Collector {
	return []prometheus.Collector{compositeUrgencyAttainedServiceTokens}
}
