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

// ISCStrategy is the evolved "Integer Starvation Counter" scheduler (AdaEvolve
// run llm_d_las_0622_1603, Solution 5). It replaces floating-point starvation
// meters with a single integer counter of consecutive losses:
//
//	score = waitSecs * (1 + 0.15*consecutiveMisses)^2 / (1 + 0.1*bank)
//
// Every Pick, each non-selected program's miss counter is incremented (capped at
// 100); the winner — or any program whose queue empties — resets to 0. The
// quadratic miss boost ramps a repeatedly-skipped program hard but predictably.
// "bank" is a decayed, clamped attained-service balance (15s half-life): a high
// balance divides the score down, so recent heavy consumers wait.
type ISCStrategy struct {
	weightService   float64
	weightHeadWait  float64
	decayFactor     float64
	halfLifeSeconds float64

	state sync.Map // key: program ID (string), value: *iscState
}

var _ Strategy = &ISCStrategy{}

// iscBankHalfLifeSeconds is the fixed half-life of the service "bank".
const iscBankHalfLifeSeconds = 15.0

// iscState is the per-program state for ISCStrategy.
type iscState struct {
	mu sync.Mutex
	// bank is the attained-service balance (decays with a 15s half-life, capped at 80).
	bank        float64
	decayAnchor time.Time
	// consecutiveMisses counts Pick losses in a row; incremented on skip, reset on service (cap 100).
	consecutiveMisses int32
}

func (s *iscState) Service() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.bank
}

func (s *iscState) ConsecutiveMisses() int32 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.consecutiveMisses
}

func (s *iscState) IncrementMiss() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.consecutiveMisses < 100 {
		s.consecutiveMisses++
	}
}

func (s *iscState) ResetMiss() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.consecutiveMisses = 0
}

// decayBankLocked applies 15s half-life decay to the bank. Caller holds s.mu.
func (s *iscState) decayBankLocked(now time.Time) {
	if s.decayAnchor.IsZero() {
		s.decayAnchor = now
		return
	}
	elapsed := now.Sub(s.decayAnchor).Seconds()
	if elapsed <= 0 {
		return
	}
	s.bank *= math.Pow(0.5, elapsed/iscBankHalfLifeSeconds)
	s.decayAnchor = now
}

// AddService decays the bank, adds a bounded deposit (clamped to [0.5, 5.0]),
// caps the total bank at 80, and resets the miss counter.
func (s *iscState) AddService(cost float64) float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now()
	s.decayBankLocked(now)
	dep := cost
	if dep < 0.5 {
		dep = 0.5
	} else if dep > 5.0 {
		dep = 5.0
	}
	s.bank += dep
	if s.bank > 80.0 {
		s.bank = 80.0
	}
	s.decayAnchor = now
	s.consecutiveMisses = 0
	return s.bank
}

// UpdateBank applies per-tick 15s half-life decay to the bank.
func (s *iscState) UpdateBank(now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.decayBankLocked(now)
}

// Decay applies idle decay to the bank (honors the configured half-life if set,
// otherwise the fixed 15s bank half-life).
func (s *iscState) Decay(now time.Time, halfLifeSeconds, factor float64) {
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
		s.bank *= math.Pow(0.5, elapsed/halfLifeSeconds)
		s.decayAnchor = now
		return
	}
	s.decayBankLocked(now)
	_ = factor
}

var iscBank = prometheus.NewGaugeVec(
	prometheus.GaugeOpts{
		Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:      "program_aware_isc_bank",
		Help:      metricsutil.HelpMsgWithStability("Time-decayed attained-service bank per program, isc strategy.", compbasemetrics.ALPHA),
	},
	[]string{"program_id"},
)

func (s *ISCStrategy) getOrCreateState(id string) *iscState {
	if a, ok := s.state.Load(id); ok {
		if st, ok := a.(*iscState); ok {
			return st
		}
	}
	fresh := &iscState{}
	actual, _ := s.state.LoadOrStore(id, fresh)
	if st, ok := actual.(*iscState); ok {
		return st
	}
	s.state.Store(id, fresh)
	return fresh
}

func (s *ISCStrategy) Name() string { return "isc" }

func (s *ISCStrategy) Pick(_ int, queues map[string]QueueInfo) flowcontrol.FlowQueueAccessor {
	now := time.Now()

	type candidate struct {
		id                string
		queue             flowcontrol.FlowQueueAccessor
		bank              float64
		consecutiveMisses int32
		waitSecs          float64
	}

	candidates := make([]candidate, 0, len(queues))

	for id, qi := range queues {
		if qi.Metrics == nil {
			continue
		}

		st := s.getOrCreateState(id)

		if qi.Len == 0 {
			// No pending requests. If nothing in flight either, decay and reset misses.
			if qi.Metrics.InFlight() == 0 {
				st.Decay(now, s.halfLifeSeconds, s.decayFactor)
				st.ResetMiss()
			}
			continue
		}

		// Decay bank on every tick.
		st.UpdateBank(now)

		bank := st.Service()
		misses := st.ConsecutiveMisses()

		var headWaitSecs float64
		if head := qi.Queue.Peek(); head != nil {
			headWaitSecs = now.Sub(head.EnqueueTime()).Seconds()
		}

		candidates = append(candidates, candidate{
			id:                id,
			queue:             qi.Queue,
			bank:              bank,
			consecutiveMisses: misses,
			waitSecs:          headWaitSecs,
		})
	}

	if len(candidates) == 0 {
		return nil
	}

	// Sort for deterministic selection.
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].id < candidates[j].id
	})

	var bestQueue flowcontrol.FlowQueueAccessor
	bestScore := math.Inf(-1)

	// score = wait * (1 + kMiss*misses)^2 / (1 + kBank*bank)
	const kMiss = 0.15 // miss multiplier coefficient
	const kBank = 0.1  // bank discount coefficient

	for _, c := range candidates {
		// Miss boost: grows quadratically with consecutive misses.
		missBoost := 1.0 + kMiss*float64(c.consecutiveMisses)
		starveMultiplier := missBoost * missBoost

		// Bank discount: mild penalty for recently-served programs.
		bankDiscount := 1.0 + kBank*c.bank

		score := c.waitSecs * starveMultiplier / bankDiscount

		if score > bestScore {
			bestScore = score
			bestQueue = c.queue
		}
	}

	// Increment miss counter for all non-selected programs.
	for _, c := range candidates {
		if c.queue != bestQueue {
			s.getOrCreateState(c.id).IncrementMiss()
		}
	}

	return bestQueue
}

func (s *ISCStrategy) OnPreRequest(_ *ProgramMetrics, _ *fwksched.InferenceRequest) {}

func (s *ISCStrategy) OnCompleted(_ *ProgramMetrics, request *fwksched.InferenceRequest, response *fwkrc.Response) {
	if request == nil || response == nil {
		return
	}
	promptTokens := int64(response.Usage.PromptTokens)
	completionTokens := int64(response.Usage.CompletionTokens)
	// Fixed token cost 1x/2x; AddService clamps the deposit and caps the bank.
	cost := float64(weightInputToken*promptTokens + weightOutputToken*completionTokens)
	id := programIDFor(request)
	// AddService handles bank accumulation AND resets consecutiveMisses to 0.
	bankVal := s.getOrCreateState(id).AddService(cost)
	iscBank.WithLabelValues(id).Set(bankVal)
}

func (s *ISCStrategy) EvictProgram(id string) {
	s.state.Delete(id)
	iscBank.DeleteLabelValues(id)
}

func (s *ISCStrategy) Collectors() []prometheus.Collector {
	return []prometheus.Collector{iscBank}
}
