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

// ExpBackoffStrategy is the evolved "exponential backoff on time-since-served"
// scheduler (AdaEvolve run llm_d_las_0622_1510, Solution 4). Attained service is
// tracked and decayed but NOT used for priority — this is a time-fair (rather
// than compute-fair) policy. The longer a program has gone unserved, the
// exponentially more urgent it becomes:
//
//	score = expFactor * waitUrgency * queueBonus + 0.0001*headWaitMs
//	  expFactor   = exp(beta * timeSinceLastServe / maxTimeSinceLast)  // beta = 0.8
//	  waitUrgency = sqrt(headWaitMs / avgHeadWait)                     // dampened relative wait
//	  queueBonus  = 1 + 0.1*queueDepthEWMA
//
// Because expFactor is normalized by the worst-neglected program, starvation is
// essentially impossible: a long-ignored program's score runs away.
type ExpBackoffStrategy struct {
	weightService   float64
	weightHeadWait  float64
	decayFactor     float64
	halfLifeSeconds float64

	state sync.Map // key: program ID (string), value: *expBackoffState
}

var _ Strategy = &ExpBackoffStrategy{}

// expBackoffState is the per-program state for ExpBackoffStrategy.
type expBackoffState struct {
	mu              sync.Mutex
	attainedService float64
	// lastServeTime is when this program was last served; deliberately never decayed.
	lastServeTime time.Time
	// queueDepthEWMA is a smoothed queue depth for congestion awareness (alpha = 0.3).
	queueDepthEWMA float64
	decayAnchor    time.Time
}

func (s *expBackoffState) Service() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.attainedService
}

// AddService charges the completion cost and marks the program as just served.
func (s *expBackoffState) AddService(cost float64) float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.attainedService += cost
	now := time.Now()
	s.lastServeTime = now
	s.decayAnchor = now
	return s.attainedService
}

// UpdateQueueDepth folds the current queue depth into an EWMA (alpha = 0.3).
func (s *expBackoffState) UpdateQueueDepth(depth float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	const alpha = 0.3
	if s.queueDepthEWMA == 0 {
		s.queueDepthEWMA = depth
	} else {
		s.queueDepthEWMA = alpha*depth + (1-alpha)*s.queueDepthEWMA
	}
}

func (s *expBackoffState) GetLastServeTime() time.Time {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastServeTime
}

func (s *expBackoffState) GetQueueDepth() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.queueDepthEWMA
}

// Decay applies idle-only decay to attainedService and queueDepthEWMA. It
// deliberately does NOT decay lastServeTime, so the backoff clock keeps running.
func (s *expBackoffState) Decay(now time.Time, halfLifeSeconds, factor float64) {
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
		f := math.Pow(0.5, elapsed/halfLifeSeconds)
		s.attainedService *= f
		s.queueDepthEWMA *= f
		s.decayAnchor = now
		return
	}
	s.attainedService *= factor
	s.queueDepthEWMA *= factor
}

var expBackoffAttainedServiceTokens = prometheus.NewGaugeVec(
	prometheus.GaugeOpts{
		Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:      "program_aware_exp_backoff_attained_service_tokens",
		Help:      metricsutil.HelpMsgWithStability("Time-decayed attained service (weighted tokens consumed) per program, exp-backoff strategy.", compbasemetrics.ALPHA),
	},
	[]string{"program_id"},
)

func (s *ExpBackoffStrategy) getOrCreateState(id string) *expBackoffState {
	if a, ok := s.state.Load(id); ok {
		if st, ok := a.(*expBackoffState); ok {
			return st
		}
	}
	fresh := &expBackoffState{}
	actual, _ := s.state.LoadOrStore(id, fresh)
	if st, ok := actual.(*expBackoffState); ok {
		return st
	}
	s.state.Store(id, fresh)
	return fresh
}

func (s *ExpBackoffStrategy) Name() string { return "exp-backoff" }

func (s *ExpBackoffStrategy) Pick(_ int, queues map[string]QueueInfo) flowcontrol.FlowQueueAccessor {
	now := time.Now()

	// First pass: collect all candidates and compute normalization factors.
	type candidate struct {
		id            string
		headWaitMs    float64
		queue         flowcontrol.FlowQueueAccessor
		lastServeTime time.Time
		queueDepth    float64
	}
	var candidates []candidate
	maxTimeSinceLast := 0.0
	maxHeadWait := 0.0
	totalHeadWait := 0.0
	activeCount := 0

	// Sort IDs for deterministic iteration.
	idsSorted := make([]string, 0, len(queues))
	for id := range queues {
		idsSorted = append(idsSorted, id)
	}
	sort.Strings(idsSorted)

	for _, id := range idsSorted {
		qi := queues[id]
		if qi.Metrics == nil || qi.Len == 0 {
			continue
		}
		head := qi.Queue.Peek()
		if head == nil {
			continue
		}
		headWait := now.Sub(head.EnqueueTime())
		headWaitMs := headWait.Seconds() * 1000

		st := s.getOrCreateState(id)
		// Apply decay to service cost (but NOT lastServeTime).
		st.Decay(now, s.halfLifeSeconds, s.decayFactor)

		// Update queue depth tracking.
		st.UpdateQueueDepth(float64(qi.Len))

		lastServeTime := st.GetLastServeTime()
		// On first tick, initialize to 60s ago to avoid initial starvation.
		if lastServeTime.IsZero() {
			lastServeTime = now.Add(-60 * time.Second)
		}

		candidates = append(candidates, candidate{
			id:            id,
			headWaitMs:    headWaitMs,
			queue:         qi.Queue,
			lastServeTime: lastServeTime,
			queueDepth:    st.GetQueueDepth(),
		})

		timeSinceLast := now.Sub(lastServeTime).Seconds()
		if timeSinceLast > maxTimeSinceLast {
			maxTimeSinceLast = timeSinceLast
		}
		if headWaitMs > maxHeadWait {
			maxHeadWait = headWaitMs
		}
		totalHeadWait += headWaitMs
		activeCount++
	}

	if len(candidates) == 0 {
		return nil
	}

	// Avoid division by zero.
	if maxTimeSinceLast < 1.0 {
		maxTimeSinceLast = 1.0
	}
	if maxHeadWait < 1.0 {
		maxHeadWait = 1.0
	}
	avgHeadWait := totalHeadWait / float64(activeCount)

	// Beta controls exponential aggressiveness. Tuned for congested workload:
	// beta=0.8 gives strong prioritization to neglected programs.
	const beta = 0.8

	type scored struct {
		score float64
		queue flowcontrol.FlowQueueAccessor
		id    string
	}
	var scoredCandidates []scored

	for _, c := range candidates {
		timeSinceLast := now.Sub(c.lastServeTime).Seconds()
		// Clamp to avoid overflow (max 1 hour).
		if timeSinceLast > 3600 {
			timeSinceLast = 3600
		}

		// Exponential backoff factor: ~2.2x boost at max neglect.
		expFactor := math.Exp(beta * timeSinceLast / maxTimeSinceLast)

		// Normalize head wait: sqrt to moderate extreme waits.
		waitUrgency := c.headWaitMs / math.Max(avgHeadWait, 1.0)
		waitUrgency = math.Sqrt(waitUrgency)

		// Queue depth bonus: prefer deeper queues (more waiting work).
		queueBonus := 1.0 + 0.1*c.queueDepth

		// Combined score: exponential backoff * wait urgency * queue bonus.
		score := expFactor * waitUrgency * queueBonus

		// Small tiebreaker on head wait to prefer longer-waiting when exp is similar.
		score += c.headWaitMs * 0.0001

		scoredCandidates = append(scoredCandidates, scored{
			score: score,
			queue: c.queue,
			id:    c.id,
		})
	}

	// Deterministic sort by score descending, then by ID for ties.
	sort.SliceStable(scoredCandidates, func(i, j int) bool {
		if scoredCandidates[i].score != scoredCandidates[j].score {
			return scoredCandidates[i].score > scoredCandidates[j].score
		}
		return scoredCandidates[i].id < scoredCandidates[j].id
	})

	if len(scoredCandidates) > 0 {
		return scoredCandidates[0].queue
	}
	return nil
}

func (s *ExpBackoffStrategy) OnPreRequest(_ *ProgramMetrics, _ *fwksched.InferenceRequest) {}

func (s *ExpBackoffStrategy) OnCompleted(_ *ProgramMetrics, request *fwksched.InferenceRequest, response *fwkrc.Response) {
	if request == nil || response == nil {
		return
	}
	promptTokens := int64(response.Usage.PromptTokens)
	completionTokens := int64(response.Usage.CompletionTokens)
	cost := float64(promptTokens)*1.0 + float64(completionTokens)*2.0
	id := programIDFor(request)
	service := s.getOrCreateState(id).AddService(cost)
	expBackoffAttainedServiceTokens.WithLabelValues(id).Set(service)
}

func (s *ExpBackoffStrategy) EvictProgram(id string) {
	s.state.Delete(id)
	expBackoffAttainedServiceTokens.DeleteLabelValues(id)
}

func (s *ExpBackoffStrategy) Collectors() []prometheus.Collector {
	return []prometheus.Collector{expBackoffAttainedServiceTokens}
}
