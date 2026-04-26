package programaware

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/flowcontrol"
	fcmocks "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/flowcontrol/mocks"
	requestcontrol "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/requestcontrol"
	requesthandling "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/requesthandling"
	scheduling "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
)

// testDRR returns a DRRStrategy with default weights for tests.
func testDRR() *DRRStrategy {
	return &DRRStrategy{
		weightDeficit:  defaultDRRWeightDeficit,
		weightHeadWait: defaultDRRWeightHeadWait,
		quantumTokens:  defaultDRRQuantumTokens,
	}
}

// testRequest returns an InferenceRequest for OnPreRequest tests.
func testRequest() *scheduling.InferenceRequest {
	return &scheduling.InferenceRequest{}
}

func TestNewStrategy_Valid(t *testing.T) {
	s, err := newStrategy(Config{Strategy: "drr"})
	require.NoError(t, err)
	assert.Equal(t, "drr", s.Name())

	s, err = newStrategy(Config{Strategy: "las"})
	require.NoError(t, err)
	assert.Equal(t, "las", s.Name())

	s, err = newStrategy(Config{Strategy: ""})
	require.NoError(t, err)
	assert.Equal(t, "las", s.Name())

	s, err = newStrategy(Config{Strategy: "evolved"})
	require.NoError(t, err)
	assert.Equal(t, "evolved", s.Name())
}

func TestNewStrategy_Invalid(t *testing.T) {
	_, err := newStrategy(Config{Strategy: "unknown"})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unknown scoring strategy")
}

func TestFactory_StrategyConfig(t *testing.T) {
	p, err := ProgramAwarePluginFactory("test", []byte(`{"strategy":"drr"}`), nil)
	require.NoError(t, err)
	plugin := p.(*ProgramAwarePlugin)
	assert.Equal(t, "drr", plugin.strategy.Name())
}

func TestFactory_DefaultStrategy(t *testing.T) {
	p, err := ProgramAwarePluginFactory("test", nil, nil)
	require.NoError(t, err)
	plugin := p.(*ProgramAwarePlugin)
	assert.Equal(t, "las", plugin.strategy.Name())
}

func TestFactory_InvalidStrategy(t *testing.T) {
	_, err := ProgramAwarePluginFactory("test", []byte(`{"strategy":"wfq"}`), nil)
	assert.Error(t, err)
}

// =============================================================================
// DRR Strategy tests
// =============================================================================

// makeQueueInfo builds a QueueInfo with a mock queue for testing.
func makeQueueInfo(id string, queueLen int, metrics *ProgramMetrics, enqueueTime time.Time) QueueInfo {
	return QueueInfo{
		Queue: &fcmocks.MockFlowQueueAccessor{
			LenV:     queueLen,
			FlowKeyV: flowcontrol.FlowKey{ID: id},
			PeekHeadV: &fcmocks.MockQueueItemAccessor{
				EnqueueTimeV:     enqueueTime,
				OriginalRequestV: &fcmocks.MockFlowControlRequest{IDV: id + "-req"},
			},
		},
		Metrics: metrics,
		Len:     queueLen,
	}
}

// makeEmptyQueueInfo builds a QueueInfo for an empty queue.
func makeEmptyQueueInfo(id string, metrics *ProgramMetrics) QueueInfo {
	return QueueInfo{
		Queue: &fcmocks.MockFlowQueueAccessor{
			LenV:     0,
			FlowKeyV: flowcontrol.FlowKey{ID: id},
		},
		Metrics: metrics,
		Len:     0,
	}
}

func TestDRRStrategy_Pick_AllocatesQuantum(t *testing.T) {
	s := testDRR()
	m := &ProgramMetrics{}
	now := time.Now()

	queues := map[string]QueueInfo{"prog": makeQueueInfo("prog", 3, m, now)}
	s.Pick(0, queues)

	assert.Equal(t, defaultDRRQuantumTokens, m.Deficit(), "non-empty queue should receive quantum")
}

func TestDRRStrategy_Pick_QuantumAccumulates(t *testing.T) {
	s := testDRR()
	m := &ProgramMetrics{}
	now := time.Now()

	for range 5 {
		queues := map[string]QueueInfo{"prog": makeQueueInfo("prog", 1, m, now)}
		s.Pick(0, queues)
		s.OnPreRequest(nil, testRequest()) // reset for next dispatch cycle
	}
	assert.Equal(t, defaultDRRQuantumTokens*5, m.Deficit(), "deficit should accumulate across rounds")
}

func TestDRRStrategy_Pick_IdleNoQuantum(t *testing.T) {
	s := testDRR()
	m := &ProgramMetrics{}
	now := time.Now()

	// Accumulate 3 rounds of quantum while active.
	for range 3 {
		queues := map[string]QueueInfo{"prog": makeQueueInfo("prog", 2, m, now)}
		s.Pick(0, queues)
		s.OnPreRequest(nil, testRequest())
	}
	assert.Equal(t, defaultDRRQuantumTokens*3, m.Deficit())

	// Queue drains — deficit should NOT increase (no quantum for empty queues).
	queues := map[string]QueueInfo{"prog": makeEmptyQueueInfo("prog", m)}
	s.Pick(0, queues)
	assert.Equal(t, defaultDRRQuantumTokens*3, m.Deficit(), "idle queues do not receive quantum")
}

func TestDRRStrategy_OnCompleted_DeductsTokens(t *testing.T) {
	s := testDRR()
	m := &ProgramMetrics{}
	m.AddDeficit(defaultDRRQuantumTokens) // one round of quantum

	resp := &requestcontrol.Response{Usage: requesthandling.Usage{PromptTokens: 700, CompletionTokens: 300}}
	s.OnCompleted(m, nil, resp) // weighted cost: 700*1 + 300*2 = 1300
	assert.Equal(t, int64(-300), m.Deficit(), "weighted 1300-token cost against 1000 quantum")
}

func TestDRRStrategy_OnCompleted_GoesNegativeOnOveruse(t *testing.T) {
	s := testDRR()
	m := &ProgramMetrics{}
	m.AddDeficit(defaultDRRQuantumTokens) // 1000 tokens

	resp := &requestcontrol.Response{Usage: requesthandling.Usage{PromptTokens: 1500, CompletionTokens: 500}}
	s.OnCompleted(m, nil, resp) // weighted cost: 1500*1 + 500*2 = 2500
	assert.Equal(t, int64(-1500), m.Deficit(), "deficit should be negative after overuse")
}

func TestDRRStrategy_Pick_PreferHighDeficit(t *testing.T) {
	s := testDRR()
	now := time.Now()

	mHigh := &ProgramMetrics{}
	mHigh.AddDeficit(20000)

	mLow := &ProgramMetrics{}
	mLow.DeductTokens(20000)

	queues := map[string]QueueInfo{
		"high": makeQueueInfo("high", 1, mHigh, now),
		"low":  makeQueueInfo("low", 1, mLow, now),
	}

	selected, scores := s.Pick(0, queues)
	require.NotNil(t, selected)
	assert.Equal(t, "high", selected.FlowKey().ID)
	assert.Greater(t, scores["high"], scores["low"],
		"high-deficit queue should outscore overserved queue")
}

func TestDRRStrategy_Pick_QuantumOncePerCycle(t *testing.T) {
	s := testDRR()
	m := &ProgramMetrics{}
	now := time.Now()

	// First Pick allocates quantum.
	queues := map[string]QueueInfo{"prog": makeQueueInfo("prog", 3, m, now)}
	s.Pick(0, queues)
	assert.Equal(t, defaultDRRQuantumTokens, m.Deficit(), "first Pick should allocate quantum")

	// Second Pick without OnPrerequest — same program already seen, no extra quantum.
	s.Pick(0, queues)
	assert.Equal(t, defaultDRRQuantumTokens, m.Deficit(), "second Pick without OnPrerequest should not allocate again")
}

func TestDRRStrategy_OnPrerequest_ResetsQuantum(t *testing.T) {
	s := testDRR()
	m := &ProgramMetrics{}
	now := time.Now()

	queues := map[string]QueueInfo{"prog": makeQueueInfo("prog", 3, m, now)}
	s.Pick(0, queues)
	assert.Equal(t, defaultDRRQuantumTokens, m.Deficit())

	// OnPrerequest resets the cycle — next Pick should allocate again.
	s.OnPreRequest(nil, testRequest())
	s.Pick(0, queues)
	assert.Equal(t, defaultDRRQuantumTokens*2, m.Deficit(), "Pick after OnPrerequest should allocate quantum again")
}

// =============================================================================
// DRR deficit decay tests
// =============================================================================

func testDRRWithDecay(halfLife float64) *DRRStrategy {
	return &DRRStrategy{
		weightDeficit:          defaultDRRWeightDeficit,
		weightHeadWait:         defaultDRRWeightHeadWait,
		quantumTokens:          defaultDRRQuantumTokens,
		deficitHalfLifeSeconds: halfLife,
	}
}

func TestDRRStrategy_DecayDeficit_FirstCallNoDecay(t *testing.T) {
	m := &ProgramMetrics{}
	m.AddDeficit(10000)

	m.DecayDeficitTimed(60.0, time.Now())
	assert.Equal(t, int64(10000), m.Deficit(),
		"first decay call should not reduce deficit (only sets timestamp)")
}

func TestDRRStrategy_DecayDeficit_HalvesAtHalfLife(t *testing.T) {
	m := &ProgramMetrics{}
	m.AddDeficit(10000)

	now := time.Now()
	m.DecayDeficitTimed(60.0, now)                     // initialize
	m.DecayDeficitTimed(60.0, now.Add(60*time.Second)) // one half-life

	assert.Equal(t, int64(5000), m.Deficit(),
		"deficit should halve after exactly one half-life")
}

func TestDRRStrategy_DecayDeficit_NegativeDeficitDecays(t *testing.T) {
	m := &ProgramMetrics{}
	m.DeductTokens(10000) // deficit = -10000

	now := time.Now()
	m.DecayDeficitTimed(60.0, now)
	m.DecayDeficitTimed(60.0, now.Add(60*time.Second))

	assert.Equal(t, int64(-5000), m.Deficit(),
		"negative deficit should also decay toward zero")
}

func TestDRRStrategy_DecayDeficit_ConsistentWindow(t *testing.T) {
	// Single call over 60s vs many calls over 60s should yield same result.
	m1 := &ProgramMetrics{}
	m1.AddDeficit(10000)
	now := time.Now()
	m1.DecayDeficitTimed(60.0, now)
	m1.DecayDeficitTimed(60.0, now.Add(60*time.Second))
	single := m1.Deficit()

	m2 := &ProgramMetrics{}
	m2.AddDeficit(10000)
	m2.DecayDeficitTimed(60.0, now)
	for i := 1; i <= 100; i++ {
		m2.DecayDeficitTimed(60.0, now.Add(time.Duration(i)*600*time.Millisecond))
	}
	many := m2.Deficit()

	// Wider tolerance than LAS (which uses float64) because int64 truncation
	// on each intermediate step compounds rounding error.
	assert.InDelta(t, float64(single), float64(many), 50.0,
		"time-based decay should be consistent regardless of call frequency")
}

func TestDRRStrategy_Pick_IdleDecaysDeficit(t *testing.T) {
	s := testDRRWithDecay(60.0)
	m := &ProgramMetrics{}
	m.AddDeficit(10000)
	now := time.Now()

	// Initialize the decay timer.
	m.DecayDeficitTimed(60.0, now)

	// Simulate 60s later (one half-life) — deficit should halve.
	m.DecayDeficitTimed(60.0, now.Add(60*time.Second))
	assert.Equal(t, int64(5000), m.Deficit(),
		"idle queue deficit should halve after one half-life")

	// Verify Pick on non-empty queue adds quantum without decay.
	m2 := &ProgramMetrics{}
	m2.AddDeficit(10000)
	queues2 := map[string]QueueInfo{"prog": makeQueueInfo("prog", 1, m2, now)}
	s.Pick(0, queues2)
	assert.Equal(t, 10000+defaultDRRQuantumTokens, m2.Deficit(),
		"non-empty queue should get quantum, no decay")
}

func TestDRRStrategy_Pick_NoDecayOnNonEmpty(t *testing.T) {
	s := testDRRWithDecay(60.0)
	m := &ProgramMetrics{}
	now := time.Now()

	for range 100 {
		queues := map[string]QueueInfo{"prog": makeQueueInfo("prog", 1, m, now)}
		s.Pick(0, queues)
		s.OnPreRequest(nil, testRequest())
	}
	// All 100 quanta should have accumulated — no decay on non-empty queues.
	assert.Equal(t, defaultDRRQuantumTokens*100, m.Deficit(),
		"non-empty queues should not be decayed")
}

func TestDRRStrategy_Pick_NoDecayWhenDisabled(t *testing.T) {
	s := testDRR() // deficitHalfLifeSeconds = 0 (no decay)
	m := &ProgramMetrics{}
	m.AddDeficit(10000)

	// Empty queue with decay disabled — deficit should not change.
	queues := map[string]QueueInfo{"prog": makeEmptyQueueInfo("prog", m)}
	s.Pick(0, queues)
	assert.Equal(t, int64(10000), m.Deficit(),
		"with deficitHalfLifeSeconds=0, no decay should occur")
}

func TestFactory_DeficitHalfLifeSeconds(t *testing.T) {
	p, err := ProgramAwarePluginFactory("test", []byte(`{"strategy":"drr","deficitHalfLifeSeconds":30}`), nil)
	require.NoError(t, err)
	plugin := p.(*ProgramAwarePlugin)
	drr := plugin.strategy.(*DRRStrategy)
	assert.Equal(t, 30.0, drr.deficitHalfLifeSeconds)
}

func TestFactory_DeficitHalfLifeSecondsDefault(t *testing.T) {
	p, err := ProgramAwarePluginFactory("test", []byte(`{"strategy":"drr"}`), nil)
	require.NoError(t, err)
	plugin := p.(*ProgramAwarePlugin)
	drr := plugin.strategy.(*DRRStrategy)
	assert.Equal(t, defaultDRRDeficitHalfLifeSeconds, drr.deficitHalfLifeSeconds)
}

// =============================================================================
// DRR Pick integration tests
// =============================================================================

func TestDRR_Pick_TokenHeavyProgramDeprioritized(t *testing.T) {
	p := &ProgramAwarePlugin{strategy: testDRR()}

	// "heavy" has consumed many tokens relative to its quantum allocation.
	mHeavy := p.getOrCreateMetrics("heavy")
	mHeavy.AddDeficit(defaultDRRQuantumTokens * 2)
	mHeavy.DeductTokens(defaultDRRQuantumTokens * 10)

	// "light" has only consumed its fair share.
	mLight := p.getOrCreateMetrics("light")
	mLight.AddDeficit(defaultDRRQuantumTokens * 2)
	mLight.DeductTokens(defaultDRRQuantumTokens * 1)

	now := time.Now()
	queueHeavy := &fcmocks.MockFlowQueueAccessor{
		LenV:     5,
		FlowKeyV: flowcontrol.FlowKey{ID: "heavy"},
		PeekHeadV: &fcmocks.MockQueueItemAccessor{
			EnqueueTimeV:     now,
			OriginalRequestV: &fcmocks.MockFlowControlRequest{IDV: "heavy-req-1"},
		},
	}
	queueLight := &fcmocks.MockFlowQueueAccessor{
		LenV:     1,
		FlowKeyV: flowcontrol.FlowKey{ID: "light"},
		PeekHeadV: &fcmocks.MockQueueItemAccessor{
			EnqueueTimeV:     now,
			OriginalRequestV: &fcmocks.MockFlowControlRequest{IDV: "light-req-1"},
		},
	}

	band := &fcmocks.MockPriorityBandAccessor{
		IterateQueuesFunc: func(cb func(flowcontrol.FlowQueueAccessor) bool) {
			cb(queueHeavy)
			cb(queueLight)
		},
	}

	queue, err := p.Pick(context.Background(), band)
	require.NoError(t, err)
	assert.Equal(t, "light", queue.FlowKey().ID,
		"light program (fair deficit) should be preferred over token-heavy program")
}

// =============================================================================
// LAS Strategy tests
// =============================================================================

// testService returns a LASStrategy with default weights for tests.
func testService() *LASStrategy {
	return &LASStrategy{
		weightService:  defaultServiceWeightService,
		weightHeadWait: defaultServiceWeightHeadWait,
		decayFactor:    defaultServiceDecayFactor,
	}
}

func TestLASStrategy_Name(t *testing.T) {
	s := testService()
	assert.Equal(t, "las", s.Name())
}

func TestLASStrategy_Pick_DecaysService(t *testing.T) {
	s := testService()
	m := &ProgramMetrics{}
	m.AddService(1000.0)
	now := time.Now()

	queues := map[string]QueueInfo{"prog": makeQueueInfo("prog", 5, m, now)}
	s.Pick(0, queues)

	assert.InDelta(t, 1000.0*defaultServiceDecayFactor, m.AttainedService(), 0.01,
		"Pick should decay attained service")
}

func TestLASStrategy_OnCompleted_AddsService(t *testing.T) {
	s := testService()
	m := &ProgramMetrics{}

	// 100 input + 50 output → weighted: 100*1 + 50*2 = 200
	resp := &requestcontrol.Response{Usage: requesthandling.Usage{PromptTokens: 100, CompletionTokens: 50}}
	s.OnCompleted(m, nil, resp)
	assert.InDelta(t, 200.0, m.AttainedService(), 0.01,
		"OnCompleted should add weighted token cost to attained service")
}

func TestLASStrategy_Pick_PreferLowService(t *testing.T) {
	s := testService()
	now := time.Now()

	mLow := &ProgramMetrics{}
	mLow.AddService(100.0)

	mHigh := &ProgramMetrics{}
	mHigh.AddService(10000.0)

	queues := map[string]QueueInfo{
		"low":  makeQueueInfo("low", 1, mLow, now),
		"high": makeQueueInfo("high", 1, mHigh, now),
	}

	selected, scores := s.Pick(0, queues)
	require.NotNil(t, selected)
	assert.Equal(t, "low", selected.FlowKey().ID)
	assert.Greater(t, scores["low"], scores["high"],
		"underserved queue should outscore overserved queue")
}

func TestLASStrategy_Pick_ColdStartUsesHeadWait(t *testing.T) {
	s := testService()

	mOld := &ProgramMetrics{}
	mNew := &ProgramMetrics{}

	queues := map[string]QueueInfo{
		"old": makeQueueInfo("old", 1, mOld, time.Now().Add(-500*time.Millisecond)),
		"new": makeQueueInfo("new", 1, mNew, time.Now()),
	}

	selected, scores := s.Pick(0, queues)
	require.NotNil(t, selected)
	assert.Equal(t, "old", selected.FlowKey().ID)
	assert.Greater(t, scores["old"], scores["new"],
		"with zero service, longer-waiting queue should outscore newer queue")
}

func TestLASStrategy_DecayForgetsOldService(t *testing.T) {
	s := testService()
	m := &ProgramMetrics{}
	m.AddService(1000.0)
	now := time.Now()

	// After many decay cycles, service should approach 0.
	for range 1000 {
		queues := map[string]QueueInfo{"prog": makeQueueInfo("prog", 1, m, now)}
		s.Pick(0, queues)
	}
	// 1000 * 0.995^1000 ≈ 6.7 — verify significant decay occurred.
	assert.Less(t, m.AttainedService(), 10.0,
		"after 1000 decay cycles, attained service should be nearly forgotten")
}

func TestNewStrategy_LAS(t *testing.T) {
	s, err := newStrategy(Config{Strategy: "las"})
	require.NoError(t, err)
	assert.Equal(t, "las", s.Name())
}

func TestFactory_LASStrategy(t *testing.T) {
	p, err := ProgramAwarePluginFactory("test", []byte(`{"strategy":"las"}`), nil)
	require.NoError(t, err)
	plugin := p.(*ProgramAwarePlugin)
	assert.Equal(t, "las", plugin.strategy.Name())
}

func testServiceTimed(halfLife float64) *LASStrategy {
	return &LASStrategy{
		weightService:   defaultServiceWeightService,
		weightHeadWait:  defaultServiceWeightHeadWait,
		halfLifeSeconds: halfLife,
	}
}

func TestLASStrategy_TimedDecay_FirstCallNoDecay(t *testing.T) {
	m := &ProgramMetrics{}
	m.AddService(1000.0)

	// First call should set lastDecayTime but not decay.
	m.DecayServiceTimed(30.0, time.Now())
	assert.InDelta(t, 1000.0, m.AttainedService(), 0.01,
		"first timed decay call should not reduce service")
}

func TestLASStrategy_TimedDecay_HalvesAtHalfLife(t *testing.T) {
	m := &ProgramMetrics{}
	m.AddService(1000.0)

	now := time.Now()
	m.DecayServiceTimed(30.0, now)                     // initialize lastDecayTime
	m.DecayServiceTimed(30.0, now.Add(30*time.Second)) // exactly one half-life later

	assert.InDelta(t, 500.0, m.AttainedService(), 1.0,
		"service should halve after exactly one half-life")
}

func TestLASStrategy_TimedDecay_ConsistentWindow(t *testing.T) {
	// Whether we call DecayServiceTimed once after 30s or 100 times over 30s,
	// the result should be the same.
	m1 := &ProgramMetrics{}
	m1.AddService(1000.0)
	now := time.Now()
	m1.DecayServiceTimed(30.0, now)
	m1.DecayServiceTimed(30.0, now.Add(30*time.Second))
	singleCall := m1.AttainedService()

	m2 := &ProgramMetrics{}
	m2.AddService(1000.0)
	m2.DecayServiceTimed(30.0, now)
	for i := 1; i <= 100; i++ {
		m2.DecayServiceTimed(30.0, now.Add(time.Duration(i)*300*time.Millisecond))
	}
	manyCalls := m2.AttainedService()

	assert.InDelta(t, singleCall, manyCalls, 1.0,
		"time-based decay should produce same result regardless of call frequency")
}

func TestLASStrategy_Pick_UsesTimedDecay(t *testing.T) {
	s := testServiceTimed(30.0)
	m := &ProgramMetrics{}
	m.AddService(1000.0)
	now := time.Now()

	// First Pick initializes timer.
	queues := map[string]QueueInfo{"prog": makeQueueInfo("prog", 1, m, now)}
	s.Pick(0, queues)
	assert.InDelta(t, 1000.0, m.AttainedService(), 0.01)
}

func TestFactory_ServiceHalfLifeSeconds(t *testing.T) {
	p, err := ProgramAwarePluginFactory("test", []byte(`{"strategy":"las","serviceHalfLifeSeconds":30}`), nil)
	require.NoError(t, err)
	plugin := p.(*ProgramAwarePlugin)
	svc := plugin.strategy.(*LASStrategy)
	assert.Equal(t, 30.0, svc.halfLifeSeconds)
}

// =============================================================================
// RR (Round-Robin) Strategy tests
// =============================================================================

func TestNewStrategy_RR(t *testing.T) {
	s, err := newStrategy(Config{Strategy: "rr"})
	require.NoError(t, err)
	assert.Equal(t, "rr", s.Name())
}

func TestFactory_RRStrategy(t *testing.T) {
	p, err := ProgramAwarePluginFactory("test", []byte(`{"strategy":"rr"}`), nil)
	require.NoError(t, err)
	plugin := p.(*ProgramAwarePlugin)
	assert.Equal(t, "rr", plugin.strategy.Name())
}

// simulateRRCycle runs one Pick() cycle on the RRStrategy directly.
// Returns the selected program ID (or "" if none).
func simulateRRCycle(s *RRStrategy, allIDs []string, nonEmptyIDs []string) string {
	now := time.Now()

	// Build QueueInfo map: empty queues for IDs not in nonEmptyIDs.
	nonEmptySet := make(map[string]bool, len(nonEmptyIDs))
	for _, id := range nonEmptyIDs {
		nonEmptySet[id] = true
	}

	queues := make(map[string]QueueInfo, len(allIDs))
	for _, id := range allIDs {
		if nonEmptySet[id] {
			queues[id] = makeQueueInfo(id, 1, nil, now)
		} else {
			queues[id] = makeEmptyQueueInfo(id, nil)
		}
	}

	selected, _ := s.Pick(0, queues)
	if selected == nil {
		return ""
	}
	return selected.FlowKey().ID
}

func TestRRStrategy_BasicCycle(t *testing.T) {
	s := &RRStrategy{}
	ids := []string{"alpha", "beta", "gamma"}

	// Sorted order: alpha, beta, gamma.
	// With no lastSelected, startIndex=0 → alpha first.
	picked := simulateRRCycle(s, ids, ids)
	assert.Equal(t, "alpha", picked, "first cycle should pick alpha (first in sorted order)")

	// After alpha, cursor advances → beta.
	picked = simulateRRCycle(s, ids, ids)
	assert.Equal(t, "beta", picked, "second cycle should pick beta")

	// After beta → gamma.
	picked = simulateRRCycle(s, ids, ids)
	assert.Equal(t, "gamma", picked, "third cycle should pick gamma")

	// After gamma → wraps to alpha.
	picked = simulateRRCycle(s, ids, ids)
	assert.Equal(t, "alpha", picked, "fourth cycle should wrap back to alpha")
}

func TestRRStrategy_SkipsEmptyQueues(t *testing.T) {
	s := &RRStrategy{}
	allIDs := []string{"alpha", "beta", "gamma"}
	nonEmpty := []string{"alpha", "gamma"} // beta is empty

	// Sorted: alpha, beta, gamma. startIndex=0 → alpha wins.
	picked := simulateRRCycle(s, allIDs, nonEmpty)
	assert.Equal(t, "alpha", picked)

	// After alpha, startIndex=1 (beta). beta is empty.
	// gamma: distance = (2-1+3)%3 = 1, score = 3-1 = 2
	// alpha: distance = (0-1+3)%3 = 2, score = 3-2 = 1
	// gamma wins.
	picked = simulateRRCycle(s, allIDs, nonEmpty)
	assert.Equal(t, "gamma", picked, "should skip empty beta and pick gamma")

	// After gamma, startIndex=0 (wraps). alpha wins again.
	picked = simulateRRCycle(s, allIDs, nonEmpty)
	assert.Equal(t, "alpha", picked, "should wrap back to alpha")
}

func TestRRStrategy_WrapAround(t *testing.T) {
	s := &RRStrategy{}

	// Set cursor to "c" (last in sorted order).
	s.lastSelected.Store(0, "c")

	ids := []string{"a", "b", "c"}
	// Next cycle: startIndex = (index_of_c + 1) % 3 = 0 → "a" should win.
	picked := simulateRRCycle(s, ids, ids)
	assert.Equal(t, "a", picked, "should wrap from c to a")
}

func TestRRStrategy_SingleQueue(t *testing.T) {
	s := &RRStrategy{}
	ids := []string{"solo"}

	for i := range 5 {
		picked := simulateRRCycle(s, ids, ids)
		assert.Equal(t, "solo", picked, "single queue should always be picked (iteration %d)", i)
	}
}

func TestRRStrategy_Pick_UpdatesCursor(t *testing.T) {
	s := &RRStrategy{}

	queues := map[string]QueueInfo{
		"alpha": makeQueueInfo("alpha", 1, nil, time.Now()),
		"beta":  makeQueueInfo("beta", 1, nil, time.Now()),
	}

	selected, _ := s.Pick(0, queues)
	require.NotNil(t, selected)

	v, ok := s.lastSelected.Load(0)
	assert.True(t, ok)
	assert.Equal(t, selected.FlowKey().ID, v.(string), "lastSelected should be updated after Pick")
}

func TestRRStrategy_NoQueues(t *testing.T) {
	s := &RRStrategy{}

	// Empty cycle — no queues at all.
	picked := simulateRRCycle(s, nil, nil)
	assert.Equal(t, "", picked, "no queues should yield empty pick")
}

func TestFactory_RRDeferCursor(t *testing.T) {
	p, err := ProgramAwarePluginFactory("test", []byte(`{"strategy":"rr","deferRRCursor":true}`), nil)
	require.NoError(t, err)
	plugin := p.(*ProgramAwarePlugin)
	rr := plugin.strategy.(*RRStrategy)
	assert.True(t, rr.deferCursor)
}

func TestRRStrategy_DeferCursor_PickDoesNotAdvance(t *testing.T) {
	s := &RRStrategy{deferCursor: true}
	ids := []string{"alpha", "beta", "gamma"}

	picked := simulateRRCycle(s, ids, ids)
	assert.Equal(t, "alpha", picked)

	v, ok := s.lastSelected.Load(0)
	assert.True(t, ok)
	assert.Equal(t, "alpha", v.(string), "first Pick advances lastSelected")

	// Second Pick without OnPreRequest returns same queue.
	picked = simulateRRCycle(s, ids, ids)
	assert.Equal(t, "alpha", picked, "repeated Pick without OnPreRequest returns same queue")
}

func TestRRStrategy_DeferCursor_OnPreRequestCommits(t *testing.T) {
	s := &RRStrategy{deferCursor: true}
	ids := []string{"alpha", "beta", "gamma"}

	simulateRRCycle(s, ids, ids)       // picks alpha, sets moveCursor=false
	s.OnPreRequest(nil, testRequest()) // resets moveCursor=true

	v, _ := s.lastSelected.Load(0)
	assert.Equal(t, "alpha", v.(string))

	// Next pick should advance past alpha.
	picked := simulateRRCycle(s, ids, ids)
	assert.Equal(t, "beta", picked)
}

func TestRRStrategy_DeferCursor_RepeatedPickWithoutDispatch(t *testing.T) {
	s := &RRStrategy{deferCursor: true}
	ids := []string{"alpha", "beta", "gamma"}

	// Pick three times without OnPreRequest — cursor never advances.
	for range 3 {
		picked := simulateRRCycle(s, ids, ids)
		assert.Equal(t, "alpha", picked, "without OnPreRequest, alpha is always picked")
	}
}

func TestRRStrategy_DeferCursor_FullCycle(t *testing.T) {
	s := &RRStrategy{deferCursor: true}
	ids := []string{"alpha", "beta", "gamma"}

	expected := []string{"alpha", "beta", "gamma", "alpha"}
	for _, want := range expected {
		picked := simulateRRCycle(s, ids, ids)
		assert.Equal(t, want, picked)
		s.OnPreRequest(nil, testRequest()) // commit after dispatch
	}
}

func TestRRStrategy_DeferCursor_AllEmpty(t *testing.T) {
	s := &RRStrategy{deferCursor: true}
	allIDs := []string{"alpha", "beta"}

	picked := simulateRRCycle(s, allIDs, nil) // all empty
	assert.Equal(t, "", picked)

	_, ok := s.lastSelected.Load(0)
	assert.False(t, ok, "lastSelected should be cleared when all queues are empty")
}

func TestRR_Pick_CyclesThroughPrograms(t *testing.T) {
	p := &ProgramAwarePlugin{strategy: &RRStrategy{}}

	now := time.Now()
	makeQueue := func(id string) *fcmocks.MockFlowQueueAccessor {
		return &fcmocks.MockFlowQueueAccessor{
			LenV:     1,
			FlowKeyV: flowcontrol.FlowKey{ID: id},
			PeekHeadV: &fcmocks.MockQueueItemAccessor{
				EnqueueTimeV:     now,
				OriginalRequestV: &fcmocks.MockFlowControlRequest{IDV: id + "-req"},
			},
		}
	}

	band := &fcmocks.MockPriorityBandAccessor{
		IterateQueuesFunc: func(cb func(flowcontrol.FlowQueueAccessor) bool) {
			cb(makeQueue("alpha"))
			cb(makeQueue("beta"))
			cb(makeQueue("gamma"))
		},
	}

	// Three picks should cycle alpha → beta → gamma.
	expected := []string{"alpha", "beta", "gamma"}
	for i, want := range expected {
		queue, err := p.Pick(context.Background(), band)
		require.NoError(t, err)
		assert.Equal(t, want, queue.FlowKey().ID,
			"Pick #%d should select %s", i+1, want)
	}

	// Fourth pick wraps to alpha.
	queue, err := p.Pick(context.Background(), band)
	require.NoError(t, err)
	assert.Equal(t, "alpha", queue.FlowKey().ID, "Pick #4 should wrap to alpha")
}

func TestDRR_Pick_QuantumAllocatedDuringPick(t *testing.T) {
	p := &ProgramAwarePlugin{strategy: testDRR()}

	// Two fresh programs with no prior state.
	_ = p.getOrCreateMetrics("alpha")
	_ = p.getOrCreateMetrics("beta")

	now := time.Now()
	makeQueue := func(id string) *fcmocks.MockFlowQueueAccessor {
		return &fcmocks.MockFlowQueueAccessor{
			LenV:     1,
			FlowKeyV: flowcontrol.FlowKey{ID: id},
			PeekHeadV: &fcmocks.MockQueueItemAccessor{
				EnqueueTimeV:     now,
				OriginalRequestV: &fcmocks.MockFlowControlRequest{IDV: id + "-req"},
			},
		}
	}

	band := &fcmocks.MockPriorityBandAccessor{
		IterateQueuesFunc: func(cb func(flowcontrol.FlowQueueAccessor) bool) {
			cb(makeQueue("alpha"))
			cb(makeQueue("beta"))
		},
	}

	_, err := p.Pick(context.Background(), band)
	require.NoError(t, err)

	// Both queues should have received a quantum during Pick().
	alphaMetrics, _ := p.programMetrics.Load("alpha")
	betaMetrics, _ := p.programMetrics.Load("beta")

	// Deficit after Pick() = quantumTokens (added by strategy.Pick(), not yet deducted)
	assert.Equal(t, defaultDRRQuantumTokens, alphaMetrics.(*ProgramMetrics).Deficit())
	assert.Equal(t, defaultDRRQuantumTokens, betaMetrics.(*ProgramMetrics).Deficit())
}

// =============================================================================
// Evolved (Two-Tier) Strategy tests
// =============================================================================

func testEvolved() *EvolvedStrategy {
	return &EvolvedStrategy{
		decayFactor: defaultEvolvedDecayFactor,
		tierOffset:  defaultEvolvedTierOffset,
	}
}

func TestEvolvedStrategy_Name(t *testing.T) {
	s := testEvolved()
	assert.Equal(t, "evolved", s.Name())
}

func TestNewStrategy_Evolved(t *testing.T) {
	s, err := newStrategy(Config{Strategy: "evolved"})
	require.NoError(t, err)
	assert.Equal(t, "evolved", s.Name())
}

func TestFactory_EvolvedStrategy(t *testing.T) {
	p, err := ProgramAwarePluginFactory("test", []byte(`{"strategy":"evolved"}`), nil)
	require.NoError(t, err)
	plugin := p.(*ProgramAwarePlugin)
	assert.Equal(t, "evolved", plugin.strategy.Name())
}

func TestFactory_EvolvedConfigValues(t *testing.T) {
	p, err := ProgramAwarePluginFactory("test", []byte(`{"strategy":"evolved","evolvedDecayFactor":0.999,"evolvedTierOffset":5000}`), nil)
	require.NoError(t, err)
	plugin := p.(*ProgramAwarePlugin)
	evolved := plugin.strategy.(*EvolvedStrategy)
	assert.Equal(t, 0.999, evolved.decayFactor)
	assert.Equal(t, 5000.0, evolved.tierOffset)
}

func TestEvolvedStrategy_Pick_EmptyQueues(t *testing.T) {
	s := testEvolved()
	selected, scores := s.Pick(0, map[string]QueueInfo{})
	assert.Nil(t, selected)
	assert.Nil(t, scores)
}

func TestEvolvedStrategy_Pick_SingleQueue(t *testing.T) {
	s := testEvolved()
	m := &ProgramMetrics{}
	m.SetFirstArrival(time.Now().Add(-1 * time.Second))
	m.totalRequests.Store(5)
	m.dispatchedCount.Store(3)
	now := time.Now()

	queues := map[string]QueueInfo{"prog": makeQueueInfo("prog", 2, m, now)}
	selected, scores := s.Pick(0, queues)
	require.NotNil(t, selected)
	assert.Equal(t, "prog", selected.FlowKey().ID)
	assert.Contains(t, scores, "prog")
}

func TestEvolvedStrategy_Pick_AllEmptyQueues(t *testing.T) {
	s := testEvolved()
	m := &ProgramMetrics{}

	queues := map[string]QueueInfo{"prog": makeEmptyQueueInfo("prog", m)}
	selected, scores := s.Pick(0, queues)
	assert.Nil(t, selected)
	assert.Nil(t, scores)
}

func TestEvolvedStrategy_Pick_DecayOnlyOnDispatch(t *testing.T) {
	s := testEvolved()
	m := &ProgramMetrics{}
	m.AddService(1000.0)
	m.SetFirstArrival(time.Now().Add(-5 * time.Second))
	m.totalRequests.Store(10)
	m.dispatchedCount.Store(5)
	now := time.Now()

	// Pick without shouldDecay flag — no decay should occur.
	queues := map[string]QueueInfo{"prog": makeQueueInfo("prog", 3, m, now)}
	s.Pick(0, queues)
	assert.InDelta(t, 1000.0, m.AttainedService(), 0.01,
		"Pick without shouldDecay flag should NOT decay service")

	// Simulate a dispatch: OnPreRequest sets shouldDecay.
	req := &scheduling.InferenceRequest{Objectives: scheduling.RequestObjectives{Priority: 0}}
	s.OnPreRequest(m, req)

	// Now Pick should decay.
	s.Pick(0, queues)
	expected := 1000.0 * defaultEvolvedDecayFactor
	assert.InDelta(t, expected, m.AttainedService(), 0.01,
		"Pick after OnPreRequest should decay service")
}

func TestEvolvedStrategy_Pick_FinishingTierWins(t *testing.T) {
	s := testEvolved()
	now := time.Now()

	// "finishing" program: high progress (>0.65 → auto-promote)
	mFinishing := &ProgramMetrics{}
	mFinishing.SetFirstArrival(now.Add(-10 * time.Second))
	mFinishing.totalRequests.Store(10)
	mFinishing.dispatchedCount.Store(8)
	mFinishing.totalInputTokens.Store(800)
	mFinishing.totalOutputTokens.Store(400)

	// "young" program: low progress, fresh
	mYoung := &ProgramMetrics{}
	mYoung.SetFirstArrival(now.Add(-1 * time.Second))
	mYoung.totalRequests.Store(10)
	mYoung.dispatchedCount.Store(2)
	mYoung.totalInputTokens.Store(200)
	mYoung.totalOutputTokens.Store(100)

	queues := map[string]QueueInfo{
		"finishing": makeQueueInfo("finishing", 2, mFinishing, now),
		"young":     makeQueueInfo("young", 5, mYoung, now),
	}

	selected, scores := s.Pick(0, queues)
	require.NotNil(t, selected)
	assert.Equal(t, "finishing", selected.FlowKey().ID,
		"finishing-tier program should be selected over young-tier program")
	assert.Greater(t, scores["finishing"], scores["young"],
		"finishing-tier score should exceed young-tier score by tier offset")
}

func TestEvolvedStrategy_Pick_HighProgressPromotion(t *testing.T) {
	s := testEvolved()
	now := time.Now()

	// Program with progress > 0.65 should be promoted to finishing tier.
	m := &ProgramMetrics{}
	m.SetFirstArrival(now.Add(-1 * time.Second)) // very fresh (wouldn't qualify by age)
	m.dispatchedCount.Store(7)                    // dispatched=7
	m.totalRequests.Store(10)
	m.totalInputTokens.Store(700)
	m.totalOutputTokens.Store(350)

	// qLen=1, inFlight=0 → remaining=1, progress = 7/(7+1) = 0.875 > 0.65
	queues := map[string]QueueInfo{
		"prog": makeQueueInfo("prog", 1, m, now),
	}

	selected, scores := s.Pick(0, queues)
	require.NotNil(t, selected)
	// Score should include tier offset (10000) since program is in finishing tier.
	assert.Greater(t, scores["prog"], defaultEvolvedTierOffset,
		"high-progress program should be promoted to finishing tier (score > tier offset)")
}

func TestEvolvedStrategy_Pick_EmergencyWaitPromotion(t *testing.T) {
	s := testEvolved()

	// Program waiting >200 seconds should be promoted to finishing tier.
	m := &ProgramMetrics{}
	m.SetFirstArrival(time.Now().Add(-250 * time.Second))
	m.totalRequests.Store(5)
	m.dispatchedCount.Store(1)
	m.totalInputTokens.Store(100)
	m.totalOutputTokens.Store(50)

	// Enqueue time 250s ago → head_wait > 200s
	enqueueTime := time.Now().Add(-250 * time.Second)
	queues := map[string]QueueInfo{
		"starved": makeQueueInfo("starved", 3, m, enqueueTime),
	}

	selected, scores := s.Pick(0, queues)
	require.NotNil(t, selected)
	assert.Greater(t, scores["starved"], defaultEvolvedTierOffset,
		"program waiting >200s should be emergency-promoted to finishing tier")
}

func TestEvolvedStrategy_Pick_LowRemainingPromotion(t *testing.T) {
	s := testEvolved()
	now := time.Now()

	// remaining <= 2 and dispatched > 0 → promoted
	m := &ProgramMetrics{}
	m.SetFirstArrival(now.Add(-500 * time.Millisecond))
	m.dispatchedCount.Store(5)
	m.totalRequests.Store(7)
	m.totalInputTokens.Store(500)
	m.totalOutputTokens.Store(250)

	// qLen=1, inFlight=0 → remaining=1
	queues := map[string]QueueInfo{
		"almost_done": makeQueueInfo("almost_done", 1, m, now),
	}

	selected, scores := s.Pick(0, queues)
	require.NotNil(t, selected)
	assert.Greater(t, scores["almost_done"], defaultEvolvedTierOffset,
		"program with remaining<=2 and dispatched>0 should be promoted")
}

func TestEvolvedStrategy_Pick_YoungTierFairShare(t *testing.T) {
	s := testEvolved()
	now := time.Now()

	// Two young programs, same age and queue size. One has lower attained service.
	mUnderserved := &ProgramMetrics{}
	mUnderserved.SetFirstArrival(now.Add(-500 * time.Millisecond))
	mUnderserved.totalRequests.Store(20)
	mUnderserved.dispatchedCount.Store(2)
	mUnderserved.AddService(100.0) // low service
	mUnderserved.totalInputTokens.Store(200)
	mUnderserved.totalOutputTokens.Store(100)

	mOverserved := &ProgramMetrics{}
	mOverserved.SetFirstArrival(now.Add(-500 * time.Millisecond))
	mOverserved.totalRequests.Store(20)
	mOverserved.dispatchedCount.Store(2)
	mOverserved.AddService(10000.0) // high service
	mOverserved.totalInputTokens.Store(200)
	mOverserved.totalOutputTokens.Store(100)

	queues := map[string]QueueInfo{
		"underserved": makeQueueInfo("underserved", 10, mUnderserved, now),
		"overserved":  makeQueueInfo("overserved", 10, mOverserved, now),
	}

	selected, scores := s.Pick(0, queues)
	require.NotNil(t, selected)
	assert.Equal(t, "underserved", selected.FlowKey().ID,
		"underserved program should be preferred in young tier")
	assert.Greater(t, scores["underserved"], scores["overserved"],
		"lower attained service should yield higher score in young tier")
}

func TestEvolvedStrategy_Pick_SRPTInFinishingTier(t *testing.T) {
	s := testEvolved()
	now := time.Now()

	// Both finishing-tier (progress > 0.65), but "light" has less remaining work.
	mLight := &ProgramMetrics{}
	mLight.SetFirstArrival(now.Add(-10 * time.Second))
	mLight.dispatchedCount.Store(9)
	mLight.totalRequests.Store(10)
	mLight.totalInputTokens.Store(900)
	mLight.totalOutputTokens.Store(450)
	// qLen=1, remaining=1, progress=9/(9+1)=0.9

	mHeavy := &ProgramMetrics{}
	mHeavy.SetFirstArrival(now.Add(-10 * time.Second))
	mHeavy.dispatchedCount.Store(7)
	mHeavy.totalRequests.Store(10)
	mHeavy.totalInputTokens.Store(700)
	mHeavy.totalOutputTokens.Store(350)
	// qLen=5, remaining=5, progress=7/(7+5)=0.583 > 0.65? No.
	// Need higher dispatched. Let's make dispatched=20, qLen=5.
	mHeavy.dispatchedCount.Store(20)
	mHeavy.totalInputTokens.Store(2000)
	mHeavy.totalOutputTokens.Store(1000)
	// progress=20/(20+5)=0.8 → finishing. avg_tok=(2000+2000)/20=200. est_work=5*200=1000

	queues := map[string]QueueInfo{
		"light": makeQueueInfo("light", 1, mLight, now),
		"heavy": makeQueueInfo("heavy", 5, mHeavy, now),
	}

	selected, scores := s.Pick(0, queues)
	require.NotNil(t, selected)
	assert.Equal(t, "light", selected.FlowKey().ID,
		"in finishing tier, program with less remaining work (SRPT) should win")
	assert.Greater(t, scores["light"], scores["heavy"])
}

func TestEvolvedStrategy_OnPreRequest_IncrementInFlight(t *testing.T) {
	s := testEvolved()
	m := &ProgramMetrics{}

	req := &scheduling.InferenceRequest{Objectives: scheduling.RequestObjectives{Priority: 0}}
	s.OnPreRequest(m, req)
	assert.Equal(t, int64(1), m.InFlight())

	s.OnPreRequest(m, req)
	assert.Equal(t, int64(2), m.InFlight())
}

func TestEvolvedStrategy_OnPreRequest_SetsFirstArrival(t *testing.T) {
	s := testEvolved()
	m := &ProgramMetrics{}

	assert.True(t, m.FirstArrival().IsZero(), "firstArrival should be zero before any request")

	req := &scheduling.InferenceRequest{Objectives: scheduling.RequestObjectives{Priority: 0}}
	s.OnPreRequest(m, req)
	first := m.FirstArrival()
	assert.False(t, first.IsZero(), "firstArrival should be set after OnPreRequest")

	// Second call should not change it.
	time.Sleep(1 * time.Millisecond)
	s.OnPreRequest(m, req)
	assert.Equal(t, first, m.FirstArrival(), "firstArrival should not change on second call")
}

func TestEvolvedStrategy_OnCompleted_DecrementAndService(t *testing.T) {
	s := testEvolved()
	m := &ProgramMetrics{}
	m.IncrementInFlight()
	m.IncrementInFlight()
	assert.Equal(t, int64(2), m.InFlight())

	resp := &requestcontrol.Response{Usage: requesthandling.Usage{PromptTokens: 100, CompletionTokens: 50}}
	s.OnCompleted(m, nil, resp)

	assert.Equal(t, int64(1), m.InFlight(), "inFlight should decrement on completion")
	assert.InDelta(t, 200.0, m.AttainedService(), 0.01,
		"attained service should increase by weighted cost (100*1 + 50*2 = 200)")
}

func TestEvolvedStrategy_OnCompleted_NilSafe(t *testing.T) {
	s := testEvolved()
	// Should not panic.
	s.OnCompleted(nil, nil, nil)
	s.OnCompleted(&ProgramMetrics{}, nil, nil)
}

func TestEvolved_Pick_IntegrationViaPlugin(t *testing.T) {
	p := &ProgramAwarePlugin{strategy: testEvolved()}
	now := time.Now()

	// Set up two programs: one near completion, one just started.
	mNearDone := p.getOrCreateMetrics("near_done")
	mNearDone.SetFirstArrival(now.Add(-5 * time.Second))
	mNearDone.dispatchedCount.Store(9)
	mNearDone.totalRequests.Store(10)
	mNearDone.totalInputTokens.Store(900)
	mNearDone.totalOutputTokens.Store(450)

	mFresh := p.getOrCreateMetrics("fresh")
	mFresh.SetFirstArrival(now.Add(-100 * time.Millisecond))
	mFresh.dispatchedCount.Store(1)
	mFresh.totalRequests.Store(10)
	mFresh.totalInputTokens.Store(100)
	mFresh.totalOutputTokens.Store(50)

	queueNearDone := &fcmocks.MockFlowQueueAccessor{
		LenV:     1,
		FlowKeyV: flowcontrol.FlowKey{ID: "near_done"},
		PeekHeadV: &fcmocks.MockQueueItemAccessor{
			EnqueueTimeV:     now,
			OriginalRequestV: &fcmocks.MockFlowControlRequest{IDV: "near_done-req"},
		},
	}
	queueFresh := &fcmocks.MockFlowQueueAccessor{
		LenV:     8,
		FlowKeyV: flowcontrol.FlowKey{ID: "fresh"},
		PeekHeadV: &fcmocks.MockQueueItemAccessor{
			EnqueueTimeV:     now,
			OriginalRequestV: &fcmocks.MockFlowControlRequest{IDV: "fresh-req"},
		},
	}

	band := &fcmocks.MockPriorityBandAccessor{
		IterateQueuesFunc: func(cb func(flowcontrol.FlowQueueAccessor) bool) {
			cb(queueNearDone)
			cb(queueFresh)
		},
	}

	queue, err := p.Pick(context.Background(), band)
	require.NoError(t, err)
	assert.Equal(t, "near_done", queue.FlowKey().ID,
		"near-completion program should be prioritized by evolved strategy")
}
