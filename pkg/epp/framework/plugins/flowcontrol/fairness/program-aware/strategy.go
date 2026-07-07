package programaware

import (
	"fmt"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// Strategy is the fairness scheduling policy. All methods must be safe for
// concurrent use.
type Strategy interface {
	Name() string
	Pick(bandPriority int, queues map[string]QueueInfo) flowcontrol.FlowQueueAccessor
	OnPreRequest(metrics *ProgramMetrics, request *fwksched.InferenceRequest)
	OnCompleted(metrics *ProgramMetrics, request *fwksched.InferenceRequest, response *fwkrc.Response)
	EvictProgram(id string)
	Collectors() []prometheus.Collector
}

type QueueInfo struct {
	Queue   flowcontrol.FlowQueueAccessor
	Metrics *ProgramMetrics
	Len     int
}

func newStrategy(cfg Config) (Strategy, error) {
	switch cfg.Strategy {
	case "", "las":
		return &LASStrategy{
			weightService:   cfg.LASWeightService,
			weightHeadWait:  cfg.LASWeightHeadWait,
			decayFactor:     cfg.LASDecayFactor,
			halfLifeSeconds: cfg.LASHalfLifeSeconds,
		}, nil
	case "composite-urgency":
		// Evolved AdaEvolve run llm_d_las_0622_2242 (Solution 10).
		return &CompositeUrgencyStrategy{
			weightService:   cfg.LASWeightService,
			weightHeadWait:  cfg.LASWeightHeadWait,
			decayFactor:     cfg.LASDecayFactor,
			halfLifeSeconds: cfg.LASHalfLifeSeconds,
		}, nil
	case "exp-backoff":
		// Evolved AdaEvolve run llm_d_las_0622_1510 (Solution 4).
		return &ExpBackoffStrategy{
			weightService:   cfg.LASWeightService,
			weightHeadWait:  cfg.LASWeightHeadWait,
			decayFactor:     cfg.LASDecayFactor,
			halfLifeSeconds: cfg.LASHalfLifeSeconds,
		}, nil
	case "tiered-kv":
		// Evolved AdaEvolve run llm_d_las_0622_0904 (Solution 3).
		return &TieredKVStrategy{
			weightService:   cfg.LASWeightService,
			weightHeadWait:  cfg.LASWeightHeadWait,
			decayFactor:     cfg.LASDecayFactor,
			halfLifeSeconds: cfg.LASHalfLifeSeconds,
		}, nil
	case "isc":
		// Evolved AdaEvolve run llm_d_las_0622_1603 (Solution 5).
		return &ISCStrategy{
			weightService:   cfg.LASWeightService,
			weightHeadWait:  cfg.LASWeightHeadWait,
			decayFactor:     cfg.LASDecayFactor,
			halfLifeSeconds: cfg.LASHalfLifeSeconds,
		}, nil
	default:
		return nil, fmt.Errorf("unknown scoring strategy %q: supported strategies are \"las\", \"composite-urgency\", \"exp-backoff\", \"tiered-kv\", \"isc\"", cfg.Strategy)
	}
}

func rangeNormalize(v, min, max float64) float64 {
	if max == min {
		return 0.5
	}
	return (v - min) / (max - min)
}
