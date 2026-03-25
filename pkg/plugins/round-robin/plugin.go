/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package roundrobin

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"sync"
	"time"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	requestcontrol "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
)

const (
	// RoundRobinPluginType is the registered type name for this plugin.
	RoundRobinPluginType = "round-robin-fairness-with-metrics"

	// fairnessIDHeader is the standard header used to identify the program.
	fairnessIDHeader = "x-gateway-inference-fairness-id"

	// defaultFairnessID is the flow key assigned by the upstream framework when
	// no x-gateway-inference-fairness-id header is present on the request.
	defaultFairnessID = "default-flow"
)

// Compile-time interface assertions.
var (
	_ flowcontrol.FairnessPolicy       = &roundRobin{}
	_ requestcontrol.PrepareDataPlugin = &roundRobin{}
	_ requestcontrol.PreRequest        = &roundRobin{}
	_ requestcontrol.ResponseComplete  = &roundRobin{}
)

// RoundRobinPluginFactory creates a new roundRobin instance.
//
//nolint:revive
func RoundRobinPluginFactory(name string, _ json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	if name == "" {
		name = RoundRobinPluginType
	}
	return &roundRobin{name: name}, nil
}

// roundRobin implements FairnessPolicy with Prometheus instrumentation
// and request lifecycle hooks for per-program metrics.
type roundRobin struct {
	name string

	// programMetrics stores aggregated metrics per program.
	// Key: program ID (string), Value: *ProgramMetrics.
	programMetrics sync.Map

	// requestTimestamps tracks when each request was enqueued,
	// used to compute wait time in PreRequest.
	// Key: request ID (string), Value: time.Time.
	requestTimestamps sync.Map
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *roundRobin) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{
		Type: RoundRobinPluginType,
		Name: p.name,
	}
}

// roundRobinCursor holds the mutable cursor for a specific priority band.
// It is initialized via NewState and stored on the PriorityBandAccessor.
type roundRobinCursor struct {
	mu           sync.Mutex
	lastSelected *flowcontrol.FlowKey
}

// NewState initializes the policy state for a specific priority band.
func (p *roundRobin) NewState(_ context.Context) any {
	return &roundRobinCursor{}
}

// Pick selects the next flow queue in a round-robin fashion from the given priority band.
// It retrieves the band-specific state, locks it, and advances the cursor.
func (p *roundRobin) Pick(
	_ context.Context,
	flowGroup flowcontrol.PriorityBandAccessor,
) (flowcontrol.FlowQueueAccessor, error) {
	start := time.Now()
	defer func() {
		pickLatencyUs.Observe(float64(time.Since(start).Microseconds()))
		fairnessIndex.Set(p.computeFairnessIndex())
	}()

	if flowGroup == nil {
		return nil, nil //nolint:nilnil
	}

	v := flowGroup.PolicyState()
	c, ok := v.(*roundRobinCursor)
	if !ok {
		return nil, fmt.Errorf("invalid state type for RoundRobin policy: expected *roundRobinCursor, got %T", v)
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	keys := flowGroup.FlowKeys()
	if len(keys) == 0 {
		c.lastSelected = nil // Reset cursor if no flows are present.
		return nil, nil
	}

	// Sort for deterministic ordering.
	slices.SortFunc(keys, func(a, b flowcontrol.FlowKey) int { return a.Compare(b) })

	startIndex := 0
	if c.lastSelected != nil {
		// Find the index of the last selected flow.
		// If it's not found (e.g., the flow was removed), we'll start from the beginning (index 0).
		if idx := slices.Index(keys, *c.lastSelected); idx != -1 {
			startIndex = (idx + 1) % len(keys)
		}
	}

	numFlows := len(keys)
	for i := range numFlows {
		currentIdx := (startIndex + i) % numFlows
		currentKey := keys[currentIdx]
		queue := flowGroup.Queue(currentKey.ID)
		if queue != nil && queue.Len() > 0 {
			c.lastSelected = &currentKey

			// Record the selected item's enqueue time so PreRequest can compute
			// the actual flow control queue wait time (enqueue -> dispatch).
			if head := queue.PeekHead(); head != nil {
				p.requestTimestamps.Store(head.OriginalRequest().ID(), head.EnqueueTime())
			}

			return queue, nil
		}
	}

	// No non-empty queue was found.
	c.lastSelected = nil
	return nil, nil //nolint:nilnil
}

// computeFairnessIndex returns Jain's Fairness Index over the average per-request
// throughput (tokens/sec) for each program. Throughput directly measures what each
// program is getting from the system, making it a better fairness signal than wait time.
// Returns 1.0 when fewer than 2 programs have throughput data.
func (p *roundRobin) computeFairnessIndex() float64 {
	var sum, sumSq float64
	var n float64
	p.programMetrics.Range(func(_, value any) bool {
		m := value.(*ProgramMetrics)
		x := m.AverageThroughput()
		if x == 0 {
			return true
		}
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

// getOrCreateMetrics returns the ProgramMetrics for the given program ID, creating it if needed.
func (p *roundRobin) getOrCreateMetrics(programID string) *ProgramMetrics {
	if metricsRaw, ok := p.programMetrics.Load(programID); ok {
		return metricsRaw.(*ProgramMetrics)
	}
	m := &ProgramMetrics{}
	actual, _ := p.programMetrics.LoadOrStore(programID, m)
	return actual.(*ProgramMetrics)
}
