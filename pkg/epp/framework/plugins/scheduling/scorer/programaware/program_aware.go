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

package programaware

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"sync"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

const (
	// ProgramAwareType is the type of the program-aware scorer.
	ProgramAwareType = "program-aware-scorer"

	// defaultMissThreshold is the number of consecutive commits during which a
	// program's pinned pod may be absent from the candidate set before the pin
	// migrates to a new pod. It exists to keep a program from re-pinning on a
	// transient blip (a brief watch/subset gap): a real pod-down persists across
	// many requests and crosses the threshold, a blip does not.
	defaultMissThreshold = 3

	// maxDumpEntries bounds the DumpState payload.
	maxDumpEntries = 100
)

// compile-time type assertions
var (
	_ fwksched.Scorer       = &ProgramAware{}
	_ fwkrc.PreRequest      = &ProgramAware{}
	_ fwkplugin.StateDumper = &ProgramAware{}
)

// Parameters defines the parameters for the program-aware scorer.
type Parameters struct {
	// MissThreshold is the number of consecutive commits a program's pinned pod
	// may be absent before the program migrates. Defaults to defaultMissThreshold
	// when unset (non-positive).
	MissThreshold int `json:"missThreshold"`
}

// Factory defines the factory function for the program-aware scorer.
func Factory(name string, rawParameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	parameters := Parameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' scorer - %w", ProgramAwareType, err)
		}
	}
	return NewProgramAware(name, &parameters), nil
}

// NewProgramAware creates a new program-aware scorer.
func NewProgramAware(name string, params *Parameters) *ProgramAware {
	missThreshold := defaultMissThreshold
	if params != nil && params.MissThreshold > 0 {
		missThreshold = params.MissThreshold
	}
	return &ProgramAware{
		typedName:     fwkplugin.TypedName{Type: ProgramAwareType, Name: name},
		pins:          map[string]string{},
		podCount:      map[string]int{},
		misses:        map[string]int{},
		missThreshold: missThreshold,
	}
}

// ProgramAware pins each program (identified by the request's FairnessID) to a single
// pod so a program's traffic consistently lands on one endpoint. New programs are placed
// on the pod pinned by the fewest programs, keeping pods balanced by program count. A
// program migrates to a new pod only when its pinned pod leaves the candidate set (the
// pod-down signal) for missThreshold consecutive commits.
//
// State is per-EPP-instance and lives only in memory; there is no cross-replica
// coordination (matching the other stateful scorer, no-hit-lru).
type ProgramAware struct {
	typedName fwkplugin.TypedName

	// mu guards pins, podCount, misses, and rrCursor.
	mu sync.RWMutex
	// pins maps a program ID (FairnessID) to its pinned pod key (NamespacedName string).
	pins map[string]string
	// podCount maps a pod key to the number of programs currently pinned to it.
	podCount map[string]int
	// misses maps a program ID to the count of consecutive commits during which its
	// pinned pod was absent from the candidate set.
	misses map[string]int
	// rrCursor rotates the least-loaded tie-break so new programs spread round-robin.
	rrCursor int

	missThreshold int
}

// TypedName returns the type and name tuple of this plugin instance.
func (s *ProgramAware) TypedName() fwkplugin.TypedName {
	return s.typedName
}

// Category returns the preference the scorer applies. A program prefers its pinned
// endpoint, so this is an Affinity scorer.
func (s *ProgramAware) Category() fwksched.ScorerCategory {
	return fwksched.Affinity
}

// programID returns the program identity for a request, defaulting like the director does.
func programID(request *fwksched.InferenceRequest) string {
	if request == nil || request.FairnessID == "" {
		return metadata.DefaultFairnessID
	}
	return request.FairnessID
}

// Score prefers the program's pinned pod when it is present in the candidate set,
// otherwise it prefers the least-loaded present pod. The preferred pod scores 1.0 and
// every other candidate scores 0.0. Score does not mutate pin state; the pin is committed
// in PreRequest against the pod the picker actually selected, since the picker (weighing
// all scorers) may land on a different pod than this scorer prefers.
func (s *ProgramAware) Score(ctx context.Context, request *fwksched.InferenceRequest, pods []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	logger := log.FromContext(ctx).V(logging.DEBUG)

	scores := make(map[fwksched.Endpoint]float64, len(pods))
	if len(pods) == 0 {
		return scores
	}
	for _, pod := range pods {
		scores[pod] = 0.0
	}

	program := programID(request)

	s.mu.RLock()
	pinnedKey := s.pins[program]
	s.mu.RUnlock()

	var target fwksched.Endpoint
	if pinnedKey != "" {
		for _, pod := range pods {
			if podKey(pod) == pinnedKey {
				target = pod
				break
			}
		}
	}
	if target == nil {
		// No pin, or the pinned pod is absent from this request's candidate set.
		// Prefer the least-loaded present pod without changing the pin here.
		target = s.leastLoadedPod(pods)
	}

	scores[target] = 1.0

	logger.Info("program-aware score", "program", program, "target", podKey(target), "candidates", len(pods))
	return scores
}

// leastLoadedPod returns the candidate pod pinned by the fewest programs, breaking ties
// with a rotating cursor so new programs spread round-robin across equally loaded pods.
func (s *ProgramAware) leastLoadedPod(pods []fwksched.Endpoint) fwksched.Endpoint {
	keys := make([]string, len(pods))
	byKey := make(map[string]fwksched.Endpoint, len(pods))
	for i, pod := range pods {
		k := podKey(pod)
		keys[i] = k
		byKey[k] = pod
	}
	sort.Strings(keys) // deterministic order independent of candidate-slice ordering

	s.mu.RLock()
	defer s.mu.RUnlock()

	minCount := -1
	candidates := make([]string, 0, len(keys))
	for _, k := range keys {
		c := s.podCount[k]
		switch {
		case minCount == -1 || c < minCount:
			minCount = c
			candidates = append(candidates[:0], k)
		case c == minCount:
			candidates = append(candidates, k)
		}
	}
	// Round-robin tie-break over the least-loaded set for an even spread.
	return byKey[candidates[s.rrCursor%len(candidates)]]
}

// PreRequest commits the pin based on the pod the picker actually selected. It is the
// only place pin state is mutated.
func (s *ProgramAware) PreRequest(ctx context.Context, request *fwksched.InferenceRequest, schedulingResult *fwksched.SchedulingResult) {
	logger := log.FromContext(ctx).V(logging.TRACE)

	chosen, ok := chosenPod(schedulingResult)
	if !ok {
		return
	}
	program := programID(request)

	s.mu.Lock()
	defer s.mu.Unlock()
	s.rrCursor++ // advance the round-robin tie-break each commit

	existing, pinned := s.pins[program]
	switch {
	case !pinned:
		// First time we see this program: pin it to the chosen pod.
		s.pins[program] = chosen
		s.podCount[chosen]++
		delete(s.misses, program)
		logger.Info("program-aware pin created", "program", program, "pod", chosen)
	case existing == chosen:
		// Pin honored (or the picker re-selected it): reset the miss counter.
		delete(s.misses, program)
	default:
		// The picker used a different pod than the pin, i.e. the pinned pod was
		// absent. Stay sticky until the pod has been absent missThreshold times,
		// then migrate the pin.
		s.misses[program]++
		if s.misses[program] >= s.missThreshold {
			s.podCount[existing]--
			if s.podCount[existing] <= 0 {
				delete(s.podCount, existing)
			}
			s.pins[program] = chosen
			s.podCount[chosen]++
			delete(s.misses, program)
			logger.Info("program-aware pin migrated", "program", program, "from", existing, "to", chosen)
		}
	}
}

// programAwareDump is the sanitized DumpState payload: the current program->pod pins and
// per-pod program counts, capped so the payload stays bounded.
type programAwareDump struct {
	Pins       map[string]string `json:"pins"`
	PodCount   map[string]int    `json:"podCount"`
	TotalPins  int               `json:"totalPins"`
	MaxEntries int               `json:"maxEntries"`
}

// DumpState reports the current pins and per-pod program counts for debugging.
func (s *ProgramAware) DumpState() (json.RawMessage, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	dump := programAwareDump{
		Pins:       make(map[string]string, len(s.pins)),
		PodCount:   make(map[string]int, len(s.podCount)),
		TotalPins:  len(s.pins),
		MaxEntries: maxDumpEntries,
	}
	for program, pod := range s.pins {
		if len(dump.Pins) >= maxDumpEntries {
			break
		}
		dump.Pins[program] = pod
	}
	for pod, count := range s.podCount {
		if len(dump.PodCount) >= maxDumpEntries {
			break
		}
		dump.PodCount[pod] = count
	}
	return json.Marshal(dump)
}

// podKey returns the stable identifier used to pin a program to a pod.
func podKey(pod fwksched.Endpoint) string {
	return pod.GetMetadata().NamespacedName.String()
}

// chosenPod returns the pod the picker selected for the primary profile, if any.
func chosenPod(result *fwksched.SchedulingResult) (string, bool) {
	if result == nil {
		return "", false
	}
	profile, ok := result.ProfileResults[result.PrimaryProfileName]
	if !ok || profile == nil || len(profile.TargetEndpoints) == 0 {
		return "", false
	}
	return podKey(profile.TargetEndpoints[0]), true
}
