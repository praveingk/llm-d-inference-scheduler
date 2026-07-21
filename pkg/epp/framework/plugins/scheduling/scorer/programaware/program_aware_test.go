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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

// newEndpoint builds a candidate endpoint identified by name.
func newEndpoint(name string) fwksched.Endpoint {
	return fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: types.NamespacedName{Name: name}},
		&fwkdl.Metrics{},
		nil,
	)
}

// topScored returns the single endpoint scored 1.0, requiring every other endpoint to be 0.0.
func topScored(t *testing.T, scores map[fwksched.Endpoint]float64) fwksched.Endpoint {
	t.Helper()
	var top fwksched.Endpoint
	for ep, score := range scores {
		if score == 1.0 {
			require.Nil(t, top, "expected exactly one endpoint scored 1.0")
			top = ep
		} else {
			assert.Equal(t, 0.0, score, "non-preferred endpoint must score 0.0")
		}
	}
	require.NotNil(t, top, "expected one endpoint scored 1.0")
	return top
}

// result wraps a chosen endpoint in a single-primary-profile SchedulingResult.
func result(chosen fwksched.Endpoint) *fwksched.SchedulingResult {
	return &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{chosen}},
		},
	}
}

// dispatch runs one Score+PreRequest cycle for a program against pods, committing the
// pod that Score preferred (the common case where this scorer wins the pick).
func dispatch(t *testing.T, s *ProgramAware, program string, pods []fwksched.Endpoint) fwksched.Endpoint {
	t.Helper()
	req := &fwksched.InferenceRequest{RequestID: program + "-req", FairnessID: program}
	chosen := topScored(t, s.Score(context.Background(), req, pods))
	s.PreRequest(context.Background(), req, result(chosen))
	return chosen
}

func newScorer(t *testing.T) *ProgramAware {
	t.Helper()
	return NewProgramAware(ProgramAwareType, &Parameters{MissThreshold: defaultMissThreshold})
}

func TestNewProgramGetsPinnedAndBalanced(t *testing.T) {
	s := newScorer(t)
	pods := []fwksched.Endpoint{newEndpoint("pod-a"), newEndpoint("pod-b"), newEndpoint("pod-c")}

	// Three distinct programs should land on three distinct pods (least-loaded spread).
	seen := map[string]bool{}
	for _, program := range []string{"p1", "p2", "p3"} {
		chosen := dispatch(t, s, program, pods)
		seen[podKey(chosen)] = true
	}
	assert.Len(t, seen, 3, "three programs should spread across all three pods")

	s.mu.RLock()
	defer s.mu.RUnlock()
	assert.Len(t, s.pins, 3)
	for _, count := range s.podCount {
		assert.Equal(t, 1, count, "each pod should carry exactly one program")
	}
}

func TestPinIsStableAcrossRequests(t *testing.T) {
	s := newScorer(t)
	pods := []fwksched.Endpoint{newEndpoint("pod-a"), newEndpoint("pod-b"), newEndpoint("pod-c")}

	first := dispatch(t, s, "p1", pods)
	for i := 0; i < 5; i++ {
		again := dispatch(t, s, "p1", pods)
		assert.Equal(t, podKey(first), podKey(again), "program must stay on its pinned pod")
	}
}

func TestStickyDoesNotMigrateOnTransientAbsence(t *testing.T) {
	s := newScorer(t)
	pods := []fwksched.Endpoint{newEndpoint("pod-a"), newEndpoint("pod-b")}

	pinned := dispatch(t, s, "p1", pods)

	// The pinned pod is briefly absent for fewer than missThreshold commits.
	remaining := remove(pods, pinned)
	for i := 0; i < defaultMissThreshold-1; i++ {
		fallback := dispatch(t, s, "p1", remaining)
		assert.NotEqual(t, podKey(pinned), podKey(fallback), "fallback must avoid the absent pod")
	}

	// The pinned pod returns; the program must be back on its original pin.
	restored := dispatch(t, s, "p1", pods)
	assert.Equal(t, podKey(pinned), podKey(restored), "transient absence must not migrate the pin")
}

func TestMigratesAfterThresholdMisses(t *testing.T) {
	s := newScorer(t)
	pods := []fwksched.Endpoint{newEndpoint("pod-a"), newEndpoint("pod-b")}

	pinned := dispatch(t, s, "p1", pods)
	remaining := remove(pods, pinned)

	// The pinned pod is absent for missThreshold consecutive commits: migrate.
	var last fwksched.Endpoint
	for i := 0; i < defaultMissThreshold; i++ {
		last = dispatch(t, s, "p1", remaining)
	}

	s.mu.RLock()
	newPin := s.pins["p1"]
	s.mu.RUnlock()
	assert.Equal(t, podKey(last), newPin, "pin should have migrated to the fallback pod")
	assert.NotEqual(t, podKey(pinned), newPin)

	// Even when the old pod returns, the program stays on the migrated pin.
	after := dispatch(t, s, "p1", pods)
	assert.Equal(t, newPin, podKey(after))
}

func TestEmptyFairnessIDUsesDefault(t *testing.T) {
	s := newScorer(t)
	pods := []fwksched.Endpoint{newEndpoint("pod-a"), newEndpoint("pod-b")}

	req := &fwksched.InferenceRequest{RequestID: "r1"} // no FairnessID
	chosen := topScored(t, s.Score(context.Background(), req, pods))
	s.PreRequest(context.Background(), req, result(chosen))

	s.mu.RLock()
	defer s.mu.RUnlock()
	_, ok := s.pins[metadata.DefaultFairnessID]
	assert.True(t, ok, "empty FairnessID must pin under the default flow")
}

func TestSingleCandidate(t *testing.T) {
	s := newScorer(t)
	only := newEndpoint("pod-a")
	scores := s.Score(context.Background(), &fwksched.InferenceRequest{RequestID: "r1", FairnessID: "p1"},
		[]fwksched.Endpoint{only})
	assert.Equal(t, 1.0, scores[only])
}

func TestScoreDoesNotMutatePins(t *testing.T) {
	s := newScorer(t)
	pods := []fwksched.Endpoint{newEndpoint("pod-a"), newEndpoint("pod-b")}

	// Score alone (no PreRequest) must not create a pin.
	s.Score(context.Background(), &fwksched.InferenceRequest{RequestID: "r1", FairnessID: "p1"}, pods)
	s.mu.RLock()
	defer s.mu.RUnlock()
	assert.Empty(t, s.pins, "Score must not commit a pin")
}

// remove returns pods without the given endpoint (by pod key).
func remove(pods []fwksched.Endpoint, drop fwksched.Endpoint) []fwksched.Endpoint {
	out := make([]fwksched.Endpoint, 0, len(pods))
	for _, p := range pods {
		if podKey(p) != podKey(drop) {
			out = append(out, p)
		}
	}
	return out
}
