/*
Copyright 2026 The Kubernetes Authors.

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

package handlers

import (
	"testing"

	"github.com/stretchr/testify/assert"

	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/requestrecord"
)

// TestBuildRequestRecord_MergesCandidatesPrefixAndTargets verifies the record
// carries every candidate's load, records each prefix producer's per-pod
// prediction under its own producer name, and flags the routed winners.
func TestBuildRequestRecord_MergesCandidatesPrefixAndTargets(t *testing.T) {
	req := &fwksched.InferenceRequest{RequestID: "r1"}

	// Three candidates in the fleet; only pod1 was routed to.
	req.PutAttribute(requestrecord.PodStateAttrKey, map[string]requestrecord.PodDispatchState{
		"default/pod1": {KVCacheUtil: 0.4, QueueSize: 2, RunningRequests: 3},
		"default/pod2": {KVCacheUtil: 0.1, QueueSize: 0, RunningRequests: 1},
		"default/pod3": {KVCacheUtil: 0.9, QueueSize: 7, RunningRequests: 5},
	})
	// Two prefix producers, each parking under its own scoped key. The names are
	// the producers' runtime instance names: the approximate producer is
	// auto-injected as the default producer, so its name is its type
	// "approx-prefix-cache-producer" (not the scorer that consumes its data). A
	// non-winner (pod2) has the largest approximate match; pod3 has none.
	req.PutAttribute(requestrecord.PrefixPerPodAttrKey("approx-prefix-cache-producer"), map[string]requestrecord.PrefixPodMatch{
		"default/pod1": {MatchBlocks: 1, CachedBlockCount: 1},
		"default/pod2": {MatchBlocks: 4, CachedBlockCount: 4},
	})
	req.PutAttribute(requestrecord.PrefixPrimaryAttrKey("approx-prefix-cache-producer"), requestrecord.PrefixMatch{
		HitBlocks: 1, TotalBlocks: 4, BlockSize: 64,
	})
	req.PutAttribute(requestrecord.PrefixPerPodAttrKey("precise-prefix-cache-producer"), map[string]requestrecord.PrefixPodMatch{
		"default/pod1": {MatchBlocks: 2, CachedBlockCount: 3},
	})
	req.PutAttribute(requestrecord.PrefixPrimaryAttrKey("precise-prefix-cache-producer"), requestrecord.PrefixMatch{
		HitBlocks: 2, TotalBlocks: 4, BlockSize: 64,
	})
	req.PutAttribute(requestrecord.TargetPodsAttrKey, []string{"default/pod1"})

	reqCtx := &RequestContext{SchedulingRequest: req}

	rec := buildRequestRecord(reqCtx, "flow-a")

	assert.Equal(t, "r1", rec.RequestID)
	assert.Equal(t, "flow-a", rec.FairnessID)

	// All three candidates present with load intact.
	assert.Len(t, rec.PodStateAtDispatch, 3)
	assert.Equal(t, 0.9, rec.PodStateAtDispatch["default/pod3"].KVCacheUtil)

	// Both producers recorded, keyed by producer name, without collision.
	assert.Len(t, rec.Prefix, 2)
	approx := rec.Prefix["approx-prefix-cache-producer"]
	assert.Equal(t, 1, approx.PerPod["default/pod1"].MatchBlocks)
	assert.Equal(t, 4, approx.PerPod["default/pod2"].MatchBlocks)
	// A candidate with no predicted match is absent (zero value on lookup).
	assert.Equal(t, 0, approx.PerPod["default/pod3"].MatchBlocks)
	assert.Equal(t, 0.25, approx.Primary.HitRatio)

	precise := rec.Prefix["precise-prefix-cache-producer"]
	assert.Equal(t, 2, precise.PerPod["default/pod1"].MatchBlocks)
	assert.Equal(t, 3, precise.PerPod["default/pod1"].CachedBlockCount)

	// Only the routed winner is a target.
	assert.Equal(t, []string{"default/pod1"}, rec.TargetPods)
}
