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
// carries every candidate's load, overlays each candidate's predicted prefix
// match by namespace/name, and flags the routed winners separately.
func TestBuildRequestRecord_MergesCandidatesPrefixAndTargets(t *testing.T) {
	req := &fwksched.InferenceRequest{RequestID: "r1"}

	// Three candidates in the fleet; only pod1 was routed to.
	req.PutAttribute(requestrecord.PodStateAttrKey, map[string]requestrecord.PodDispatchState{
		"default/pod1": {KVCacheUtil: 0.4, QueueSize: 2, RunningRequests: 3},
		"default/pod2": {KVCacheUtil: 0.1, QueueSize: 0, RunningRequests: 1},
		"default/pod3": {KVCacheUtil: 0.9, QueueSize: 7, RunningRequests: 5},
	})
	// A non-winner (pod2) has the largest predicted prefix match; pod3 has none.
	req.PutAttribute(requestrecord.PrefixPerPodAttrKey, map[string]int{
		"default/pod1": 1,
		"default/pod2": 4,
	})
	req.PutAttribute(requestrecord.TargetPodsAttrKey, []string{"default/pod1"})

	reqCtx := &RequestContext{SchedulingRequest: req}

	rec := buildRequestRecord(reqCtx, "flow-a")

	assert.Equal(t, "r1", rec.RequestID)
	assert.Equal(t, "flow-a", rec.FairnessID)

	// All three candidates present with load intact.
	assert.Len(t, rec.PodStateAtDispatch, 3)
	assert.Equal(t, 0.9, rec.PodStateAtDispatch["default/pod3"].KVCacheUtil)

	// Prefix blocks overlaid per candidate, including the non-winner pod2.
	assert.Equal(t, 1, rec.PodStateAtDispatch["default/pod1"].PrefixMatchBlocks)
	assert.Equal(t, 4, rec.PodStateAtDispatch["default/pod2"].PrefixMatchBlocks)
	// A candidate with no predicted match stays at zero.
	assert.Equal(t, 0, rec.PodStateAtDispatch["default/pod3"].PrefixMatchBlocks)

	// Only the routed winner is a target.
	assert.Equal(t, []string{"default/pod1"}, rec.TargetPods)
}
