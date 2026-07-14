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

package requestcontrol

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/handlers"
	"github.com/llm-d/llm-d-router/pkg/epp/requestrecord"
)

// TestPrepareRequest_ParksPodDispatchState verifies prepareRequest snapshots
// each target pod's load, keyed by namespace/name, onto the scheduling request.
func TestPrepareRequest_ParksPodDispatchState(t *testing.T) {
	pod1 := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{
			Address:        "192.168.1.100",
			Port:           "8000",
			NamespacedName: types.NamespacedName{Name: "pod1", Namespace: "default"},
		},
		&fwkdl.Metrics{KVCacheUsagePercent: 0.42, WaitingQueueSize: 5, RunningRequestsSize: 3},
		fwkdl.NewAttributes(),
	)
	pod2 := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{
			Address:        "192.168.2.100",
			Port:           "8000",
			NamespacedName: types.NamespacedName{Name: "pod2", Namespace: "default"},
		},
		&fwkdl.Metrics{KVCacheUsagePercent: 0.10, WaitingQueueSize: 0, RunningRequestsSize: 1},
		fwkdl.NewAttributes(),
	)

	result := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{pod1, pod2}},
		},
	}

	reqCtx := &handlers.RequestContext{SchedulingRequest: &fwksched.InferenceRequest{RequestID: "r1"}}

	d := &Director{}
	if _, err := d.prepareRequest(t.Context(), reqCtx, result); err != nil {
		t.Fatalf("prepareRequest returned error: %v", err)
	}

	ps, ok := fwksched.ReadRequestAttribute[map[string]requestrecord.PodDispatchState](reqCtx.SchedulingRequest, requestrecord.PodStateAttrKey)
	assert.True(t, ok, "pod-state attribute should be present")
	assert.Len(t, ps, 2)

	assert.Equal(t, requestrecord.PodDispatchState{KVCacheUtil: 0.42, QueueSize: 5, RunningRequests: 3}, ps["default/pod1"])
	assert.Equal(t, requestrecord.PodDispatchState{KVCacheUtil: 0.10, QueueSize: 0, RunningRequests: 1}, ps["default/pod2"])
}

// TestPrepareRequest_CapturesPrefillPod verifies the prefill node from the
// separate P/D "prefill" profile is captured alongside the decode primary.
func TestPrepareRequest_CapturesPrefillPod(t *testing.T) {
	decode := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{
			Address:        "192.168.1.100",
			Port:           "8000",
			NamespacedName: types.NamespacedName{Name: "decode1", Namespace: "default"},
		},
		&fwkdl.Metrics{KVCacheUsagePercent: 0.5, WaitingQueueSize: 2, RunningRequestsSize: 4},
		fwkdl.NewAttributes(),
	)
	prefill := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{
			Address:        "192.168.9.100",
			Port:           "8000",
			NamespacedName: types.NamespacedName{Name: "prefill1", Namespace: "default"},
		},
		&fwkdl.Metrics{KVCacheUsagePercent: 0.9, WaitingQueueSize: 11, RunningRequestsSize: 7},
		fwkdl.NewAttributes(),
	)

	result := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{decode}},
			"prefill": {TargetEndpoints: []fwksched.Endpoint{prefill}},
		},
	}

	reqCtx := &handlers.RequestContext{SchedulingRequest: &fwksched.InferenceRequest{RequestID: "r2"}}

	d := &Director{}
	if _, err := d.prepareRequest(t.Context(), reqCtx, result); err != nil {
		t.Fatalf("prepareRequest returned error: %v", err)
	}

	ps, ok := fwksched.ReadRequestAttribute[map[string]requestrecord.PodDispatchState](reqCtx.SchedulingRequest, requestrecord.PodStateAttrKey)
	assert.True(t, ok, "pod-state attribute should be present")
	assert.Len(t, ps, 2, "both decode and prefill pods should be captured")
	assert.Equal(t, requestrecord.PodDispatchState{KVCacheUtil: 0.5, QueueSize: 2, RunningRequests: 4}, ps["default/decode1"])
	assert.Equal(t, requestrecord.PodDispatchState{KVCacheUtil: 0.9, QueueSize: 11, RunningRequests: 7}, ps["default/prefill1"])
}
