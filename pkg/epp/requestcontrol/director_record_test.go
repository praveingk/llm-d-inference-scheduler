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

func endpoint(name string) fwksched.Endpoint {
	return fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{
			Address:        "192.168.1.100",
			Port:           "8000",
			NamespacedName: types.NamespacedName{Name: name, Namespace: "default"},
		},
		&fwkdl.Metrics{},
		fwkdl.NewAttributes(),
	)
}

// TestPrepareRequest_ParksTargetPods verifies prepareRequest records the routed
// winners (decode primary plus the P/D prefill node) on the scheduling request,
// keyed by namespace/name. The full-candidate pod state is captured upstream in
// HandleRequestBody, not here.
func TestPrepareRequest_ParksTargetPods(t *testing.T) {
	result := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{endpoint("decode1")}},
			"prefill": {TargetEndpoints: []fwksched.Endpoint{endpoint("prefill1")}},
		},
	}

	reqCtx := &handlers.RequestContext{SchedulingRequest: &fwksched.InferenceRequest{RequestID: "r1"}}

	d := &Director{}
	if _, err := d.prepareRequest(t.Context(), reqCtx, result); err != nil {
		t.Fatalf("prepareRequest returned error: %v", err)
	}

	tp, ok := fwksched.ReadRequestAttribute[[]string](reqCtx.SchedulingRequest, requestrecord.TargetPodsAttrKey)
	assert.True(t, ok, "target-pods attribute should be present")
	assert.ElementsMatch(t, []string{"default/decode1", "default/prefill1"}, tp)
}
