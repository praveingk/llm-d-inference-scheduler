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

// Package requestrecord implements a debug-only per-request event log. Each
// completed request contributes one Record capturing the exact token counts,
// the pods it was routed to, the router prefix-match prediction, and each
// target pod's load snapshot at dispatch time. Records accumulate in a bounded
// in-memory Sink drained over HTTP by an external collector; nothing is written
// to disk by the EPP. The feature is gated off by default and carries no
// production guarantees.
package requestrecord

// PrefixMatchAttrKey and PodStateAttrKey name the values parked on an
// InferenceRequest during scheduling for the response handler to read at
// end-of-stream. They are exported so the producing plugins and the consuming
// handler agree on a single key.
const (
	PrefixMatchAttrKey = "requestrecord/prefix-match"
	PodStateAttrKey    = "requestrecord/pod-state-at-dispatch"
)

// PrefixMatch is the router-side prefix-cache prediction for a request's
// primary target, captured when the approximate-prefix plugin scores the
// selection. Lengths are in blocks; multiply by BlockSize for tokens.
type PrefixMatch struct {
	HitBlocks   int `json:"hit_blocks"`
	TotalBlocks int `json:"total_blocks"`
	BlockSize   int `json:"block_size"`
}

// PodDispatchState is a pod's load at the moment it was chosen for a request.
type PodDispatchState struct {
	KVCacheUtil     float64 `json:"kv_util"`
	QueueSize       int     `json:"queue_size"`
	RunningRequests int     `json:"running_requests"`
}

// Record is one completed request's raw observation. All values are exact as
// read from their authoritative source; nothing is aggregated.
type Record struct {
	RequestID      string `json:"request_id"`
	FairnessID     string `json:"fairness_id"`
	Priority       int    `json:"priority"`
	IncomingModel  string `json:"incoming_model"`
	TargetModel    string `json:"target_model"`
	TargetEndpoint string `json:"target_endpoint"`

	// TargetPods lists the pods (namespace/name) the request was routed to:
	// the decode primary plus the prefill node in P/D mode.
	TargetPods []string `json:"target_pods"`

	TSReceivedMs   int64 `json:"ts_received_ms"`
	TSFirstTokenMs int64 `json:"ts_first_token_ms"`
	TSCompleteMs   int64 `json:"ts_complete_ms"`

	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	CachedTokens int `json:"cached_tokens"`

	PrefixHitBlocks   int     `json:"prefix_hit_blocks"`
	PrefixTotalBlocks int     `json:"prefix_total_blocks"`
	PrefixBlockSize   int     `json:"prefix_block_size"`
	PrefixHitRatio    float64 `json:"prefix_hit_ratio"`

	// PodStateAtDispatch maps target pod (namespace/name) to its load snapshot
	// when it was picked.
	PodStateAtDispatch map[string]PodDispatchState `json:"pod_state_at_dispatch,omitempty"`
}
