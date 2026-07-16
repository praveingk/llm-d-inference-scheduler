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

// PodStateAttrKey and TargetPodsAttrKey name the values parked on an
// InferenceRequest during scheduling for the response handler to read at
// end-of-stream. They are exported so the producing code and the consuming
// handler agree on a single key.
const (
	PodStateAttrKey   = "requestrecord/pod-state-at-dispatch"
	TargetPodsAttrKey = "requestrecord/target-pods"

	// prefixAttrPrefix scopes prefix-match parking by producer name so multiple
	// prefix producers (e.g. approximate and precise) each write their own
	// prediction without overwriting one another.
	prefixPrimaryAttrPrefix = "requestrecord/prefix-primary/"
	prefixPerPodAttrPrefix  = "requestrecord/prefix-per-pod/"
)

// PrefixPrimaryAttrKey returns the per-request attribute key under which a
// prefix producer parks its primary-target PrefixMatch, scoped by producer name.
func PrefixPrimaryAttrKey(producerName string) string {
	return prefixPrimaryAttrPrefix + producerName
}

// PrefixPerPodAttrKey returns the per-request attribute key under which a prefix
// producer parks its per-candidate PrefixPodMatch map, scoped by producer name.
func PrefixPerPodAttrKey(producerName string) string {
	return prefixPerPodAttrPrefix + producerName
}

// PrefixPrimaryAttrPrefix is the shared prefix of every producer-scoped
// primary-match key, so the handler can enumerate parked producers.
const PrefixPrimaryAttrPrefix = prefixPrimaryAttrPrefix

// PrefixMatch is a prefix producer's prediction for a request's primary target.
// Lengths are in blocks; multiply by BlockSize for tokens.
type PrefixMatch struct {
	HitBlocks   int     `json:"hit_blocks"`
	TotalBlocks int     `json:"total_blocks"`
	BlockSize   int     `json:"block_size"`
	HitRatio    float64 `json:"hit_ratio"`
}

// PrefixPodMatch is one candidate pod's predicted prefix match from a single
// producer. MatchBlocks is the value the scorer ranked on (tier-weighted for
// the precise producer); CachedBlockCount is the literal contiguous cached
// block count (equal to MatchBlocks for the approximate producer).
type PrefixPodMatch struct {
	MatchBlocks      int `json:"match_blocks"`
	CachedBlockCount int `json:"cached_block_count"`
}

// PrefixSource is one prefix producer's full prediction for a request: the
// primary-target match plus the per-candidate match map, keyed by producer name
// in Record.Prefix.
type PrefixSource struct {
	Primary PrefixMatch               `json:"primary"`
	PerPod  map[string]PrefixPodMatch `json:"per_pod,omitempty"`
}

// PodDispatchState is a candidate pod's load at the moment the request was
// scheduled. Recorded for every candidate the scheduler considered, not only
// the winner, so the scorer's ranking can be reconstructed offline. Per-pod
// prefix predictions live under Record.Prefix, attributed per producer.
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
	// the decode primary plus the prefill node in P/D mode. These are the
	// winners among the candidates in PodStateAtDispatch.
	TargetPods []string `json:"target_pods"`

	TSReceivedMs   int64 `json:"ts_received_ms"`
	TSDispatchedMs int64 `json:"ts_dispatched_ms"`
	TSFirstTokenMs int64 `json:"ts_first_token_ms"`
	TSCompleteMs   int64 `json:"ts_complete_ms"`

	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	CachedTokens int `json:"cached_tokens"`

	// Prefix maps each prefix producer's instance name to its prediction for
	// this request. Multiple producers (e.g. approximate and precise) each
	// contribute one entry; empty when no prefix producer ran.
	Prefix map[string]PrefixSource `json:"prefix,omitempty"`

	// PodStateAtDispatch maps every candidate pod (namespace/name) the scheduler
	// considered to its load snapshot at dispatch.
	PodStateAtDispatch map[string]PodDispatchState `json:"pod_state_at_dispatch,omitempty"`
}
