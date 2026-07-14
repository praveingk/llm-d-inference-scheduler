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

package requestrecord

import (
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
)

const (
	// DebugPath is the drain endpoint served on the metrics/admin port.
	DebugPath = "/debug/request-records"
	// DroppedHeader reports the Sink's cumulative dropped count so a collector
	// can detect buffer overflow.
	DroppedHeader = "X-Records-Dropped"
)

// MetricsHandlerRegistrar registers HTTP handlers on the process metrics/admin
// server. It matches the interface used by the plugin-state debug handler.
type MetricsHandlerRegistrar interface {
	AddMetricsServerExtraHandler(path string, handler http.Handler) error
}

// SetupHandler registers the drain handler for sink at DebugPath.
func SetupHandler(registrar MetricsHandlerRegistrar, sink *Sink) error {
	if registrar == nil {
		return errors.New("metrics handler registrar is not configured")
	}
	if sink == nil {
		return errors.New("request record sink is not configured")
	}
	return registrar.AddMetricsServerExtraHandler(DebugPath, NewHandler(sink))
}

// NewHandler returns an http.Handler that, on GET, drains sink and writes the
// records as JSONL (one JSON object per line). The response is empty when the
// buffer holds no records. The cumulative dropped count is reported in the
// DroppedHeader.
func NewHandler(sink *Sink) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.Header().Set("Allow", http.MethodGet)
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if sink == nil {
			http.Error(w, "request record sink is not configured", http.StatusInternalServerError)
			return
		}

		records, dropped := sink.Drain()

		w.Header().Set("Content-Type", "application/x-ndjson")
		w.Header().Set(DroppedHeader, strconv.FormatUint(dropped, 10))
		w.WriteHeader(http.StatusOK)

		enc := json.NewEncoder(w)
		for i := range records {
			// Encoder writes a trailing newline per Encode, yielding JSONL.
			if err := enc.Encode(&records[i]); err != nil {
				// The header is already sent; nothing further to do but stop.
				return
			}
		}
	})
}
