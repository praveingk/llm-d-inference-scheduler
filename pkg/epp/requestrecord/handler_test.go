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
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestHandlerDrainsAsJSONL(t *testing.T) {
	s := NewSink(10)
	s.Add(Record{RequestID: "r1", FairnessID: "prog-a", CachedTokens: 12})
	s.Add(Record{RequestID: "r2", FairnessID: "prog-b", CachedTokens: 34})

	h := NewHandler(s)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, DebugPath, nil))

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rr.Code)
	}
	if rr.Header().Get(DroppedHeader) != "0" {
		t.Fatalf("dropped header = %q, want 0", rr.Header().Get(DroppedHeader))
	}

	lines := strings.Split(strings.TrimRight(rr.Body.String(), "\n"), "\n")
	if len(lines) != 2 {
		t.Fatalf("got %d JSONL lines, want 2: %q", len(lines), rr.Body.String())
	}
	var rec Record
	if err := json.Unmarshal([]byte(lines[0]), &rec); err != nil {
		t.Fatalf("line 0 is not valid JSON: %v", err)
	}
	if rec.RequestID != "r1" || rec.CachedTokens != 12 {
		t.Fatalf("unexpected first record: %+v", rec)
	}

	// The GET drained the buffer: a second GET yields nothing.
	rr2 := httptest.NewRecorder()
	h.ServeHTTP(rr2, httptest.NewRequest(http.MethodGet, DebugPath, nil))
	if body := strings.TrimSpace(rr2.Body.String()); body != "" {
		t.Fatalf("second drain returned %q, want empty", body)
	}
}

func TestHandlerRejectsNonGET(t *testing.T) {
	h := NewHandler(NewSink(10))
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, httptest.NewRequest(http.MethodPost, DebugPath, nil))
	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", rr.Code)
	}
}

func TestHandlerReportsDropped(t *testing.T) {
	s := NewSink(2)
	for _, id := range []string{"a", "b", "c", "d"} {
		s.Add(Record{RequestID: id})
	}
	h := NewHandler(s)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, DebugPath, nil))
	if rr.Header().Get(DroppedHeader) != "2" {
		t.Fatalf("dropped header = %q, want 2", rr.Header().Get(DroppedHeader))
	}
}
