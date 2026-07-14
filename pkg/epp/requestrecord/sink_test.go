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

import "testing"

func TestSinkDrainReturnsAndClears(t *testing.T) {
	s := NewSink(10)
	for i := 0; i < 3; i++ {
		s.Add(Record{RequestID: string(rune('a' + i))})
	}

	got, dropped := s.Drain()
	if len(got) != 3 {
		t.Fatalf("drain returned %d records, want 3", len(got))
	}
	if dropped != 0 {
		t.Fatalf("dropped = %d, want 0", dropped)
	}
	if got[0].RequestID != "a" || got[2].RequestID != "c" {
		t.Fatalf("records out of insertion order: %+v", got)
	}

	second, _ := s.Drain()
	if len(second) != 0 {
		t.Fatalf("second drain returned %d records, want 0", len(second))
	}
}

func TestSinkOverflowDropsOldestAndCounts(t *testing.T) {
	const cap = 5
	const total = 8
	s := NewSink(cap)
	for i := 0; i < total; i++ {
		s.Add(Record{RequestID: string(rune('a' + i))})
	}

	got, dropped := s.Drain()
	if len(got) != cap {
		t.Fatalf("drain returned %d records, want %d", len(got), cap)
	}
	if dropped != total-cap {
		t.Fatalf("dropped = %d, want %d", dropped, total-cap)
	}
	// The newest cap records survive: 'd'..'h'.
	if got[0].RequestID != "d" || got[cap-1].RequestID != "h" {
		t.Fatalf("unexpected surviving records: %+v", got)
	}

	// Dropped count is cumulative across drains.
	s.Add(Record{RequestID: "x"})
	_, dropped2 := s.Drain()
	if dropped2 != total-cap {
		t.Fatalf("dropped after non-overflow add = %d, want %d", dropped2, total-cap)
	}
}

func TestNewSinkDefaultsCapacity(t *testing.T) {
	s := NewSink(0)
	if s.capacity != DefaultCapacity {
		t.Fatalf("capacity = %d, want %d", s.capacity, DefaultCapacity)
	}
}
