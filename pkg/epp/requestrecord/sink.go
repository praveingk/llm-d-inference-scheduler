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

import "sync"

// DefaultCapacity bounds the in-memory buffer. At a few requests per second a
// collector draining every several seconds never approaches this; it exists
// only so a stalled collector cannot grow memory without bound.
const DefaultCapacity = 100000

// Sink is a bounded, drain-on-read buffer of Records. When full it overwrites
// the oldest record and counts the drop so loss is visible to the collector,
// never silent. Safe for concurrent Add and Drain.
type Sink struct {
	mu       sync.Mutex
	capacity int
	buf      []Record
	dropped  uint64
}

// NewSink returns a Sink holding up to capacity records. A non-positive
// capacity falls back to DefaultCapacity.
func NewSink(capacity int) *Sink {
	if capacity <= 0 {
		capacity = DefaultCapacity
	}
	return &Sink{
		capacity: capacity,
		buf:      make([]Record, 0, capacity),
	}
}

// Add appends a record. If the buffer is full the oldest record is discarded
// and the dropped counter is incremented.
func (s *Sink) Add(rec Record) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.buf) >= s.capacity {
		// Drop the oldest. A single reslice-copy keeps the newest capacity
		// records; drops here are pathological (collector fully stalled), so
		// the O(n) shift is acceptable and keeps the structure a plain slice.
		copy(s.buf, s.buf[1:])
		s.buf = s.buf[:s.capacity-1]
		s.dropped++
	}
	s.buf = append(s.buf, rec)
}

// Drain returns all buffered records in insertion order and the cumulative
// dropped count, then clears the buffer. The dropped count is monotonic across
// the Sink's life so a collector can observe total loss, not just per-drain.
func (s *Sink) Drain() ([]Record, uint64) {
	s.mu.Lock()
	defer s.mu.Unlock()

	out := s.buf
	s.buf = make([]Record, 0, s.capacity)
	return out, s.dropped
}
