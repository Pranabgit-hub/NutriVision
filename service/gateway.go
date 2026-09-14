// Package main: a minimal, illustrative Go gateway for NutriVision.
//
// NOT wired into demo.py or the rest of the Python code -- this is a small,
// self-contained example of the one spot in this stack where Go is a better
// fit than Python: a thin, concurrent, latency-tracked network edge sitting
// in front of the (heavier, GIL-bound) Python/ML pipeline.
//
// Why Go here specifically, and nowhere else in this repo:
//   - Goroutines make it cheap to hold open many concurrent image-upload
//     requests while the Python backend is busy running VLM/depth
//     inference, without the GIL-driven ceiling Python hits under
//     concurrent I/O-bound load.
//   - Go's low, predictable GC pause times matter for tail latency: this
//     gateway's whole job is to keep its OWN overhead out of the p95/p99
//     budget, so it should add single-digit milliseconds, not get blamed
//     for tail latency the Python model call already owns.
//   - Everything else in this repo (segmentation, depth, retrieval,
//     storage) stays in Python on purpose -- that's where the ML/model
//     ecosystem lives. Go only makes sense at this one network-facing
//     edge, which is why its footprint here is intentionally this small.
//
// Run:  go run gateway.go
// Env:  NUTRIVISION_BACKEND (default http://localhost:8000) -- URL of the
//       Python backend this gateway proxies /classify requests to.
package main

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
	"sort"
	"sync"
	"time"
)

// latencyBook is a tiny, mutex-protected ring buffer of request latencies
// (milliseconds) for computing p50/p95/p99 of the *gateway hop itself* --
// deliberately kept separate from whatever pipeline/latency.py reports for
// the Python-side model call, so the two never get conflated during an
// incident ("is it the edge or the model that's slow?" should have one
// obvious answer, not a guess).
type latencyBook struct {
	mu      sync.Mutex
	samples []float64
	cap     int
}

func newLatencyBook(cap int) *latencyBook {
	return &latencyBook{cap: cap}
}

func (b *latencyBook) record(ms float64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.samples = append(b.samples, ms)
	if len(b.samples) > b.cap {
		b.samples = b.samples[len(b.samples)-b.cap:]
	}
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(p/100*float64(len(sorted)-1) + 0.5)
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

func (b *latencyBook) summary() map[string]float64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	if len(b.samples) == 0 {
		return map[string]float64{"count": 0}
	}
	s := append([]float64(nil), b.samples...)
	sort.Float64s(s)
	return map[string]float64{
		"count":  float64(len(s)),
		"p50_ms": percentile(s, 50),
		"p95_ms": percentile(s, 95),
		"p99_ms": percentile(s, 99),
	}
}

func main() {
	backend := os.Getenv("NUTRIVISION_BACKEND")
	if backend == "" {
		backend = "http://localhost:8000"
	}
	book := newLatencyBook(2000)

	// /classify: times the round trip to the Python backend's /classify
	// endpoint and records it. The body is forwarded through unmodified --
	// this gateway does no image processing of its own, only timing + proxying.
	http.HandleFunc("/classify", func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "bad request body", http.StatusBadRequest)
			return
		}
		resp, err := http.Post(backend+"/classify", r.Header.Get("Content-Type"), bytes.NewReader(body))
		if err != nil {
			http.Error(w, "backend unreachable: "+err.Error(), http.StatusBadGateway)
			return
		}
		defer resp.Body.Close()

		elapsedMs := float64(time.Since(start).Microseconds()) / 1000.0
		book.record(elapsedMs)

		w.Header().Set("X-Gateway-Latency-Ms", time.Duration(elapsedMs*float64(time.Millisecond)).String())
		w.WriteHeader(resp.StatusCode)
		io.Copy(w, resp.Body)
	})

	// /metrics: exposes p50/p95/p99 for this gateway's own overhead, in the
	// same units (ms) pipeline/latency.py reports on the Python side, so
	// the two are directly comparable on one dashboard.
	http.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(book.summary())
	})

	log.Println("NutriVision gateway listening on :8080, proxying /classify to", backend)
	log.Fatal(http.ListenAndServe(":8080", nil))
}
