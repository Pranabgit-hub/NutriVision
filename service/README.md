# `service/` -- Go gateway (illustrative, optional)

This directory holds one file, `gateway.go`: a minimal Go HTTP server that
would sit in front of the Python pipeline in a real deployment. It is **not**
wired into `demo.py`, `pipeline/`, or anything else in this repo -- the rest
of the project is deliberately all-Python, and stays that way. This is the
one spot where Go earns its place instead.

## Why Go, only here

Everything that touches models or data (segmentation, depth, VLM
classification, nutrition retrieval, storage) stays in Python -- that's
where the ML ecosystem lives, and there's no reason to fight that. But the
network-facing edge in front of it has different constraints: it needs to
hold open many concurrent image-upload requests cheaply while the Python
backend is busy doing inference, and its own overhead needs to be
negligible and predictable so it never shows up as the cause of a latency
regression. Go's goroutines and low, predictable GC pauses fit that job
well; Python's GIL makes the same job harder under concurrent I/O load. So:
Go at the edge, Python for everything ML -- a small, common, low-risk
polyglot split, not a rewrite of the pipeline.

## What it does

- `POST /classify` -- reads the request body, forwards it to
  `NUTRIVISION_BACKEND` (default `http://localhost:8000`) unmodified, times
  the round trip, and returns the backend's response with an
  `X-Gateway-Latency-Ms` header attached.
- `GET /metrics` -- returns `{count, p50_ms, p95_ms, p99_ms}` for the
  gateway's own overhead, computed over the last 2,000 requests.

The gateway's own p50/p95/p99 is tracked **separately** from the
model-call latency `pipeline/latency.py` records on the Python side (see
that file's docstring). Keeping them separate means "was it the edge or the
model that was slow?" has one obvious answer from two numbers, not a guess
from one blended average.

## Running it

```bash
cd service
go run gateway.go
# in another terminal, with the Python backend running on :8000:
curl -X POST --data-binary @some_meal_photo.png http://localhost:8080/classify
curl http://localhost:8080/metrics
```

There is no `/classify` HTTP endpoint on the Python side in this repo yet --
`vlm_classifier.classify_segment()` is called in-process from `pipeline.py`,
not over HTTP. Wiring an actual FastAPI/Flask `/classify` route behind this
gateway is the natural next step for a real deployment; this file is scoped
to demonstrate *where* Go fits and *why*, not to ship a full server.
