# NutriVision

Image of a meal → per-ingredient volume/mass → macros + micronutrients →
daily tracking → grounded nutrition chatbot.

## What's real vs. mocked in this repo

This environment has no network access to model hubs (Hugging Face, etc.),
so the two heavy vision models can't have their weights downloaded here.
Everything else is fully implemented and tested.

| Component | Status |
|---|---|
| Nutrition DB + fuzzy retrieval (`nutrition_db.py`) | **Real.** 30-food seed CSV, rapidfuzz matching. |
| Volume estimation from depth+mask (`volume_estimator.py`) | **Real.** Pure numpy geometry, no model weights needed. |
| Macro calculation + aggregation (`nutrition_db.py`, `storage.py`) | **Real.** Deterministic arithmetic. |
| Daily tracking, SQLite storage (`storage.py`) | **Real.** Incremental rollups, tested. |
| Chatbot tool-calling loop (`chatbot.py`) | **Real**, calls the Claude API — needs `ANTHROPIC_API_KEY`. |
| VLM ingredient classification (`vlm_classifier.py`) | **Real path implemented**, calls the Claude API on image crops — needs `ANTHROPIC_API_KEY`. Mock fallback (`classify_mock`) for offline dev. |
| Monocular depth estimation (`depth_estimation.py`) | **Interface implemented**, real load path attempts `transformers` + Depth Anything V2 — **requires model-hub network access this environment doesn't have.** `DepthEstimator.mock()` provides a synthetic depth map so the rest of the pipeline is testable. |
| Instance segmentation (`segmentation.py`) | **Interface implemented**, real path is SAM2 — same model-hub constraint as above. `Segmenter.mock_segments()` stands in. |

Run `python demo.py` to see the full pipeline execute end-to-end (using the
mocks for the two vision stages) — it detects three mock "items," estimates
their volume via real geometry against a synthetic depth map, converts to
mass, looks up real macros from the nutrition DB, logs the meal to SQLite,
and prints daily progress against targets. Run `pytest tests/ -v` for the
11-test suite covering every deterministic component.

**To make this fully real**: run `depth_estimation.DepthEstimator.load()` and
`segmentation.Segmenter.load()` (currently stubbed to raise a clear error)
somewhere with Hugging Face access, and set `ANTHROPIC_API_KEY` for the VLM
and chatbot calls. No other code changes needed — `pipeline.py` doesn't care
whether depth/segments are real or mocked, only whether `depth.metric is True`.

## Architecture

```
 photo
   │
   ▼
 ┌─────────────────────┐      ┌──────────────────────┐
 │ Segmentation (SAM2)  │ ───▶ │ per-item masks        │
 └─────────────────────┘      └──────────────────────┘
   │                                    │
   ▼                                    ▼
 ┌─────────────────────┐      ┌──────────────────────┐
 │ Depth estimation      │     │ VLM classification    │
 │ (Depth Pro / DA-V2)   │     │ (name each mask)      │
 └─────────────────────┘      └──────────────────────┘
   │                                    │
   ▼                                    ▼
 ┌───────────────────────────────────────────────────┐
 │ Volume estimator: height-above-plate × pixel area  │
 │  integrated per mask → mL → × density → grams       │
 └───────────────────────────────────────────────────┘
                       │
                       ▼
 ┌───────────────────────────────────────────────────┐
 │ Nutrition DB retrieval (fuzzy match label → entry) │
 │ + deterministic macro scaling (grams × per-100g)    │
 └───────────────────────────────────────────────────┘
                       │
              (human confirms/edits portions)
                       │
                       ▼
 ┌───────────────────────────────────────────────────┐
 │ Storage: per-meal log + incremental daily rollup    │
 └───────────────────────────────────────────────────┘
                       │
                       ▼
 ┌───────────────────────────────────────────────────┐
 │ Chatbot: tool-calls get_daily_summary/get_meal_     │
 │ history rather than having history stuffed in-prompt│
 └───────────────────────────────────────────────────┘
```

## Design decisions worth knowing before you extend this

**Volume estimation is the highest-risk stage, by a wide margin.** A single
RGB image has no inherent metric scale — relative depth alone leaves your
volume estimate off by an unknown multiplicative factor. `pipeline.py`
enforces this by refusing to run if `depth.metric` isn't `True`
(`scale_from_reference_object()` is the calibration path: put a known-size
object — a card, a standard plate rim — in frame once). Amorphous foods
(curry, dal, soup) also break clean geometric reconstruction; there's no
handled fallback for that yet, only a documented gap (see
`volume_estimator.py` docstring). If you only fix one thing before a real
deployment, fix this one — recent research (2026) on multi-food volume
benchmarks confirms geometry-based reconstruction with an explicit scale
reference consistently beats VLM-only volume guesses; this repo's structure
follows that finding rather than asking the VLM to eyeball portions.

**The "RAG" step is retrieval + arithmetic, not generation.** Once you have
`(ingredient_label, grams)`, computing macros is `grams/100 × per-100g
values` — deterministic. `nutrition_db.py` does embedding-free fuzzy-name
retrieval (rapidfuzz) to match a free-text VLM label to a canonical DB row;
there's an `EmbeddingRetriever` stub documenting how to upgrade to real
semantic retrieval once you have model-hub access for an embedding model.
No LLM is asked to "generate" a calorie number anywhere in this codebase.

**Human-in-the-loop is load-bearing, not optional polish.** Errors compound
multiplicatively through segmentation → classification → density lookup →
volume → mass, so `pipeline.py` flags any item below a match-confidence
threshold (`needs_confirmation`) for the user to review/edit before logging.
Treat the auto-detected numbers as a first draft, not ground truth, in any
UI you build on top of this.

**The chatbot doesn't get the user's full history stuffed into its prompt.**
It gets tool definitions (`get_daily_summary`, `get_meal_history`) and calls
them when it needs numbers. This keeps context bounded as history grows and
keeps arithmetic in `storage.py` rather than in the model. It's also scoped
away from clinical/medical nutrition advice by its system prompt — see
`chatbot.py` for the exact boundary and why.

## Extending the nutrition DB

`data/nutrition_db.csv` has 30 seed entries. Swap in USDA FoodData Central
or IFCT (Indian Food Composition Tables — likely better coverage for
South-Asian dishes) exports for real coverage; `NutritionDB._load()` expects
the same 14-column schema, so either reshape your source export to match or
extend `FoodEntry` and the loader together.

## Known gaps not addressed in this scaffold

- No handling for mixed/amorphous dishes where geometric reconstruction
  doesn't apply cleanly (curry, soup) — needs a container-volume or
  VLM-portion-class fallback.
- No multi-user auth/API layer — `storage.py` is a plain SQLite wrapper,
  fine for one user or a prototype, swap for Postgres + an API layer for
  real multi-user deployment.
- No image capture/upload UI — see the companion dashboard artifact for a
  frontend mockup of the logging + chatbot experience.

## Evaluation harness (`eval/`)

Four benchmarks, two of which produce real numbers today and two of which
need data this sandbox can't reach:

| Benchmark | Script | Status |
|---|---|---|
| Retrieval accuracy (label → nutrition-DB entry) | `eval/retrieval_eval.py` | **Real numbers**, 90 hand-labeled pairs, no external data needed |
| Confidence-flag precision/recall | `eval/confidence_flag_eval.py` | **Real numbers**, reuses the same 90 labeled pairs |
| End-to-end macro MAPE vs. Nutrition5k | `eval/nutrition5k_eval.py` | Harness only — needs the real dataset (not reachable from this sandbox) + real depth/seg/VLM weights |
| Segmentation mean IoU | `eval/segmentation_eval.py` | Harness only — needs real photos + human-annotated masks |

Run `python -m eval.retrieval_eval` and `python -m eval.confidence_flag_eval
--sweep` to reproduce the numbers below. Run `python -m pytest tests/
test_eval.py -v` for the harness's own unit tests.

**Retrieval accuracy** (90 labeled pairs, spanning easy paraphrases through
hard regional-name/synonym cases):
- top-1: 68.9%, top-3: 84.4%
- By difficulty: easy 89.8% / 100% (top-1/top-3), medium 45.0% / 75.0%, hard
  0% / 18.2%
- **Takeaway**: plain fuzzy string matching (rapidfuzz) is a fine baseline
  for near-exact labels but degrades sharply on regional names and
  paraphrases ("chapati" → roti, "dahi" → curd) — exactly the case the
  `EmbeddingRetriever` stub in `nutrition_db.py` exists to fix. Don't ship
  fuzzy-only retrieval without either that upgrade or a larger, synonym-rich
  seed dataset.

**Confidence-flag precision/recall** at the pipeline's default threshold
(match score < 70 ⇒ flag for user confirmation):
- precision 73.9%, recall 60.7%, F1 66.7% (TP=17, FP=6, TN=56, FN=11)
- Threshold sweep shows recall reaches 100% at threshold=90, at a precision
  cost (51.9%) — i.e. flagging more aggressively catches every wrong match
  but roughly doubles how often a correct match gets needlessly flagged for
  review. **The 11 false negatives are the concerning case** (wrong match,
  not flagged, scored ≥70 anyway) — see `eval/confidence_flag_eval.py`
  output for the specific failing queries.

Both numbers above are computed against this repo's 30-food seed DB and
90-pair label set — they'll shift once the DB is extended (see "Extending
the nutrition DB") or a larger label set is built, and they say nothing
about end-to-end photo accuracy, which needs the Nutrition5k harness plus
real model weights to measure.

