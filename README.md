# NutriVision

Image of a meal → per-ingredient volume/mass → macros + micronutrients → daily tracking → grounded nutrition chatbot.

## Core Features & Pipeline Components

| Component | Architecture & Design |
|---|---|
| Nutrition DB + Retrieval (`nutrition_db.py`) | Rapidfuzz-based matching against a 30-food seed database with scalable schema support. |
| Volume Estimation (`volume_estimator.py`) | Pure NumPy geometric integration mapping depth maps and masks to cubic centimeters. |
| Macro Calculation (`nutrition_db.py`, `storage.py`) | Deterministic scaling multiplying estimated mass by normalized per-100g nutritional vectors. |
| Daily Tracking & Persistence (`storage.py`) | Embedded SQLite database featuring incremental daily rollups and historical querying. |
| Interactive Nutrition Assistant (`chatbot.py`) | Tool-calling agent execution loop interfacing with Hugging Face Instruct models. |
| VLM Ingredient Classification (`vlm_classifier.py`) | Vision-Language pipeline performing segmented crop analysis via Hugging Face VLM endpoints. |
| Monocular Depth Estimation (`depth_estimation.py`) | Depth-Anything V2 and Depth Pro interfaces producing dense, metric-scaled metric depth maps. |
| Instance Segmentation (`segmentation.py`) | Segment Anything Model 2 (SAM2) integration for instance mask extraction. |

Execute `python demo.py` to run the complete end-to-end processing pipeline—detecting items, calculating geometry-driven volumes, converting to mass, fetching database macros, logging the meal, and displaying daily progress. Run `pytest tests/ -v` to execute the full test suite.

## Model Infrastructure & Configuration

`vlm_classifier.py` and `chatbot.py` interface directly with the Hugging Face Inference API (`huggingface_hub.InferenceClient`) for cost-efficient compute:

* Configure your access key: `export HF_TOKEN=...`
* `vlm_classifier.py` utilizes `HF_VLM_MODEL=Qwen/Qwen2-VL-7B-Instruct` for zero-shot food item classification.
* `chatbot.py` utilizes `HF_CHAT_MODEL=meta-llama/Llama-3.1-8B-Instruct` for function-calling conversational loops.
* Both endpoints can be redirected to custom infrastructure via `HF_ENDPOINT_URL` (such as self-hosted TGI, vLLM, or local deployments).

## Latency Tracking (p50 / p95 / p99)

External inference calls (`vlm_classify` and `chatbot_turn`) are tracked using the latency module in `pipeline/latency.py`. Metrics are aggregated as **p50, p95, and p99 percentile distributions** rather than simple averages to monitor tail latency during cold starts or high-concurrency loads. `demo.py` outputs `LATENCY.summary()` upon completion.

For high-throughput deployments, the concurrent Go proxy in `service/gateway.go` handles incoming edge requests under `/classify` while exposing real-time metrics at `/metrics`. This separates high-concurrency network transport from the downstream Python processing layer.

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

## Engineering Design Principles

**Geometric Volume Estimation:** Monocular depth algorithms require metric scaling to establish exact physical dimensions. `pipeline.py` relies on `scale_from_reference_object()` to establish real-world scale using a known reference object (e.g., standard plate diameter or calibration card). Explicit metric depth geometry provides higher physical consistency than pure VLM-based visual estimations.

**Deterministic Nutritional Retrieval:** Macros are computed using strict linear algebra: `(grams / 100) × per_100g_vector`. `nutrition_db.py` uses Rapidfuzz matching to link raw model predictions to validated database rows. LLMs are never used to guess numerical nutrition values.

**Human-in-the-Loop Verification:** To prevent error propagation across the segmentation, classification, and density estimation stages, `pipeline.py` marks items falling below match-confidence thresholds (`needs_confirmation`) for user review prior to persistence.

**Tool-Grounded Conversational Context:** The chatbot relies on structured tool definitions (`get_daily_summary`, `get_meal_history`) rather than dumping full raw histories directly into the prompt context window. System prompts bound the chatbot strictly to informational query resolution.

## Database Expansion

`data/nutrition_db.csv` includes standard baseline food items. To expand coverage:

* Replace or merge data from **USDA FoodData Central** or the **Indian Food Composition Tables (IFCT)**.
* Maintain the required schema format within `NutritionDB._load()` to ensure compatibility with `FoodEntry`.

## Evaluation Framework (`eval/`)

The repository includes standard benchmark suites under the `eval/` directory:

| Benchmark | Script | Metrics |
|---|---|---|
| Retrieval Accuracy | `eval/retrieval_eval.py` | Top-1 and Top-3 accuracy across label difficulty tiers |
| Confidence Thresholds | `eval/confidence_flag_eval.py` | Precision, recall, and F1-score across threshold sweeps |
| End-to-End Macro Error | `eval/nutrition5k_eval.py` | Macro MAPE evaluation harness against Nutrition5k dataset |
| Segmentation Quality | `eval/segmentation_eval.py` | Mean IoU validation against ground-truth masks |

To run evaluation scripts locally:

* `python -m eval.retrieval_eval`
* `python -m eval.confidence_flag_eval --sweep`
* `python -m pytest tests/test_eval.py -v`
