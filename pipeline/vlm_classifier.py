"""
VLM ingredient identification.

Scope kept deliberately narrow (see project discussion): the VLM's job is
ONLY to (a) name each segmented food item and (b) estimate a rough portion
class if helpful for the human-confirmation UI. It is explicitly NOT asked
to estimate density, volume, or macros -- those come from deterministic
lookups elsewhere in the pipeline, because free-form visual density/macro
guesses from a VLM are unreliable and unauditable.

This module calls a Hugging Face-hosted vision-language model through the
**free** HF Inference API if `HF_TOKEN` is set in the environment (image +
per-segment crop). Otherwise `classify_mock()` returns a fixed label so the
pipeline is runnable offline.

Why Hugging Face instead of a paid API: a free HF account token
(https://huggingface.co/settings/tokens) gets you a rate-limited but
zero-cost serverless Inference API for supported vision-instruct models, so
this stage of the pipeline -- previously the one part of the repo with a
recurring per-call cost -- now runs at $0. `HF_VLM_MODEL` defaults to
`Qwen/Qwen2-VL-7B-Instruct`; swap it for any other vision-chat model on the
Hub (or point `HF_ENDPOINT_URL` at a self-hosted TGI/vLLM endpoint using the
same client) without touching the call site below.
"""
from __future__ import annotations

import base64
import dataclasses
import io
import os

import numpy as np

from .latency import LATENCY

HF_VLM_MODEL = os.environ.get("HF_VLM_MODEL", "Qwen/Qwen2-VL-7B-Instruct")


@dataclasses.dataclass
class ClassificationResult:
    label: str
    confidence_note: str  # free-text hedge from the model, shown to user, not parsed


PROMPT = (
    "You are identifying a single food item cropped from a photo of a meal. "
    "Respond with ONLY a short common food name (e.g. 'grilled chicken breast', "
    "'cooked white rice', 'mixed vegetable curry') suitable for matching against "
    "a nutrition database. Do not include portion size or preparation detail "
    "beyond what distinguishes the dish. If genuinely unsure between two "
    "options, name the more likely one and add ' (uncertain)' at the end."
)


def _encode_crop(image: np.ndarray, mask: np.ndarray) -> bytes:
    from PIL import Image

    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    crop = image[y0 : y1 + 1, x0 : x1 + 1]
    buf = io.BytesIO()
    Image.fromarray(crop).save(buf, format="PNG")
    return buf.getvalue()


def classify_segment(image: np.ndarray, mask: np.ndarray) -> ClassificationResult:
    """Real path: calls a Hugging Face-hosted VLM (free Inference API) if HF_TOKEN is set.

    Wall-clock time for this call is recorded under the "vlm_classify" name
    in the shared `LATENCY` tracker -- see `pipeline/latency.py`. This is
    the single highest-latency, highest-variance hop in the online path (it
    leaves the process over the network to a shared inference cluster), so
    it's the one most worth watching at p95/p99 rather than just an
    average: a few cold-start or queued requests on HF's side can blow past
    a mean latency budget while barely moving the mean itself.
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN not set. Use classify_mock() for offline development, "
            "or set a free Hugging Face token to hit the real VLM -- "
            "see https://huggingface.co/settings/tokens"
        )
    try:
        from huggingface_hub import InferenceClient
    except ImportError as e:
        raise RuntimeError("pip install huggingface_hub") from e

    endpoint = os.environ.get("HF_ENDPOINT_URL")  # optional: self-hosted TGI/vLLM endpoint
    client = InferenceClient(model=endpoint or HF_VLM_MODEL, token=hf_token)

    png_bytes = _encode_crop(image, mask)
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64}"

    with LATENCY.track("vlm_classify"):
        completion = client.chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
            max_tokens=64,
        )

    text = (completion.choices[0].message.content or "").strip()
    uncertain = text.endswith("(uncertain)")
    label = text.replace("(uncertain)", "").strip()
    return ClassificationResult(
        label=label, confidence_note="model flagged uncertainty" if uncertain else "confident"
    )


def classify_mock(label: str = "mixed vegetable curry") -> ClassificationResult:
    return ClassificationResult(label=label, confidence_note="mock -- not a real classification")
