"""
VLM ingredient identification.

Scope kept deliberately narrow (see project discussion): the VLM's job is
ONLY to (a) name each segmented food item and (b) estimate a rough portion
class if helpful for the human-confirmation UI. It is explicitly NOT asked
to estimate density, volume, or macros -- those come from deterministic
lookups elsewhere in the pipeline, because free-form visual density/macro
guesses from a VLM are unreliable and unauditable.

This module will make a real call to the Claude API if ANTHROPIC_API_KEY is
set in the environment (using image + per-segment crop). Otherwise
`classify_mock()` returns a fixed label so the pipeline is runnable offline.
"""
from __future__ import annotations

import base64
import dataclasses
import io
import os

import numpy as np


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
    """Real path: calls the Claude API if ANTHROPIC_API_KEY is set."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Use classify_mock() for offline "
            "development, or set the key to hit the real VLM."
        )
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("pip install anthropic") from e

    client = anthropic.Anthropic(api_key=api_key)
    png_bytes = _encode_crop(image, mask)
    b64 = base64.b64encode(png_bytes).decode("utf-8")

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=64,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
    )
    text = "".join(block.text for block in resp.content if block.type == "text").strip()
    uncertain = text.endswith("(uncertain)")
    label = text.replace("(uncertain)", "").strip()
    return ClassificationResult(
        label=label, confidence_note="model flagged uncertainty" if uncertain else "confident"
    )


def classify_mock(label: str = "mixed vegetable curry") -> ClassificationResult:
    return ClassificationResult(label=label, confidence_note="mock -- not a real classification")
