"""OPTIONAL, OFF BY DEFAULT — consult a vision-language model (Gemma-3n-E4B
served by vLLM) to disambiguate a *very hard* crop among a few candidate classes.

This is intentionally a thin, guarded overlay — it is NOT used unless the caller
passes --vlm AND a crop is below the hard-margin threshold. Be careful with it:
  * It is advisory only; the answer is constrained to the supplied candidates and
    rejected if the model returns anything else.
  * Use it sparingly (a handful of genuinely ambiguous brand/object crops), not
    as the main classifier — it is slower and can hallucinate brands.
  * Requires a running vLLM server. Configure via env:
        DETECTY_VLM_URL    (default http://localhost:8000/v1)
        DETECTY_VLM_MODEL  (default google/gemma-3n-e4b-it)
        DETECTY_VLM_KEY    (default "EMPTY")
  * Serve, e.g.:
        vllm serve google/gemma-3n-e4b-it --max-model-len 4096
"""
import base64
import io
import os
from typing import Optional, Sequence

_ENV_URL = "DETECTY_VLM_URL"
_ENV_MODEL = "DETECTY_VLM_MODEL"
_ENV_KEY = "DETECTY_VLM_KEY"


def available() -> bool:
    try:
        import openai  # noqa: F401
        return True
    except Exception:
        return False


def _b64(pil) -> str:
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def consult(pil, candidates: Sequence[str], hints: str = "",
            timeout: float = 30.0) -> Optional[str]:
    """Return one of `candidates` the VLM judges best, or None on any failure /
    out-of-set answer. Never raises — disabled paths just return None.
    """
    if not candidates or not available():
        return None
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url=os.environ.get(_ENV_URL, "http://localhost:8000/v1"),
            api_key=os.environ.get(_ENV_KEY, "EMPTY"),
            timeout=timeout,
        )
        model = os.environ.get(_ENV_MODEL, "google/gemma-3n-e4b-it")
        opts = ", ".join(candidates)
        prompt = (
            "You are identifying a single RoboCup@Home object from a tight crop. "
            f"Choose EXACTLY ONE label from this list: [{opts}]. "
            "Use the brand/logo/text and colour. If genuinely unsure, pick the "
            "closest. Reply with ONLY the label, nothing else."
        )
        if hints:
            prompt += f" Hints: {hints}"
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{_b64(pil)}"}},
                ],
            }],
            max_tokens=16,
            temperature=0.0,
        )
        ans = (resp.choices[0].message.content or "").strip().lower()
        for c in candidates:                      # constrain to the candidate set
            if c.lower() == ans or c.lower() in ans:
                return c
        return None
    except Exception:
        return None
