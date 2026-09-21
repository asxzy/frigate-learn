"""VLM visual inspection module.

Uses an oMLX-hosted VLM (OpenAI-compatible chat completions API) to make an
independent reading of each Frigate crop. The VLM is a blind verifier: it is
never told the expected class, SAM's guess, or any label. It only sees the
crop with the detector box drawn on it and reports its own judgement:

VLM schema (strict):
  {
    "bbox_covers_object": bool,
    "class_label": "person"
  }

``class_label`` is the VLM's own label for the object, or ``""`` when there
is no object in the crop. Any missing field, wrong type, or malformed JSON
=> DROP (rejected). Agreement between the independent SAM and VLM readings,
and mask/box alignment against the Frigate bbox, are computed later by the
decision engine — never by the model.
"""

from __future__ import annotations

import base64
import io
import json
import queue as _queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
from PIL import Image

from .types import VlmResult

VLM_REQUIRED_KEYS = frozenset({
    "bbox_covers_object",
    "class_label",
})


class VlmResultError(Exception):
    """The VLM returned a response that cannot be parsed as a valid schema."""


class VlmTransportError(VlmResultError):
    """The VLM *service* failed (network/HTTP); the sample must be retried."""


_SYSTEM_PROMPT = """You are a strict, unbiased visual inspection assistant.
You are auditing one object-detection crop from a camera.
The image shows the crop with a single GREEN rectangle marking the detector's box.

You are NOT told which class the detector predicted. Judge the image on its own.

Answer TWO questions about this crop:
  bbox_covers_object: Does the green rectangle actually cover the object?
      true = the box contains the object, tightly enough that the whole object
             is inside it and the box is not mostly empty background.
      false = the box is misplaced, mostly empty, or truncates the object.
  class_label: What is the object in the crop? Give a single noun label
      (e.g. person, car, bird, bicycle, motorcycle, cat, dog, suitcase).
      If there is NO object (only background, shadow, or noise), use "" .

Return ONLY a JSON object with exactly these two keys:
{"bbox_covers_object": bool, "class_label": str}
No explanation, no markdown, no extra text.
"""


@dataclass
class OmlixReconcilerSettings:
    base_url: str = "http://127.0.0.1:8080/v1"
    api_key: str = ""
    model: str = ""
    temperature: float = 0.0
    timeout_seconds: float = 180.0
    max_retries: int = 2
    json_mode: bool = True


class VisualReconciler(Protocol):
    """Minimal interface for the VLM inspection step."""

    vlm_key: str

    def reconcile(
        self,
        image_path: str | Path,
        class_options: list[str] | None = None,
    ) -> VlmResult:
        ...


def _image_to_data_url(image_path: str | Path, max_size: int = 1024) -> str:
    """Encode a crop as a base64 JPEG data URL for the VLM request."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def parse_vlm_result(raw: dict[str, Any]) -> VlmResult:
    """Strict parser: both required fields must be present with right types.

    ``bbox_covers_object`` must be bool; ``class_label`` must be str (possibly
    empty when no object is present). Raises :class:`VlmResultError` on any
    missing field, wrong type, or malformed structure. Nothing is inferred.
    """
    if not isinstance(raw, dict):
        raise VlmResultError(f"expected JSON object, got {type(raw).__name__}")
    missing = VLM_REQUIRED_KEYS - raw.keys()
    if missing:
        raise VlmResultError(f"missing required fields: {sorted(missing)}")
    covers = raw["bbox_covers_object"]
    if not isinstance(covers, bool):
        raise VlmResultError(
            f"field 'bbox_covers_object' must be bool, got {type(covers).__name__}"
        )
    label = raw["class_label"]
    if not isinstance(label, str):
        raise VlmResultError(
            f"field 'class_label' must be str, got {type(label).__name__}"
        )
    return VlmResult(
        bbox_covers_object=covers,
        class_label=label.strip(),
        raw=raw,
    )


def _extract_json_from_response(text: str) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from wrapped model text.

    Finds the first ``{`` and the last ``}`` and parses that slice; anything
    the model wrapped around the JSON (prose, fences) is ignored.
    """
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise VlmResultError(f"no JSON object found in response: {text[:200]!r}")


@dataclass
class OmlixReconciler:
    """oMLX-backed VLM reconciler (OpenAI-compatible /v1/chat/completions).

    Talks to a local oMLX server. The image is sent as a base64 JPEG data URL
    in the user message. JSON mode is requested when supported; when the server
    does not honour response_format, the parser falls back to plain text.

    Every attempt runs on a daemon worker thread against a fresh client and
    the caller's thread waits on the result with a hard wall-clock deadline.
    A stalled server can therefore never hang the pipeline: the timeout path
    hands the client to a daemon close (never awaited) and retries/aborts.
    """

    settings: OmlixReconcilerSettings
    vlm_key: str = ""
    _client: httpx.Client | None = None

    def __post_init__(self) -> None:
        if not self.vlm_key:
            self.vlm_key = (
                f"omlx:{self.settings.base_url}:{self.settings.model}:"
                f"{self.settings.temperature}:{self.settings.json_mode}"
            )

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self.settings.api_key:
            h["Authorization"] = f"Bearer {self.settings.api_key}"
        return h

    def _new_client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.settings.base_url,
            headers=self._headers(),
            timeout=httpx.Timeout(self.settings.timeout_seconds, connect=10.0),
        )

    def _post_attempt(
        self, client: httpx.Client, payload: dict[str, Any], q: "_queue.Queue"
    ) -> None:
        try:
            q.put(("ok", client.post("/chat/completions", json=payload)))
        except Exception as exc:
            q.put(("err", exc))

    def reconcile(
        self,
        image_path: str | Path,
        class_options: list[str] | None = None,
    ) -> VlmResult:
        """Inspect one object crop via the VLM; returns the independent verdict.

        ``class_options`` narrows the label vocabulary the model may answer
        with; it is a hint, never the expected class. Raises
        :class:`VlmResultError` on transport or schema failure.
        """
        data_url = _image_to_data_url(image_path)
        if class_options:
            vocab = ", ".join(class_options)
            prompt = (
                "Allowed labels: " + vocab
                + " . Use one of these, or \"\" if no object is present."
            )
        else:
            prompt = "Label the object with a single noun, or \"\" if absent."
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ]
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
        }
        if self.settings.json_mode:
            payload["response_format"] = {"type": "json_object"}
        last_err: Exception | None = None
        for _attempt in range(1 + self.settings.max_retries):
            client = self._new_client()
            q: "_queue.Queue" = _queue.Queue(maxsize=1)
            threading.Thread(
                target=self._post_attempt,
                args=(client, payload, q),
                daemon=True,
            ).start()
            try:
                kind, item = q.get(timeout=self.settings.timeout_seconds)
            except _queue.Empty:
                last_err = VlmTransportError(
                    f"VLM timed out after {self.settings.timeout_seconds:.0f}s"
                )
                threading.Thread(target=client.close, daemon=True).start()
                continue
            if kind == "err":
                last_err = item
                client.close()
                continue
            resp = item
            try:
                resp.raise_for_status()
                body = resp.json()
                text = body["choices"][0]["message"]["content"]
                raw = _extract_json_from_response(text)
                return parse_vlm_result(raw)
            except httpx.HTTPError as exc:
                last_err = exc
                continue
            except (KeyError, IndexError, json.JSONDecodeError, VlmResultError) as exc:
                raise VlmResultError(f"VLM response malformed: {exc}") from exc
            finally:
                client.close()
        raise VlmTransportError(
            f"VLM transport failed after {1 + self.settings.max_retries} attempts: {last_err}"
        )

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None


def build_reconciler(
    base_url: str,
    model: str,
    *,
    api_key: str = "",
    temperature: float = 0.0,
    timeout_seconds: float = 180.0,
    max_retries: int = 2,
    json_mode: bool = True,
) -> OmlixReconciler:
    settings = OmlixReconcilerSettings(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        json_mode=json_mode,
    )
    return OmlixReconciler(settings=settings)


__all__ = [
    "OmlixReconciler",
    "VisualReconciler",
    "VlmResultError",
    "VlmTransportError",
    "build_reconciler",
    "parse_vlm_result",
]