"""VLM provider abstraction (Phase 4).

``VLMProvider`` is a Protocol: given image paths it returns per-image lists of
``VLMObject``. The reference implementation speaks the OpenAI-compatible chat
completions API (works with litellm, vLLM, Ollama's OpenAI bridge, etc.) with
images encoded as base64 JPEG data URLs.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path
from typing import Protocol, Sequence

import httpx
from PIL import Image

from .schema import VLMObject, VLMValidationError, extract_json_from_response, validate_vlm_output

logger = logging.getLogger(__name__)

_RETRY_STATUS = {429, 500, 502, 503, 504}


class VLMProvider(Protocol):
    def verify(self, images: Sequence[Path]) -> Sequence[Sequence[VLMObject]]: ...


class OpenAICompatibleProvider:
    """OpenAI-compatible chat completions client for image verification."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        *,
        temperature: float = 0.0,
        timeout_seconds: float = 120.0,
        system_prompt: str | None = None,
        allowed_labels: Sequence[str] | None = None,
        max_retries: int = 2,
        backoff_seconds: float = 2.0,
    ) -> None:
        if not base_url:
            raise VLMValidationError("vlm.base_url is required for the openai provider")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout_seconds
        self.system_prompt = system_prompt
        self.allowed_labels = list(allowed_labels) if allowed_labels else None
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds),
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {"Accept": "application/json"},
        )

    def close(self) -> None:
        self._client.close()

    # --- internals --------------------------------------------------------

    @staticmethod
    def _image_data_url(path: Path) -> str:
        with Image.open(path) as img:
            img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    def verify(self, images: Sequence[Path]) -> Sequence[Sequence[VLMObject]]:
        """Verify each image independently (safer for detection API servers)."""
        results: list[Sequence[VLMObject]] = []
        for image in images:
            results.append(self._verify_one(image))
        return results

    def _verify_one(self, image: Path) -> Sequence[VLMObject]:
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            try:
                messages = []
                if self.system_prompt:
                    messages.append({"role": "system", "content": self.system_prompt})
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Detect objects in this image. Bounding box "
                                    "coordinates must be normalized to the image "
                                    "size (each value in [0, 1]), never pixel values."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": self._image_data_url(image)},
                            },
                        ],
                    }
                )
                response = self._client.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": self.model,
                        "temperature": self.temperature,
                        "messages": messages,
                    },
                )
                if response.status_code in _RETRY_STATUS and attempt < attempts - 1:
                    self._backoff(attempt)
                    continue
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                parsed = extract_json_from_response(str(content))
                return validate_vlm_output(parsed, allowed_labels=self.allowed_labels)
            except VLMValidationError:
                # non-JSON / out-of-schema output: the model is stochastic, so
                # give it another sample before surfacing the rejection
                if attempt < attempts - 1:
                    self._backoff(attempt)
                    continue
                raise
            except Exception as exc:
                if (
                    attempt < attempts - 1
                    and isinstance(exc, httpx.HTTPStatusError)
                    and exc.response.status_code in _RETRY_STATUS
                ):
                    self._backoff(attempt)
                    continue
                raise VLMValidationError(f"VLM request failed for {image}: {exc}") from exc
        return []  # pragma: no cover - unreachable (raised above)

    def _backoff(self, attempt: int) -> None:
        time.sleep(self.backoff_seconds * (2 ** attempt))


def build_provider(
    base_url: str | None = None,
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
    *,
    provider_name: str = "openai",
    system_prompt: str | None = None,
    allowed_labels: Sequence[str] | None = None,
    **kwargs,
) -> VLMProvider:
    if provider_name == "openai":
        return OpenAICompatibleProvider(
            base_url=base_url or "",
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            allowed_labels=allowed_labels,
            **kwargs,
        )
    raise ValueError(f"unknown vlm provider: {provider_name!r}")


__all__ = ["VLMProvider", "OpenAICompatibleProvider", "build_provider"]