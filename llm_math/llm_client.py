"""OpenAI-compatible chat client for the bare-LLM framework.

Thin wrapper around the vendored `_GenericLLM` from `math_prove/validator.py`.
Adds:
  - configurable `temperature` and `max_tokens` (the vendored client hard-codes 0.0 / 512)
  - 3-attempt retry on connection errors and HTTP 429 / 5xx with 1/2/4s backoff
  - a `threading.Lock` so a single client can be shared across the worker pool
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

import requests

from llm_math.vendor import _GenericLLM


class BareLLMClient:
    """Lock-protected, retrying OpenAI-compatible chat client.

    The vendored `_GenericLLM` already implements the JSON-mode fallback
    (it retries without `response_format` if the endpoint rejects
    `response_format: {"type": "json_object"}`). This wrapper adds higher-level
    resilience for transient network / rate-limit errors.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str],
        api_base: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 120,
        max_retries: int = 3,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        if not api_base:
            raise ValueError("api_base is required")
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = int(timeout)
        self.max_retries = max(1, int(max_retries))
        self._lock = threading.Lock()

    def chat(
        self,
        messages: List[Dict[str, str]],
        response_format_json: bool = True,
    ) -> str:
        """Send a chat completion; return the assistant text content.

        Retries up to `max_retries` times on connection errors and HTTP
        429 / 5xx, with exponential backoff (1s, 2s, 4s).
        """
        url = self.api_base
        headers = self._build_headers()
        data = self._build_payload(messages, response_format_json=response_format_json)
        last_error: Optional[BaseException] = None

        for attempt in range(self.max_retries):
            with self._lock:
                try:
                    resp = requests.post(
                        url, headers=headers, json=data, timeout=self.timeout
                    )
                    self._raise_for_status(resp)
                    body = resp.json()
                    return self._extract_content(body)
                except (requests.exceptions.RequestException, RuntimeError) as exc:
                    last_error = exc
                    if attempt + 1 < self.max_retries and self._should_retry(exc, resp if "resp" in locals() else None):
                        time.sleep(2 ** attempt)
                        continue
                    raise

        raise RuntimeError(f"BareLLMClient.chat failed after {self.max_retries} attempts: {last_error}")

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        response_format_json: bool,
    ) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "n": 1,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if response_format_json:
            data["response_format"] = {"type": "json_object"}
        return data

    @staticmethod
    def _raise_for_status(resp: requests.Response) -> None:
        if 200 <= resp.status_code < 300:
            return
        if resp.status_code in (429, 500, 502, 503, 504):
            raise requests.exceptions.HTTPError(
                f"transient status {resp.status_code}: {resp.text[:200]}",
                response=resp,
            )
        raise RuntimeError(
            f"non-retryable status {resp.status_code}: {resp.text[:200]}"
        )

    @staticmethod
    def _should_retry(
        exc: BaseException, resp: Optional[requests.Response]
    ) -> bool:
        if isinstance(exc, requests.exceptions.HTTPError) and resp is not None:
            return resp.status_code in (429, 500, 502, 503, 504)
        if isinstance(exc, requests.exceptions.RequestException):
            return True
        # RuntimeError for non-2xx non-retryable statuses: caller already raised.
        return False

    @staticmethod
    def _extract_content(body: Dict[str, Any]) -> str:
        if "choices" in body and body["choices"]:
            return str(body["choices"][0]["message"]["content"])
        if "error" in body:
            raise RuntimeError(str(body["error"]))
        raise RuntimeError(f"unexpected response body: {str(body)[:200]}")


__all__ = ["BareLLMClient"]
