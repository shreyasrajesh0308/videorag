from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


CHAT_URL = "https://api.openai.com/v1/chat/completions"
EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"


def encode_image_data_url(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def request_chat(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    retries: int,
    json_object: bool = True,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_object:
        payload["response_format"] = {"type": "json_object"}

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        CHAT_URL,
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )

    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            retryable = e.code in (408, 409, 429, 500, 502, 503, 504)
            if not retryable or attempt >= retries:
                raise
            time.sleep(min(8.0, 0.5 * (2**attempt)))
        except urllib.error.URLError:
            if attempt >= retries:
                raise
            time.sleep(min(8.0, 0.5 * (2**attempt)))
    raise RuntimeError("chat request failed")


def request_embeddings(
    *,
    api_key: str,
    model: str,
    text: str,
    timeout_s: float,
    retries: int,
) -> list[float]:
    payload = {"model": model, "input": text}
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        EMBEDDINGS_URL,
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            emb = data["data"][0]["embedding"]
            if not isinstance(emb, list):
                raise ValueError("unexpected embeddings response")
            return [float(x) for x in emb]
        except urllib.error.HTTPError as e:
            retryable = e.code in (408, 409, 429, 500, 502, 503, 504)
            if not retryable or attempt >= retries:
                raise
            time.sleep(min(8.0, 0.5 * (2**attempt)))
        except urllib.error.URLError:
            if attempt >= retries:
                raise
            time.sleep(min(8.0, 0.5 * (2**attempt)))
    raise RuntimeError("embeddings request failed")
