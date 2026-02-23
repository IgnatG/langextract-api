"""
LangCore API — Python client example.

Demonstrates:
  1. Submit an extraction from raw text.
  2. Submit an extraction from a URL.
  3. Submit a batch.
  4. Poll a task until it completes.

Requirements:
  pip install requests        # or: uv add requests

Usage:
  python examples/python/client.py
"""

from __future__ import annotations

import time
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8000/api/v1"
DEFAULT_PROVIDER = "gpt-4o"  # override with your preferred model
POLL_INTERVAL = 2  # seconds between status checks
POLL_TIMEOUT = 120  # give up after N seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def submit_extraction(payload: dict[str, Any]) -> dict[str, Any]:
    """POST /extract and return the response dict.

    Args:
        payload: Request body as a dict.

    Returns:
        Parsed JSON response containing ``task_id`` and ``status``.

    Raises:
        requests.HTTPError: On non-2xx responses.
    """
    resp = requests.post(f"{API_BASE}/extract", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def submit_batch(payload: dict[str, Any]) -> dict[str, Any]:
    """POST /extract/batch and return the response dict.

    Args:
        payload: Request body as a dict.

    Returns:
        Parsed JSON response containing per-document ``task_id`` values.

    Raises:
        requests.HTTPError: On non-2xx responses.
    """
    resp = requests.post(f"{API_BASE}/extract/batch", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def poll_task(task_id: str) -> dict[str, Any]:
    """Poll GET /tasks/{task_id} until state is SUCCESS or FAILURE.

    Args:
        task_id: The UUID returned by a submission call.

    Returns:
        The final task response dict.

    Raises:
        TimeoutError: If the task does not complete within POLL_TIMEOUT.
        requests.HTTPError: On non-2xx responses.
    """
    deadline = time.monotonic() + POLL_TIMEOUT
    while time.monotonic() < deadline:
        resp = requests.get(f"{API_BASE}/tasks/{task_id}", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        state = data.get("state", "")
        print(f"  [{task_id[:8]}…] state={state}")
        if state in ("SUCCESS", "FAILURE"):
            return data
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"Task {task_id} did not finish within {POLL_TIMEOUT}s")


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


def example_raw_text() -> None:
    """Submit a contract clause as raw text and wait for results."""
    print("\n── Raw text extraction ──────────────────────────")
    result = submit_extraction(
        {
            "raw_text": (
                "AGREEMENT dated January 15, 2025 between Acme Corporation "
                "(Seller) and Beta LLC (Buyer). Purchase price: $12,500 for "
                "500 widgets at $25 each. Payment: net 30 days. "
                "Governed by Delaware law."
            ),
            "provider": DEFAULT_PROVIDER,
            "passes": 1,
            "idempotency_key": "demo-raw-text-001",
            "extraction_config": {
                "temperature": 0.2,
            },
        }
    )
    print(f"Submitted: task_id={result['task_id']}")
    final = poll_task(result["task_id"])
    entities = (final.get("result") or {}).get("entities", [])
    print(f"Done — {len(entities)} entities extracted:")
    for ent in entities:
        print(f"  [{ent['extraction_class']}] {ent['extraction_text']!r}")


def example_url() -> None:
    """Submit a publicly accessible .txt URL for extraction."""
    print("\n── URL extraction ───────────────────────────────")
    result = submit_extraction(
        {
            "document_url": "https://storage.example.com/contracts/agreement-2025.txt",
            "provider": DEFAULT_PROVIDER,
            "extraction_config": {
                "prompt_description": (
                    "Extract any organisations, dates, and legal terms."
                ),
                "temperature": 0.1,
            },
        }
    )
    print(f"Submitted: task_id={result['task_id']}")
    final = poll_task(result["task_id"])
    entities = (final.get("result") or {}).get("entities", [])
    print(f"Done — {len(entities)} entities extracted.")


def example_batch() -> None:
    """Submit multiple documents in a single batch request."""
    print("\n── Batch extraction ─────────────────────────────")
    result = submit_batch(
        {
            "batch_id": "demo-batch-001",
            "documents": [
                {
                    "raw_text": (
                        "Contract A: Acme Corp sells 500 units to Beta LLC "
                        "for $12,500. Delivery Q2 2025."
                    ),
                },
                {
                    "raw_text": (
                        "Contract B: Charlie Enterprises leases warehouse "
                        "space from Delta Holdings at $3,200/month for 24 months."
                    ),
                },
                {
                    "raw_text": (
                        "Contract C: Echo Inc purchases software licences from "
                        "Foxtrot SaaS Ltd at $9,000/year, auto-renewing annually."
                    ),
                },
            ],
            "provider": DEFAULT_PROVIDER,
        }
    )
    print("Batch submitted:")
    task_ids: list[str] = result.get("task_ids") or []
    for tid in task_ids:
        print(f"  task_id={tid}")

    # Poll each document task individually
    for tid in task_ids:
        final = poll_task(tid)
        entities = (final.get("result") or {}).get("entities", [])
        print(f"  [{tid[:8]}…] finished — {len(entities)} entities")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    example_raw_text()
    example_url()
    example_batch()
