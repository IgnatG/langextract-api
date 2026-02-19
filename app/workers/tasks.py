"""
Backward-compatible re-exports.

Task implementations live in ``extract_task`` and ``batch_task``.
This module re-exports them so that existing ``from
app.workers.tasks import …`` statements keep working.
"""

from app.workers.batch_task import finalize_batch
from app.workers.extract_task import extract_document

__all__ = ["extract_document", "finalize_batch"]
