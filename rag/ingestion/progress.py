"""Small tqdm-backed progress helpers for ingestion jobs."""

from __future__ import annotations

from tqdm import tqdm


class ProgressBar:
    """Lightweight wrapper around tqdm for the ingestion pipeline."""

    def __init__(self, total: int | None, label: str, width: int = 28) -> None:
        self._bar = tqdm(
            total=None if total is None else max(0, total),
            desc=label,
            ncols=width + 32,
            leave=True,
        )
        self._finished = False

    def update(self, increment: int = 1) -> None:
        """Advance the bar."""

        if not self._finished:
            self._bar.update(increment)

    def finish(self, message: str | None = None) -> None:
        """Mark the bar as complete and optionally print a final message."""

        if self._finished:
            return

        self._bar.close()
        self._finished = True
        if message:
            print(message)
