"""Module entrypoint for `python -m cli`."""

from .main import main

if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    raise SystemExit(main())
