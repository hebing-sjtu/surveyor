"""The browser app: a local server plus a single-page UI over the same library."""

from .app import create_app, run

__all__ = ["create_app", "run"]
