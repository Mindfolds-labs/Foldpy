"""FoldPy exception hierarchy."""

from __future__ import annotations


class FoldPyError(Exception):
    """Base exception for all FoldPy errors."""


class ValidationError(FoldPyError, ValueError):
    """Raised when input data validation fails."""


class EmptyInputError(FoldPyError):
    """Raised when an operation receives an empty input collection."""
