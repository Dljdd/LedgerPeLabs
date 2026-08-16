"""Stable errors emitted by the evidence-to-scenario compiler."""


class CompilerError(ValueError):
    """A rejected compilation with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
