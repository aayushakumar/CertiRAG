"""
Allow running CertiRAG as a module: ``python -m certirag``.

This delegates to the CLI entry point so that both
``certirag`` (console script) and ``python -m certirag``
behave identically.
"""

from certirag.cli import main

if __name__ == "__main__":
    main()
