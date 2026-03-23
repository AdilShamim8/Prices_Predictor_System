# Contributing to NeuroBridge

Thank you for helping improve cognitive accessibility for AI users.

## Development Setup

1. Create and activate a virtual environment.
2. Install editable dependencies:
   pip install -e ".[dev]"
3. Run checks before opening a PR:
   - pytest
   - ruff check .
   - black --check .
   - mypy neurobridge

## Pull Request Guidelines

1. Keep changes focused and small.
2. Add or update tests for behavior changes.
3. Update docs/examples for public API changes.
4. Use clear commit messages.

## Code Standards

- Python 3.9+ compatible
- Type hints for public APIs
- Docstrings for modules, classes, and public methods
- Prefer explicit errors over silent failures

## Reporting Issues

Please include:
- Reproduction steps
- Expected behavior
- Actual behavior
- Runtime details (Python version, OS)
