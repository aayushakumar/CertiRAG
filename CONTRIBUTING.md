# Contributing to CertiRAG

Thank you for your interest in contributing to CertiRAG! This document provides
guidelines and information to help you get started.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/)
code of conduct. By participating, you are expected to uphold this code. Please
report unacceptable behavior to the maintainers.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/CertiRAG.git
   cd CertiRAG
   ```
3. **Add upstream** remote:
   ```bash
   git remote add upstream https://github.com/aayushakumar/CertiRAG.git
   ```

## Development Setup

### Prerequisites

- Python 3.10+
- Git

### Installation

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install with development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Copy environment config
cp .env.example .env
```

### Verify Setup

```bash
# Run tests
make test-all

# Run linter
make lint

# Run type checker
make typecheck
```

## Project Structure

```
certirag/
├── claim_ir/       # Claim extraction & normalization
├── ingest/         # Document chunking & indexing
├── retrieve/       # Hybrid retrieval (BM25 + dense)
├── verify/         # NLI verification engine
├── render/         # Policy engine & certificates
├── schemas/        # Pydantic v2 data models
├── config.py       # Configuration management
├── pipeline.py     # Orchestrator
├── cli.py          # CLI entry point
└── utils.py        # Shared utilities

tests/
├── unit/           # Fast, isolated tests
├── integration/    # Pipeline integration tests
└── adversarial/    # Robustness tests

eval/               # Benchmark evaluation framework
ui/                 # Streamlit web interface
```

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feat/claim-splitting-improvement`
- `fix/bm25-single-doc-edge-case`
- `docs/update-api-reference`
- `test/add-normalizer-coverage`

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add hedge detection to claim normalizer
fix: handle empty evidence list in renderer policy
docs: update installation guide for Colab
test: add property-based tests for schema validation
refactor: extract span scoring into separate module
```

### Key Design Principles

1. **Fail-Closed by Default**: The renderer MUST block unverified claims. Never
   allow LLM output to bypass the policy engine.

2. **Deterministic Rendering**: `RendererPolicy.decide()` is pure logic — no ML,
   no randomness, no LLM calls. Keep it that way.

3. **Schema-First**: All data flows through Pydantic v2 schemas. Add new fields
   to schemas before using them in pipeline code.

4. **Theorem 1 Invariant**: Any change to the verification/rendering logic must
   preserve:
   ```
   VERIFIED ⟺ (entail ≥ τ_e) ∧ (¬contradict ≥ τ_c) ∧ (evidence ≥ 1)
   ```

5. **Dual-Mode**: All features must work in both LITE (CPU/API) and FULL (GPU)
   modes. Use the `CertiRAGConfig.mode` field to branch where needed.

## Testing

### Running Tests

```bash
# All tests
make test-all

# Unit tests only (fast)
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Adversarial tests
pytest tests/adversarial/ -v

# With coverage
pytest tests/ --cov=certirag --cov-report=html
```

### Writing Tests

- **Unit tests** go in `tests/unit/` — no network, no GPU, no external services
- **Integration tests** go in `tests/integration/` — test pipeline flows
- **Adversarial tests** go in `tests/adversarial/` — test robustness

Use [Hypothesis](https://hypothesis.readthedocs.io/) for property-based tests
where applicable (schema validation, score bounds, policy invariants).

Example test:
```python
from hypothesis import given, strategies as st

@given(tau_e=st.floats(0.0, 1.0), tau_c=st.floats(0.0, 1.0))
def test_policy_thresholds_bounded(tau_e, tau_c):
    """Policy thresholds must be in [0, 1]."""
    policy = RendererPolicy(tau_entail=tau_e, tau_contradict=tau_c)
    assert 0 <= policy.tau_entail <= 1
    assert 0 <= policy.tau_contradict <= 1
```

### Test Requirements

- All PRs must maintain ≥90% test pass rate
- New features require corresponding tests
- Breaking changes require updated tests
- The adversarial test suite must pass on every PR

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check
ruff check certirag/ tests/

# Fix auto-fixable issues
ruff check --fix certirag/ tests/

# Format
ruff format certirag/ tests/
```

### Style Rules

- Line length: 100 characters
- Type hints: Use them for public APIs, optional for internal helpers
- Docstrings: Google style for all public functions and classes
- Imports: Use absolute imports (`from certirag.schemas.claim_ir import Claim`)

## Pull Request Process

1. **Create a branch** from `main` with a descriptive name
2. **Make your changes** with clear, atomic commits
3. **Run tests** (`make test-all`) and fix any failures
4. **Run linting** (`make lint`) and fix any issues
5. **Update documentation** if your changes affect the public API
6. **Open a Pull Request** with:
   - Clear title following conventional commits
   - Description of what and why
   - Link to related issue(s)
   - Screenshots for UI changes
7. **Address review comments** promptly
8. **Squash and merge** once approved

### PR Checklist

- [ ] Tests pass (`make test-all`)
- [ ] Linting clean (`make lint`)
- [ ] Documentation updated (if needed)
- [ ] Schema changes are backward-compatible (if applicable)
- [ ] Theorem 1 invariant preserved (if touching render/verify)

## Reporting Issues

### Bug Reports

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) and include:
- Python version and OS
- CertiRAG version (`certirag --version` or `pip show certirag`)
- Steps to reproduce
- Expected vs actual behavior
- Error traceback (if applicable)

### Feature Requests

Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md) and describe:
- The problem you're trying to solve
- Your proposed solution
- Alternatives you've considered

## Questions?

Open a [Discussion](https://github.com/aayushakumar/CertiRAG/discussions) for:
- "How do I...?" questions
- Ideas and brainstorming
- Showcasing your use of CertiRAG
