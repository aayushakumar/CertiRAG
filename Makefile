# ─────────────────────────────────────────────────────────────
# CertiRAG Makefile
# ─────────────────────────────────────────────────────────────
# Targets for development, testing, evaluation, and deployment.
#
# Usage:
#   make install        # Install dependencies
#   make test           # Run all tests
#   make lint           # Run linters
#   make eval-alce      # Run ALCE benchmark
#   make demo           # Launch Streamlit demo
#   make docker-build   # Build Docker image
# ─────────────────────────────────────────────────────────────

.PHONY: install install-lite install-full install-dev \
        test test-unit test-integration test-adversarial test-all \
        lint format typecheck \
        eval eval-alce eval-ragtruth eval-aggrefact \
        demo clean docker-build docker-run \
        schemas docs

# ── Variables ───────────────────────────────────────────────
PYTHON ?= python3
PIP ?= pip
PYTEST ?= pytest
STREAMLIT ?= streamlit
DOCKER ?= docker

PROJECT = certirag
IMAGE_NAME = certirag
IMAGE_TAG = latest

DATA_DIR ?= data
EVAL_DIR ?= eval_results
SCHEMA_DIR ?= schemas

# ── Install ─────────────────────────────────────────────────

install: ## Install core + dev dependencies
	$(PIP) install -e ".[dev]"

install-lite: ## Install for Codespaces (CPU/API mode)
	$(PIP) install -e ".[lite,dev]"

install-full: ## Install for GPU mode (Colab Pro / local)
	$(PIP) install -e ".[full,dev]"

install-dev: ## Install everything
	$(PIP) install -e ".[lite,full,dev]"

# ── Test ────────────────────────────────────────────────────

test: test-unit ## Run default test suite (unit only)

test-unit: ## Run unit tests
	$(PYTEST) tests/unit/ -v --tb=short -m "not slow"

test-integration: ## Run integration tests
	$(PYTEST) tests/integration/ -v --tb=short

test-adversarial: ## Run adversarial robustness tests
	$(PYTEST) tests/adversarial/ -v --tb=short

test-all: ## Run all tests (unit + integration + adversarial)
	$(PYTEST) tests/ -v --tb=short

test-coverage: ## Run tests with coverage report
	$(PYTEST) tests/ -v --cov=$(PROJECT) --cov-report=html --cov-report=term

# ── Linting ─────────────────────────────────────────────────

lint: ## Run all linters
	ruff check $(PROJECT) tests eval
	ruff format --check $(PROJECT) tests eval

format: ## Auto-format code
	ruff format $(PROJECT) tests eval
	ruff check --fix $(PROJECT) tests eval

typecheck: ## Run type checking
	mypy $(PROJECT) --ignore-missing-imports

# ── Evaluation ──────────────────────────────────────────────

eval-alce: ## Run ALCE benchmark
	$(PYTHON) -m certirag.cli eval --benchmark alce --data-dir $(DATA_DIR)/alce --output-dir $(EVAL_DIR)

eval-ragtruth: ## Run RAGTruth benchmark
	$(PYTHON) -m certirag.cli eval --benchmark ragtruth --data-dir $(DATA_DIR)/ragtruth --output-dir $(EVAL_DIR)

eval-aggrefact: ## Run AggreFact benchmark
	$(PYTHON) -m certirag.cli eval --benchmark aggrefact --data-dir $(DATA_DIR)/aggrefact --output-dir $(EVAL_DIR)

eval: eval-alce eval-ragtruth eval-aggrefact ## Run all benchmarks

# ── Demo ────────────────────────────────────────────────────

demo: ## Launch Streamlit demo
	$(STREAMLIT) run ui/app.py --server.port 8501

# ── Schemas ─────────────────────────────────────────────────

schemas: ## Export JSON schemas
	$(PYTHON) -m certirag.cli export-schemas --output-dir $(SCHEMA_DIR)

# ── Docker ──────────────────────────────────────────────────

docker-build: ## Build Docker image
	$(DOCKER) build -t $(IMAGE_NAME):$(IMAGE_TAG) .

docker-run: ## Run Docker container
	$(DOCKER) run -it --rm \
		-p 8501:8501 \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/.env:/app/.env \
		$(IMAGE_NAME):$(IMAGE_TAG)

docker-run-gpu: ## Run Docker container with GPU
	$(DOCKER) run -it --rm --gpus all \
		-p 8501:8501 \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/.env:/app/.env \
		$(IMAGE_NAME):$(IMAGE_TAG)

# ── Cleanup ─────────────────────────────────────────────────

clean: ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ── Help ────────────────────────────────────────────────────

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
