"""
CertiRAG CLI
==============

Command-line interface for running CertiRAG pipeline,
evaluation, and utility commands.

Usage:
    python -m certirag.cli ingest --docs data/docs/
    python -m certirag.cli run "What is the capital of France?"
    python -m certirag.cli eval --benchmark alce --subset asqa
    python -m certirag.cli validate --schema all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from certirag.config import get_config
from certirag.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(
        prog="certirag",
        description="CertiRAG: Certified Retrieval-Augmented Generation",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--mode", choices=["lite", "full"], default=None)
    parser.add_argument("--verbose", "-v", action="store_true")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ── ingest ──────────────────────────────────────────────────
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("--docs", required=True, help="Path to documents directory or JSONL")
    ingest_parser.add_argument("--index-dir", default="data/index", help="Output index directory")

    # ── run ─────────────────────────────────────────────────────
    run_parser = subparsers.add_parser("run", help="Run CertiRAG on a query")
    run_parser.add_argument("question", help="Question to verify")
    run_parser.add_argument("--display", choices=["strict", "mixed", "debug"], default="strict")
    run_parser.add_argument("--output", type=str, default=None, help="Output JSON path")

    # ── eval ────────────────────────────────────────────────────
    eval_parser = subparsers.add_parser("eval", help="Run evaluation benchmark")
    eval_parser.add_argument("--benchmark", choices=["alce", "ragtruth", "aggrefact"], required=True)
    eval_parser.add_argument("--subset", type=str, default=None)
    eval_parser.add_argument("--data-dir", required=True, help="Path to benchmark data")
    eval_parser.add_argument("--max-examples", type=int, default=None)
    eval_parser.add_argument("--output-dir", default="eval_results")

    # ── validate ────────────────────────────────────────────────
    validate_parser = subparsers.add_parser("validate", help="Validate data schemas")
    validate_parser.add_argument("--input", required=True, help="JSON file to validate")
    validate_parser.add_argument("--schema", choices=["evidence", "claim_ir", "verification", "certificate"])

    # ── export-schemas ──────────────────────────────────────────
    schema_parser = subparsers.add_parser("export-schemas", help="Export JSON schemas")
    schema_parser.add_argument("--output-dir", default="schemas")

    args = parser.parse_args()

    if args.verbose:
        setup_logging(level="DEBUG")
    else:
        setup_logging(level="INFO")

    if args.mode:
        import os
        os.environ["CERTIRAG_MODE"] = args.mode

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "export-schemas":
        cmd_export_schemas(args)
    else:
        parser.print_help()
        sys.exit(1)


def cmd_ingest(args):
    """Ingest documents into the pipeline."""
    from certirag.pipeline import CertiRAGPipeline

    config = get_config(args.config)
    pipeline = CertiRAGPipeline(config)

    docs_path = Path(args.docs)
    documents = []

    if docs_path.is_file():
        # JSONL format
        with open(docs_path) as f:
            for line in f:
                documents.append(json.loads(line))
    elif docs_path.is_dir():
        # Directory of text files
        for txt_file in sorted(docs_path.glob("*.txt")):
            documents.append({
                "doc_id": txt_file.stem,
                "text": txt_file.read_text(),
                "title": txt_file.stem,
            })
        for json_file in sorted(docs_path.glob("*.json")):
            with open(json_file) as f:
                documents.append(json.load(f))
    else:
        print(f"Error: {docs_path} not found")
        sys.exit(1)

    print(f"Ingesting {len(documents)} documents...")
    pipeline.ingest(documents)
    print("Ingestion complete.")


def cmd_run(args):
    """Run CertiRAG on a single query."""
    from certirag.pipeline import CertiRAGPipeline

    config = get_config(args.config)
    pipeline = CertiRAGPipeline(config)

    # Check if index exists
    if not pipeline._is_ingested:
        print("Error: No documents ingested. Run 'certirag ingest' first.")
        sys.exit(1)

    result = pipeline.run(args.question, mode=args.display)

    # Display results
    print(f"\nQuestion: {args.question}")
    print(f"Mode: {args.display}")
    print(f"Latency: {result.timings.get('total_ms', 0):.0f}ms\n")

    for claim in result.verified_claims:
        icon = {"VERIFIED": "✅", "UNVERIFIED": "⚠️", "BLOCKED": "❌"}.get(claim["state"], "?")
        print(f"  {icon} {claim['text']}")

    if result.hidden_claims:
        print(f"\n  ({len(result.hidden_claims)} claims hidden)")

    if result.certificate:
        print(f"\n  Certificate: {result.certificate.run_id}")
        print(f"  Integrity: {'✅' if result.certificate.verify_integrity() else '❌'}")

    print(f"\n  Stats: {result.stats}")

    if args.output:
        output = {
            "question": result.question,
            "display": result.display,
            "timings": result.timings,
        }
        if result.certificate:
            output["certificate"] = result.certificate.model_dump()
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Results saved to {args.output}")


def cmd_eval(args):
    """Run evaluation benchmark."""
    from certirag.pipeline import CertiRAGPipeline
    from eval.runner import EvalRunner

    config = get_config(args.config)
    pipeline = CertiRAGPipeline(config)
    runner = EvalRunner(pipeline, output_dir=args.output_dir)

    if args.benchmark == "alce":
        result = runner.evaluate_alce(args.data_dir, subset=args.subset or "asqa",
                                       max_examples=args.max_examples)
    elif args.benchmark == "ragtruth":
        result = runner.evaluate_ragtruth(args.data_dir, task=args.subset,
                                          max_examples=args.max_examples)
    elif args.benchmark == "aggrefact":
        result = runner.evaluate_aggrefact(args.data_dir, subset=args.subset,
                                           max_examples=args.max_examples)

    print(f"\n{'='*60}")
    print(f"Benchmark: {result.benchmark}/{result.subset}")
    print(f"Examples: {result.num_examples}")
    print(f"Accuracy: {result.accuracy:.1%}")
    print(f"Entailed F1: {result.faithfulness.get('entailed_f1', 0):.1%}")
    print(f"AUROC: {result.auroc:.3f}")
    print(f"ECE: {result.ece:.3f}")
    print(f"{'='*60}")
    print(f"\nLaTeX row:\n{result.to_latex_row()}")


def cmd_validate(args):
    """Validate a JSON file against CertiRAG schemas."""
    from certirag.claim_ir.validator import (
        validate_certificate,
        validate_claim_ir,
        validate_evidence_chunk,
        validate_verification_result,
    )

    with open(args.input) as f:
        data = json.load(f)

    validators = {
        "evidence": validate_evidence_chunk,
        "claim_ir": validate_claim_ir,
        "verification": validate_verification_result,
        "certificate": validate_certificate,
    }

    validator = validators.get(args.schema)
    if not validator:
        print(f"Unknown schema: {args.schema}")
        sys.exit(1)

    errors = validator(data)
    if errors:
        print(f"Validation FAILED: {len(errors)} errors")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("Validation PASSED ✅")


def cmd_export_schemas(args):
    """Export JSON schemas for all data contracts."""
    from certirag.claim_ir.validator import export_all_schemas

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    schemas = export_all_schemas()
    for name, schema in schemas.items():
        path = output_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(schema, f, indent=2)
        print(f"Exported: {path}")

    print(f"\n{len(schemas)} schemas exported to {output_dir}/")


if __name__ == "__main__":
    main()
