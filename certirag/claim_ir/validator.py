"""
Claim IR Validator
===================

JSON-schema validation for Claim IR and related data structures.

Ensures that all data flowing through the pipeline conforms to
the spec defined in §6 of the README. This is the enforcement
point for FR-C1 (valid Claim IR JSON).

Usage:
    from certirag.claim_ir.validator import validate_claim_ir
    errors = validate_claim_ir(claim_ir_dict)
    if errors:
        print("Validation failed:", errors)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from certirag.schemas.claim_ir import ClaimIR
from certirag.schemas.evidence import EvidenceChunk
from certirag.schemas.verification import VerificationResult
from certirag.schemas.certificate import AuditCertificate

logger = logging.getLogger("certirag.claim_ir.validator")


def get_json_schema(schema_name: str) -> dict[str, Any]:
    """
    Export the JSON Schema for a CertiRAG data contract.

    Args:
        schema_name: One of "claim_ir", "evidence", "verification", "certificate".

    Returns:
        JSON Schema dict.
    """
    schemas = {
        "claim_ir": ClaimIR,
        "evidence": EvidenceChunk,
        "verification": VerificationResult,
        "certificate": AuditCertificate,
    }
    if schema_name not in schemas:
        raise ValueError(f"Unknown schema: {schema_name}. Use: {list(schemas.keys())}")

    return schemas[schema_name].model_json_schema()


def export_all_schemas(output_dir: str | Path) -> None:
    """
    Export all JSON Schemas to files.

    Creates one .json file per schema in the output directory.
    Useful for:
    - Documentation
    - Cross-language validation
    - Paper appendix

    Args:
        output_dir: Directory to write schema files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in ["claim_ir", "evidence", "verification", "certificate"]:
        schema = get_json_schema(name)
        path = output_dir / f"{name}_schema.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)
        logger.info(f"Exported schema: {path}")


def validate_claim_ir(data: dict[str, Any]) -> list[str]:
    """
    Validate a Claim IR dict against the schema.

    Args:
        data: Dict representation of a ClaimIR.

    Returns:
        List of error messages (empty if valid).
    """
    errors: list[str] = []

    try:
        claim_ir = ClaimIR.model_validate(data)
    except Exception as e:
        errors.append(f"Schema validation failed: {e}")
        return errors

    # Additional semantic checks beyond schema validation:

    # Check claim atomicity (heuristic: no "and" joining clauses)
    for claim in claim_ir.claims:
        if _has_conjunction_clause(claim.text):
            errors.append(
                f"Claim {claim.id} may not be atomic "
                f"(contains conjunction clause): '{claim.text[:60]}...'"
            )

    # Check evidence linkage
    for claim in claim_ir.claims:
        if not claim.has_evidence:
            errors.append(
                f"Claim {claim.id} has no evidence candidates "
                f"and is not flagged as unsupported_by_design"
            )

    # Check duplicate claim IDs
    ids = [c.id for c in claim_ir.claims]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate claim IDs found")

    return errors


def validate_evidence_chunk(data: dict[str, Any]) -> list[str]:
    """Validate an EvidenceChunk dict against the schema."""
    errors: list[str] = []
    try:
        EvidenceChunk.model_validate(data)
    except Exception as e:
        errors.append(f"Schema validation failed: {e}")
    return errors


def validate_verification_result(data: dict[str, Any]) -> list[str]:
    """Validate a VerificationResult dict against the schema."""
    errors: list[str] = []
    try:
        VerificationResult.model_validate(data)
    except Exception as e:
        errors.append(f"Schema validation failed: {e}")
    return errors


def validate_certificate(data: dict[str, Any]) -> list[str]:
    """
    Validate an AuditCertificate dict.

    Checks both schema validity and integrity hash.
    """
    errors: list[str] = []
    try:
        cert = AuditCertificate.model_validate(data)
        if cert.integrity_hash and not cert.verify_integrity():
            errors.append("Certificate integrity hash mismatch (possible tampering)")
    except Exception as e:
        errors.append(f"Schema validation failed: {e}")
    return errors


def _has_conjunction_clause(text: str) -> bool:
    """
    Heuristic check for compound claims with conjunction clauses.

    Detects patterns like:
    - "X, and Y was..."  (independent clause)
    - "X because Y"  (causal compound)
    - "X, which also Y"  (relative clause with additional fact)

    Does NOT flag:
    - "cats and dogs"  (entity list)
    - "red, blue, and green"  (attribute list)
    """
    import re

    # Patterns suggesting compound independent clauses
    compound_patterns = [
        r",\s+and\s+[A-Z]",        # ", and The..." (independent clause)
        r",\s+but\s+[A-Z]",        # ", but The..."
        r"\s+because\s+",           # "X because Y"
        r",\s+which also\s+",      # ", which also..."
        r";\s+",                    # semicolons
    ]

    for pattern in compound_patterns:
        if re.search(pattern, text):
            return True
    return False
