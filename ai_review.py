#!/usr/bin/env python
"""
ai_review.py — Gemini-powered code-review orchestrator
Copyright (c) 2025  Your Company
License: Apache-20

This module receives a patch (diff) plus optional static-analyzer findings,
builds a structured prompt, calls the Gemini API, validates the JSON response,
and returns a list of ReviewItem objects that can be posted to GitHub / GitLab.

Dependencies
------------
pip install google-generativeai pydantic backoff unidiff pylint flake8 mypy
"""
# $env:GEMINI_API_KEY = "AIzaSyAS5oiOJsxSUsK5yqsb85AUArcwo1y1JRs"


from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import backoff
import google.generativeai as genai
from pydantic import BaseModel, ValidationError, field_validator

###############################################################################
# 1  Configuration
###############################################################################

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")        # flash | pro | vision
MAX_TOKENS    = int(os.environ.get("GEMINI_MAX_TOKENS", "15360"))        # token budget per call
TEMPERATURE   = float(os.environ.get("GEMINI_TEMPERATURE", "0.2"))
TOP_P         = float(os.environ.get("GEMINI_TOP_P", "0.9"))

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

###############################################################################
# 2  Pydantic schema describing review items
###############################################################################

class ReviewItem(BaseModel):
    """Single review comment emitted by Gemini."""
    severity: str     # BLOCKER | MAJOR | MINOR | INFO
    file: str
    line: int
    explanation: str
    suggestion: str

    @field_validator("severity")
    def normalize_severity(cls, v: str) -> str:
        v = v.upper()
        if v not in {"BLOCKER", "MAJOR", "MINOR", "INFO"}:
            raise ValueError(f"Invalid severity: {v}")
        return v


###############################################################################
# 3  I/O helpers
###############################################################################

def load_json(path: str | Path) -> dict | list:
    """Read a JSON file into Python objects."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_stdin() -> str:
    """Grab everything from stdin (used when piping in a diff)."""
    return sys.stdin.read()


###############################################################################
# 4  Prompt construction
###############################################################################

PROMPT_TEMPLATE = """\
You are an experienced senior software engineer performing a thorough code review.

Context:
• Repository: {repo}
• Commit / PR: {pr_or_sha}

Diff hunks to review:
{diff}

Static-analysis findings already detected (must not be repeated):
{static_findings}

Review rubric:
1. Detect logic errors, race conditions, unhandled exceptions, resource leaks.
2. Identify security vulnerabilities (injection, auth, crypto, etc.).
3. Flag performance or concurrency issues.
4. Point out style / PEP-8 problems ONLY if not already in static_findings.
5. Suggest concrete, actionable improvements or test cases for uncovered paths.

Output ONLY valid JSON array where each element conforms to this schema:
{{
  "severity": "BLOCKER|MAJOR|MINOR|INFO",
  "file": "<relative/path.py>",
  "line": <int>,                    // 1-based target line number
  "explanation": "<why this is an issue>",
  "suggestion": "<how to fix or improve>"
}}

Do NOT wrap the JSON in markdown fences or additional text.
"""

def build_prompt(
    diff_text: str,
    static_json: list | None,
    repo: str,
    pr_or_sha: str,
) -> str:
    """Fill the prompt template with concrete content."""
    findings = json.dumps(static_json, indent=2) if static_json else "[]"
    return PROMPT_TEMPLATE.format(
        repo=repo,
        pr_or_sha=pr_or_sha,
        diff=diff_text[:200_000],          # guardrail against enormous diffs
        static_findings=findings[:50_000], # token safety
    )


###############################################################################
# 5  Gemini invocation with exponential back-off
###############################################################################

model = genai.GenerativeModel(GEMINI_MODEL)

@backoff.on_exception(
    backoff.expo,
    (genai.types.generation_types.RateLimitError, genai.types.generation_types.InternalError),
    max_time=120,
)
def call_gemini(prompt: str) -> str:
    """Hit the Gemini generate_content endpoint and return raw text."""
    response = model.generate_content(
        contents=prompt,
        generation_config={
            "max_output_tokens": 1024,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
        },
    )
    return response.text


###############################################################################
# 6  Validation loop with automatic re-asks
###############################################################################

def validate_or_reask(raw_text: str, prompt: str, depth: int = 0) -> List[ReviewItem]:
    """Ensure Gemini returns valid JSON; recursively retry if malformed."""
    try:
        data = json.loads(raw_text)
        if not isinstance(data, list):
            raise ValueError("Top-level JSON is not a list.")
        return [ReviewItem(**item) for item in data]
    except (json.JSONDecodeError, ValidationError, ValueError) as err:
        if depth >= 2:
            raise RuntimeError(f"Failed after 2 attempts: {err}") from err
        followup = (
            "Your previous reply had formatting errors:\n"
            f"{err}\n\n"
            "Please resend ONLY the corrected JSON array as per schema."
        )
        new_raw = call_gemini(prompt + "\n\n" + followup)
        return validate_or_reask(new_raw, prompt, depth + 1)


###############################################################################
# 7  Main orchestration entry-point
###############################################################################

def review(
    diff_path: str | None = None,
    static_path: str | None = None,
    repo: str | None = "<repo>",
    pr_or_sha: str | None = "<PR-123>",
) -> List[ReviewItem]:
    """
    Run a Gemini review for the supplied diff + static-analysis JSON.
    • diff_path  : file containing unified diff OR None to read from stdin
    • static_path: file containing linter JSON (pylint/flake8/mypy) OR None
    """
    diff_text = (
        Path(diff_path).read_text(encoding="utf-8") if diff_path else read_stdin()
    )
    static_findings = load_json(static_path) if static_path else []
    prompt = build_prompt(diff_text, static_findings, repo, pr_or_sha)
    raw = call_gemini(prompt)
    return validate_or_reask(raw, prompt)


###############################################################################
# 8  CLI wrapper for local testing
###############################################################################

def _cli(argv: List[str]) -> None:
    """Example: python ai_review.py --diff diff.patch --lint pylint.json"""
    import argparse

    ap = argparse.ArgumentParser(description="Gemini-powered code-review bot")
    ap.add_argument("--diff", help="patch file (default: stdin)")
    ap.add_argument(
        "--lint", help="static analyzer JSON (pylint/flake8/mypy consolidated)"
    )
    ap.add_argument("--repo", default="<repo>", help="repository name")
    ap.add_argument("--ref", default="<PR-123>", help="PR number or commit SHA")
    args = ap.parse_args(argv)

    reviews = review(
        diff_path=args.diff,
        static_path=args.lint,
        repo=args.repo,
        pr_or_sha=args.ref,
    )
    print(json.dumps([r.model_dump() for r in reviews], indent=2))


if __name__ == "__main__":
    _cli(sys.argv[1:])
