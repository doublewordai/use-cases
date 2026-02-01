"""
20 different prompts for bug detection.

Each prompt approaches code review from a different angle:
- Different bug categories (security, logic, performance, style)
- Different personas (security auditor, senior dev, QA engineer)
- Different framing (find bugs vs verify correctness)

Prompts are organized into subsets for cost-performance analysis.
"""

# Prompt subsets for ablation analysis
# Ordered from smallest (cheapest) to largest (most expensive)
PROMPT_SUBSETS = {
    # Core security prompts only - minimum viable ensemble
    "security_only": [
        "security_auditor",
        "owasp_checker",
        "input_validation",
    ],

    # Security + logic - good coverage at moderate cost
    "security_logic": [
        "security_auditor",
        "owasp_checker",
        "input_validation",
        "logic_reviewer",
        "edge_case_hunter",
        "correctness_prover",
        "type_checker",
    ],

    # Core 10 - most effective prompts based on empirical results
    "core": [
        "security_auditor",
        "owasp_checker",
        "input_validation",
        "logic_reviewer",
        "edge_case_hunter",
        "correctness_prover",
        "type_checker",
        "numeric_overflow_checker",
        "resource_leak_detector",
        "error_handling_reviewer",
    ],

    # Expanded 15 - adds persona-based and specialized prompts
    "expanded": [
        "security_auditor",
        "owasp_checker",
        "input_validation",
        "logic_reviewer",
        "edge_case_hunter",
        "correctness_prover",
        "type_checker",
        "numeric_overflow_checker",
        "resource_leak_detector",
        "error_handling_reviewer",
        "concurrency_reviewer",
        "state_machine_checker",
        "senior_dev",
        "paranoid_reviewer",
        "devils_advocate",
    ],

    # Full ensemble - all 20 prompts
    "full": [
        "security_auditor",
        "owasp_checker",
        "input_validation",
        "logic_reviewer",
        "edge_case_hunter",
        "correctness_prover",
        "state_machine_checker",
        "error_handling_reviewer",
        "resource_leak_detector",
        "type_checker",
        "numeric_overflow_checker",
        "concurrency_reviewer",
        "senior_dev",
        "qa_engineer",
        "paranoid_reviewer",
        "devils_advocate",
        "bug_or_not",
        "production_ready",
        "incident_predictor",
        "formal_verifier",
    ],
}


REVIEW_PROMPTS = [
    # Security-focused
    {
        "id": "security_auditor",
        "category": "security",
        "prompt": """You are a security auditor reviewing code for vulnerabilities.
Analyze this code for security issues including but not limited to:
- Injection vulnerabilities (SQL, command, XSS)
- Buffer overflows
- Authentication/authorization flaws
- Cryptographic weaknesses

Code:
```
{code}
```

Respond with JSON: {{"has_bug": true/false, "bugs": [{{"type": "...", "description": "...", "line": N}}]}}"""
    },
    {
        "id": "owasp_checker",
        "category": "security",
        "prompt": """Review this code against the OWASP Top 10 vulnerabilities.
Check for: injection, broken auth, sensitive data exposure, XXE, broken access control,
security misconfiguration, XSS, insecure deserialization, vulnerable components, insufficient logging.

Code:
```
{code}
```

Respond with JSON: {{"has_bug": true/false, "bugs": [{{"type": "...", "description": "...", "line": N}}]}}"""
    },
    {
        "id": "input_validation",
        "category": "security",
        "prompt": """Focus specifically on input validation and sanitization.
Does this code properly validate all inputs? Are there any paths where untrusted data
could reach sensitive operations without validation?

Code:
```
{code}
```

Respond with JSON: {{"has_bug": true/false, "bugs": [{{"type": "...", "description": "...", "line": N}}]}}"""
    },

    # Logic-focused
    {
        "id": "logic_reviewer",
        "category": "logic",
        "prompt": """You are reviewing code for logical errors.
Look for: off-by-one errors, incorrect conditions, wrong operators,
missing edge cases, infinite loops, incorrect algorithm implementation.

Code:
```
{code}
```

Respond with JSON: {{"has_bug": true/false, "bugs": [{{"type": "...", "description": "...", "line": N}}]}}"""
    },
    {
        "id": "edge_case_hunter",
        "category": "logic",
        "prompt": """Hunt for edge case bugs. Consider:
- Empty inputs, null values, zero values
- Boundary conditions (min/max values)
- Unicode and special characters
- Concurrent access scenarios

Code:
```
{code}
```

Respond with JSON: {{"has_bug": true/false, "bugs": [{{"type": "...", "description": "...", "line": N}}]}}"""
    },
    {
        "id": "correctness_prover",
        "category": "logic",
        "prompt": """Try to prove this code is correct. If you find a counterexample
(an input that produces wrong output), report it as a bug.

Code:
```
{code}
```

Respond with JSON: {{"has_bug": true/false, "bugs": [{{"type": "...", "description": "...", "line": N}}]}}"""
    },
    {
        "id": "state_machine_checker",
        "category": "logic",
        "prompt": """Analyze the state transitions in this code.
Are there invalid state transitions? Can the system get stuck?
Are all states reachable and all exits handled?

Code:
```
{code}
```

Respond with JSON: {{"has_bug": true/false, "bugs": [{{"type": "...", "description": "...", "line": N}}]}}"""
    },

    # Error handling
    {
        "id": "error_handling_reviewer",
        "category": "error_handling",
        "prompt": """Review error handling in this code.
- Are all errors caught and handled appropriately?
- Are there silent failures?
- Could exceptions propagate unexpectedly?
- Are resources properly cleaned up on error?

Code:
```
{code}
```

Respond with JSON: {{"has_bug": true/false, "bugs": [{{"type": "...", "description": "...", "line": N}}]}}"""
    },
    {
        "id": "resource_leak_detector",
        "category": "error_handling",
        "prompt": """Look for resource leaks:
- File handles not closed
- Memory not freed
- Database connections not released
- Locks not released

Code:
```
{code}
```

Respond with JSON: {{"has_bug": true/false, "bugs": [{{"type": "...", "description": "...", "line": N}}]}}"""
    },

    # Type safety
    {
        "id": "type_checker",
        "category": "types",
        "prompt": """Review for type-related bugs:
- Type mismatches
- Incorrect casts
- Null pointer dereferences
- Array type confusion

Code:
```
{code}
```

Respond with JSON: {{"has_bug": true/false, "bugs": [{{"type": "...", "description": "...", "line": N}}]}}"""
    },
    {
        "id": "numeric_overflow_checker",
        "category": "types",
        "prompt": """Check for numeric issues:
- Integer overflow/underflow
- Division by zero
- Floating point precision errors
- Sign errors

Code:
```
{code}
```

Respond with JSON: {{"has_bug": true/false, "bugs": [{{"type": "...", "description": "...", "line": N}}]}}"""
    },

    # Concurrency
    {
        "id": "concurrency_reviewer",
        "category": "concurrency",
        "prompt": """Review for concurrency bugs:
- Race conditions
- Deadlocks
- Thread safety violations
- Atomicity violations

Code:
```
{code}
```

Respond with JSON: {{"has_bug": true/false, "bugs": [{{"type": "...", "description": "...", "line": N}}]}}"""
    },

    # Persona-based
    {
        "id": "senior_dev",
        "category": "general",
        "prompt": """You are a senior developer with 20 years of experience.
Review this code as if a junior developer submitted it for code review.
What bugs would you flag before approving?

Code:
```
{code}
```

Respond with JSON: {{"has_bug": true/false, "bugs": [{{"type": "...", "description": "...", "line": N}}]}}"""
    },
    {
        "id": "qa_engineer",
        "category": "general",
        "prompt": """You are a QA engineer writing test cases.
What bugs would you expect to find when testing this code?
Think about what tests would fail.

Code:
```
{code}
```

Respond with JSON: {{"has_bug": true/false, "bugs": [{{"type": "...", "description": "...", "line": N}}]}}"""
    },
    {
        "id": "paranoid_reviewer",
        "category": "general",
        "prompt": """Be extremely paranoid. Assume the worst about all inputs
and external dependencies. What could possibly go wrong with this code?

Code:
```
{code}
```

Respond with JSON: {{"has_bug": true/false, "bugs": [{{"type": "...", "description": "...", "line": N}}]}}"""
    },
    {
        "id": "devils_advocate",
        "category": "general",
        "prompt": """Play devil's advocate. Try your hardest to break this code.
What inputs or scenarios would cause it to fail?

Code:
```
{code}
```

Respond with JSON: {{"has_bug": true/false, "bugs": [{{"type": "...", "description": "...", "line": N}}]}}"""
    },

    # Different framing
    {
        "id": "bug_or_not",
        "category": "general",
        "prompt": """Simple question: Does this code contain any bugs?
Consider functionality, edge cases, error handling, and security.

Code:
```
{code}
```

Respond with JSON: {{"has_bug": true/false, "bugs": [{{"type": "...", "description": "...", "line": N}}]}}"""
    },
    {
        "id": "production_ready",
        "category": "general",
        "prompt": """Is this code production-ready? What bugs need to be fixed
before deploying to production?

Code:
```
{code}
```

Respond with JSON: {{"has_bug": true/false, "bugs": [{{"type": "...", "description": "...", "line": N}}]}}"""
    },
    {
        "id": "incident_predictor",
        "category": "general",
        "prompt": """If this code were deployed, what production incidents might it cause?
Think about: crashes, data corruption, security breaches, performance degradation.

Code:
```
{code}
```

Respond with JSON: {{"has_bug": true/false, "bugs": [{{"type": "...", "description": "...", "line": N}}]}}"""
    },
    {
        "id": "formal_verifier",
        "category": "general",
        "prompt": """Apply formal reasoning to verify this code.
State the preconditions, postconditions, and invariants.
Does the code satisfy them? If not, what's the bug?

Code:
```
{code}
```

Respond with JSON: {{"has_bug": true/false, "bugs": [{{"type": "...", "description": "...", "line": N}}]}}"""
    },
]


def get_prompt(prompt_id: str) -> dict:
    """Get a prompt by ID."""
    for p in REVIEW_PROMPTS:
        if p["id"] == prompt_id:
            return p
    raise ValueError(f"Unknown prompt: {prompt_id}")


def get_prompts_by_category(category: str) -> list[dict]:
    """Get all prompts in a category."""
    return [p for p in REVIEW_PROMPTS if p["category"] == category]


def format_prompt(prompt: dict, code: str) -> str:
    """Format a prompt with code."""
    return prompt["prompt"].format(code=code)
