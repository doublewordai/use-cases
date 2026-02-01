"""
Juliet Test Suite loader.

Parses samples from the NIST SARD Juliet Test Suite.
Each test case file contains:
- A *_bad() function with a known vulnerability
- One or more good*() functions with the fix
"""

import json
import re
from pathlib import Path
from typing import Iterator
import click


# CWE mappings
CWE_INFO = {
    "CWE78": ("OS Command Injection", "security"),
    "CWE89": ("SQL Injection", "security"),
    "CWE90": ("LDAP Injection", "security"),
    "CWE113": ("HTTP Response Splitting", "security"),
    "CWE114": ("Process Control", "security"),
    "CWE119": ("Buffer Overflow", "security"),
    "CWE120": ("Buffer Overflow (Classic)", "security"),
    "CWE121": ("Stack-based Buffer Overflow", "security"),
    "CWE122": ("Heap-based Buffer Overflow", "security"),
    "CWE123": ("Write-what-where", "security"),
    "CWE124": ("Buffer Underwrite", "security"),
    "CWE126": ("Buffer Over-read", "security"),
    "CWE127": ("Buffer Under-read", "security"),
    "CWE134": ("Uncontrolled Format String", "security"),
    "CWE176": ("Unicode Encoding", "security"),
    "CWE190": ("Integer Overflow", "types"),
    "CWE191": ("Integer Underflow", "types"),
    "CWE194": ("Unexpected Sign Extension", "types"),
    "CWE195": ("Signed to Unsigned Conversion", "types"),
    "CWE196": ("Unsigned to Signed Conversion", "types"),
    "CWE197": ("Numeric Truncation Error", "types"),
    "CWE252": ("Unchecked Return Value", "error_handling"),
    "CWE253": ("Incorrect Check of Return Value", "error_handling"),
    "CWE256": ("Plaintext Storage of Password", "security"),
    "CWE259": ("Hard-coded Password", "security"),
    "CWE319": ("Cleartext Transmission", "security"),
    "CWE321": ("Hard-coded Cryptographic Key", "security"),
    "CWE327": ("Broken Crypto Algorithm", "security"),
    "CWE328": ("Reversible One-Way Hash", "security"),
    "CWE338": ("Weak PRNG", "security"),
    "CWE364": ("Signal Handler Race", "concurrency"),
    "CWE366": ("Race Condition in Thread", "concurrency"),
    "CWE367": ("TOCTOU Race Condition", "concurrency"),
    "CWE369": ("Divide By Zero", "logic"),
    "CWE377": ("Insecure Temporary File", "security"),
    "CWE390": ("Detection of Error without Action", "error_handling"),
    "CWE391": ("Unchecked Error Condition", "error_handling"),
    "CWE396": ("Catch Generic Exception", "error_handling"),
    "CWE397": ("Throw Generic Exception", "error_handling"),
    "CWE398": ("Code Quality", "logic"),
    "CWE400": ("Resource Exhaustion", "error_handling"),
    "CWE401": ("Memory Leak", "error_handling"),
    "CWE404": ("Improper Resource Shutdown", "error_handling"),
    "CWE415": ("Double Free", "security"),
    "CWE416": ("Use After Free", "security"),
    "CWE426": ("Untrusted Search Path", "security"),
    "CWE427": ("Uncontrolled Search Path", "security"),
    "CWE457": ("Use of Uninitialized Variable", "logic"),
    "CWE459": ("Incomplete Cleanup", "error_handling"),
    "CWE467": ("sizeof() on Pointer Type", "logic"),
    "CWE468": ("Incorrect Pointer Scaling", "logic"),
    "CWE469": ("Use of Pointer Subtraction", "logic"),
    "CWE475": ("Undefined Behavior for Input", "logic"),
    "CWE476": ("NULL Pointer Dereference", "types"),
    "CWE478": ("Missing Default Case in Switch", "logic"),
    "CWE479": ("Signal Handler Use of Non-Reentrant Function", "concurrency"),
    "CWE480": ("Use of Incorrect Operator", "logic"),
    "CWE481": ("Assigning instead of Comparing", "logic"),
    "CWE482": ("Comparing instead of Assigning", "logic"),
    "CWE483": ("Incorrect Block Delimitation", "logic"),
    "CWE484": ("Omitted Break Statement in Switch", "logic"),
    "CWE506": ("Embedded Malicious Code", "security"),
    "CWE510": ("Trapdoor", "security"),
    "CWE511": ("Logic/Time Bomb", "security"),
    "CWE526": ("Info Exposure Through Env Variables", "security"),
    "CWE534": ("Info Exposure Through Debug Log", "security"),
    "CWE535": ("Info Exposure Through Shell Error", "security"),
    "CWE546": ("Suspicious Comment", "logic"),
    "CWE561": ("Dead Code", "logic"),
    "CWE562": ("Return of Stack Variable Address", "security"),
    "CWE563": ("Unused Variable", "logic"),
    "CWE564": ("SQL Injection (Hibernate)", "security"),
    "CWE566": ("Auth Bypass Through SQL Primary", "security"),
    "CWE570": ("Expression is Always False", "logic"),
    "CWE571": ("Expression is Always True", "logic"),
    "CWE587": ("Assignment of Fixed Address", "logic"),
    "CWE588": ("Attempt to Access Child of Non-structure Pointer", "types"),
    "CWE590": ("Free of Memory not on the Heap", "security"),
    "CWE591": ("Sensitive Data Storage in Improperly Locked Memory", "security"),
    "CWE605": ("Multiple Binds to Same Port", "error_handling"),
    "CWE606": ("Unchecked Input for Loop Condition", "logic"),
    "CWE615": ("Info Exposure Through Comments", "security"),
    "CWE617": ("Reachable Assertion", "logic"),
    "CWE620": ("Unverified Password Change", "security"),
    "CWE665": ("Improper Initialization", "logic"),
    "CWE666": ("Operation on Resource in Wrong Phase", "error_handling"),
    "CWE667": ("Improper Locking", "concurrency"),
    "CWE672": ("Operation on Resource After Expiration", "error_handling"),
    "CWE675": ("Duplicate Operations on Resource", "error_handling"),
    "CWE676": ("Use of Potentially Dangerous Function", "security"),
    "CWE680": ("Integer Overflow to Buffer Overflow", "security"),
    "CWE681": ("Incorrect Conversion between Numeric Types", "types"),
    "CWE685": ("Function Call With Incorrect Number of Arguments", "logic"),
    "CWE688": ("Function Call With Incorrect Variable or Reference", "logic"),
    "CWE690": ("NULL Deref From Return", "types"),
    "CWE761": ("Free of Pointer not at Start of Buffer", "security"),
    "CWE762": ("Mismatched Memory Management", "security"),
    "CWE773": ("Missing Reference to File Descriptor", "error_handling"),
    "CWE775": ("Missing Release of File Descriptor", "error_handling"),
    "CWE780": ("Use of RSA Without OAEP", "security"),
    "CWE785": ("Use of Path Without Length Limit", "security"),
    "CWE789": ("Uncontrolled Memory Allocation", "security"),
    "CWE832": ("Unlock of Resource That is Not Locked", "concurrency"),
    "CWE833": ("Deadlock", "concurrency"),
    "CWE835": ("Infinite Loop", "logic"),
    "CWE843": ("Type Confusion", "types"),
}


def extract_bad_function(content: str) -> str | None:
    """Extract the bad() function from a Juliet test case."""
    # Find the bad function - it's between #ifndef OMITBAD and #endif
    bad_match = re.search(
        r'#ifndef OMITBAD\s*(.*?)#endif\s*/\* OMITBAD \*/',
        content,
        re.DOTALL
    )
    if bad_match:
        return bad_match.group(1).strip()

    # Alternative: find function ending in _bad()
    func_match = re.search(
        r'(void\s+\w+_bad\s*\([^)]*\)\s*\{.*?\n\})',
        content,
        re.DOTALL
    )
    if func_match:
        return func_match.group(1)

    return None


def extract_good_function(content: str) -> str | None:
    """Extract the first good() function from a Juliet test case."""
    # Find between #ifndef OMITGOOD and first static void good
    good_section = re.search(
        r'#ifndef OMITGOOD\s*(.*?)#endif\s*/\* OMITGOOD \*/',
        content,
        re.DOTALL
    )
    if good_section:
        # Get first good function
        func_match = re.search(
            r'(static\s+void\s+good\w*\s*\([^)]*\)\s*\{.*?\n\})',
            good_section.group(1),
            re.DOTALL
        )
        if func_match:
            return func_match.group(1)

    return None


def get_cwe_from_path(filepath: Path) -> str | None:
    """Extract CWE ID from file path."""
    path_str = str(filepath)
    match = re.search(r'CWE(\d+)', path_str)
    if match:
        return f"CWE{match.group(1)}"
    return None


def load_juliet_samples(
    juliet_path: Path,
    cwes: list[str] | None = None,
    max_per_cwe: int = 10,
    include_good: bool = True,
) -> list[dict]:
    """
    Load samples from extracted SARD Juliet directory.

    Args:
        juliet_path: Path to extracted Juliet directory (containing numbered folders)
        cwes: List of CWE IDs to include (e.g., ["CWE89", "CWE120"])
        max_per_cwe: Maximum samples per CWE
        include_good: Whether to include "good" (fixed) samples
    """
    samples = []
    cwe_counts = {}

    # Find all test case files
    for c_file in juliet_path.rglob("testcases/**/*.c"):
        # Skip support files
        if "testcasesupport" in str(c_file):
            continue

        cwe = get_cwe_from_path(c_file)
        if not cwe:
            continue

        if cwes and cwe not in cwes:
            continue

        # Track counts per CWE
        if cwe not in cwe_counts:
            cwe_counts[cwe] = 0

        if cwe_counts[cwe] >= max_per_cwe:
            continue

        try:
            content = c_file.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue

        # Extract bad function
        bad_func = extract_bad_function(content)
        if bad_func and len(bad_func) > 50:  # Skip trivial functions
            cwe_name, bug_type = CWE_INFO.get(cwe, (cwe, "security"))
            samples.append({
                "id": f"{c_file.stem}_bad",
                "code": bad_func,
                "language": "c",
                "has_bug": True,
                "bug_type": bug_type,
                "bug_description": cwe_name,
                "cwe": cwe,
                "source": "juliet",
                "source_file": c_file.name,
            })
            cwe_counts[cwe] += 1

        # Optionally extract good function
        if include_good and cwe_counts[cwe] < max_per_cwe:
            good_func = extract_good_function(content)
            if good_func and len(good_func) > 50:
                samples.append({
                    "id": f"{c_file.stem}_good",
                    "code": good_func,
                    "language": "c",
                    "has_bug": False,
                    "bug_type": None,
                    "bug_description": None,
                    "cwe": None,
                    "source": "juliet",
                    "source_file": c_file.name,
                })

    return samples


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        juliet_path = Path(sys.argv[1])
    else:
        juliet_path = Path("data/juliet")

    samples = load_juliet_samples(
        juliet_path,
        cwes=["CWE89", "CWE120", "CWE190", "CWE476", "CWE78"],
        max_per_cwe=5
    )

    print(f"Loaded {len(samples)} samples")
    buggy = sum(1 for s in samples if s["has_bug"])
    print(f"  Buggy: {buggy}, Clean: {len(samples) - buggy}")

    for s in samples[:10]:
        status = "BUG" if s["has_bug"] else "CLEAN"
        print(f"  [{status}] {s['id']}: {s.get('cwe', 'N/A')}")
