"""
Real-world vulnerability dataset loaders.

Supports:
- CVEfixes: 138K+ functions with before/after patches from real CVEs
- Juliet: NIST synthetic test suite (existing loader in juliet.py)
"""

import json
import sqlite3
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterator

import click

from .preprocess import preprocess_code


# CVEfixes database schema reference:
# - cve: CVE ID, published_date, cwe_id, etc.
# - commits: commit info linked to CVE
# - file_change: files changed in commits
# - method_change: functions changed with before/after code


CVEFIXES_ZENODO_URL = "https://zenodo.org/record/7029359/files/CVEfixes_v1.0.7.zip"
CVEFIXES_DB_NAME = "CVEfixes.db"


def download_cvefixes(output_dir: str, progress: bool = True) -> Path:
    """
    Download CVEfixes database from Zenodo.

    Args:
        output_dir: Directory to save the database
        progress: Show download progress

    Returns:
        Path to the SQLite database file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    db_path = output_path / CVEFIXES_DB_NAME
    if db_path.exists():
        click.echo(f"Database already exists: {db_path}")
        return db_path

    zip_path = output_path / "CVEfixes.zip"

    click.echo(f"Downloading CVEfixes from Zenodo...")
    click.echo(f"URL: {CVEFIXES_ZENODO_URL}")
    click.echo("(This is ~2GB, may take a while)")

    def report_hook(block_num, block_size, total_size):
        if progress and total_size > 0:
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 // total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            click.echo(f"\r  {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", nl=False)

    urllib.request.urlretrieve(CVEFIXES_ZENODO_URL, zip_path, report_hook)
    click.echo()  # Newline after progress

    click.echo("Extracting database...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Find the .db file in the archive
        for name in zf.namelist():
            if name.endswith('.db'):
                # Extract to output_dir
                zf.extract(name, output_path)
                extracted_path = output_path / name
                if extracted_path != db_path:
                    extracted_path.rename(db_path)
                break

    # Clean up zip
    zip_path.unlink()
    click.echo(f"Database saved to: {db_path}")

    return db_path


def load_cvefixes(
    db_path: str,
    max_samples: int = 1000,
    cwes: list[str] | None = None,
    languages: list[str] | None = None,
    min_code_length: int = 50,
    max_code_length: int = 5000,
    strip_comments: bool = True,
    balanced: bool = True,
) -> list[dict]:
    """
    Load samples from CVEfixes SQLite database.

    Each sample includes:
    - Vulnerable version (before fix) - has_bug=True
    - Patched version (after fix) - has_bug=False

    The CVEfixes schema stores methods with a `before_change` flag:
    - before_change='True': vulnerable code (before the fix)
    - before_change='False': patched code (after the fix)

    Args:
        db_path: Path to CVEfixes.db
        max_samples: Maximum total samples to return
        cwes: Filter by CWE IDs (e.g., ["CWE-79", "CWE-89"])
        languages: Filter by language (e.g., ["c", "cpp"])
        min_code_length: Minimum code length in characters
        max_code_length: Maximum code length in characters
        strip_comments: Remove comments from code
        balanced: Try to balance buggy/clean samples

    Returns:
        List of sample dicts with id, code, language, has_bug, cwe, cve_id
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Default to C/C++ for security-relevant code
    if languages is None:
        languages = ["c", "cpp", "c++"]

    # Build query to get paired before/after methods
    # Join method_change with file_change, then link to CVE via fixes table
    query = """
    SELECT
        m_before.method_change_id as before_id,
        m_after.method_change_id as after_id,
        m_before.name as method_name,
        m_before.code as code_before,
        m_after.code as code_after,
        f.programming_language,
        fixes.cve_id,
        cwe_class.cwe_id
    FROM method_change m_before
    JOIN method_change m_after
        ON m_before.file_change_id = m_after.file_change_id
        AND m_before.name = m_after.name
        AND m_before.before_change = 'True'
        AND m_after.before_change = 'False'
    JOIN file_change f ON m_before.file_change_id = f.file_change_id
    JOIN commits c ON f.hash = c.hash
    JOIN fixes ON c.hash = fixes.hash
    LEFT JOIN cwe_classification cwe_class ON fixes.cve_id = cwe_class.cve_id
    WHERE m_before.code IS NOT NULL
      AND m_after.code IS NOT NULL
      AND LENGTH(m_before.code) >= ?
      AND LENGTH(m_before.code) <= ?
      AND LENGTH(m_after.code) >= ?
      AND LENGTH(m_after.code) <= ?
    """

    params = [min_code_length, max_code_length, min_code_length, max_code_length]

    if languages:
        placeholders = ','.join('?' * len(languages))
        query += f" AND LOWER(f.programming_language) IN ({placeholders})"
        params.extend([lang.lower() for lang in languages])

    if cwes:
        placeholders = ','.join('?' * len(cwes))
        query += f" AND cwe_class.cwe_id IN ({placeholders})"
        params.extend(cwes)

    # Limit to avoid loading too much
    query += " LIMIT ?"
    params.append(max_samples * 2)  # Get more since we'll pair before/after

    cursor = conn.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    samples = []
    cwe_counts = {}

    for row in rows:
        method_name = row['method_name'] or f"func_{row['before_id']}"
        code_before = row['code_before']
        code_after = row['code_after']
        language = (row['programming_language'] or 'c').lower()
        cve_id = row['cve_id']
        cwe_id = row['cwe_id']

        # Normalize language
        if language in ('c++', 'cpp'):
            language = 'cpp'

        # Skip if before and after are identical
        if code_before.strip() == code_after.strip():
            continue

        # Preprocess code
        if strip_comments:
            code_before = preprocess_code(code_before, strip_comments=True)
            code_after = preprocess_code(code_after, strip_comments=True)

        # Track CWE distribution
        if cwe_id not in cwe_counts:
            cwe_counts[cwe_id] = 0
        cwe_counts[cwe_id] += 1

        # Add vulnerable version
        samples.append({
            "id": f"{cve_id}_{method_name}_vulnerable",
            "code": code_before,
            "language": language,
            "has_bug": True,
            "bug_type": "security",
            "bug_description": f"Vulnerability fixed in {cve_id}",
            "cwe": cwe_id,
            "cve_id": cve_id,
            "source": "cvefixes",
        })

        # Add patched version
        samples.append({
            "id": f"{cve_id}_{method_name}_patched",
            "code": code_after,
            "language": language,
            "has_bug": False,
            "bug_type": None,
            "bug_description": None,
            "cwe": None,
            "cve_id": cve_id,
            "source": "cvefixes",
        })

        if len(samples) >= max_samples:
            break

    # Balance if requested
    if balanced:
        buggy = [s for s in samples if s['has_bug']]
        clean = [s for s in samples if not s['has_bug']]
        min_count = min(len(buggy), len(clean), max_samples // 2)
        samples = buggy[:min_count] + clean[:min_count]

    return samples


def get_cvefixes_stats(db_path: str) -> dict:
    """
    Get statistics about the CVEfixes database.

    Returns:
        Dict with counts of CVEs, commits, methods, CWE distribution, etc.
    """
    conn = sqlite3.connect(db_path)

    stats = {}

    # Count tables
    stats['cve_count'] = conn.execute("SELECT COUNT(*) FROM cve").fetchone()[0]
    stats['commit_count'] = conn.execute("SELECT COUNT(*) FROM commits").fetchone()[0]
    stats['method_count'] = conn.execute("SELECT COUNT(*) FROM method_change").fetchone()[0]

    # Count paired methods (before/after)
    paired_query = """
    SELECT COUNT(*) FROM (
        SELECT DISTINCT m_before.file_change_id, m_before.name
        FROM method_change m_before
        JOIN method_change m_after
            ON m_before.file_change_id = m_after.file_change_id
            AND m_before.name = m_after.name
            AND m_before.before_change = 'True'
            AND m_after.before_change = 'False'
    )
    """
    stats['paired_methods'] = conn.execute(paired_query).fetchone()[0]

    # Top CWEs from cwe_classification table
    cwe_query = """
    SELECT cwe_id, COUNT(*) as count
    FROM cwe_classification
    WHERE cwe_id IS NOT NULL
    GROUP BY cwe_id
    ORDER BY count DESC
    LIMIT 20
    """
    stats['top_cwes'] = [(row[0], row[1]) for row in conn.execute(cwe_query)]

    # Language distribution
    lang_query = """
    SELECT LOWER(programming_language) as lang, COUNT(*) as count
    FROM file_change
    WHERE programming_language IS NOT NULL
    GROUP BY lang
    ORDER BY count DESC
    LIMIT 10
    """
    stats['languages'] = [(row[0], row[1]) for row in conn.execute(lang_query)]

    conn.close()
    return stats


def iter_cvefixes_samples(
    db_path: str,
    batch_size: int = 100,
    **kwargs
) -> Iterator[list[dict]]:
    """
    Iterate over CVEfixes samples in batches.

    Useful for processing large datasets without loading everything into memory.

    Args:
        db_path: Path to database
        batch_size: Number of samples per batch
        **kwargs: Passed to load_cvefixes

    Yields:
        Batches of sample dicts
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    offset = 0
    while True:
        samples = load_cvefixes(
            db_path,
            max_samples=batch_size,
            **kwargs
        )
        if not samples:
            break
        yield samples
        offset += batch_size


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        db_path = "data/CVEfixes.db"

    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        print("Run: bug-ensemble fetch-cvefixes to download")
        sys.exit(1)

    print("=== CVEfixes Statistics ===")
    stats = get_cvefixes_stats(db_path)
    print(f"CVEs: {stats['cve_count']:,}")
    print(f"Commits: {stats['commit_count']:,}")
    print(f"Methods: {stats['method_count']:,}")

    print("\nTop CWEs:")
    for cwe, count in stats['top_cwes'][:10]:
        print(f"  {cwe}: {count:,}")

    print("\nLanguages:")
    for lang, count in stats['languages']:
        print(f"  {lang}: {count:,}")

    print("\n=== Sample Load Test ===")
    samples = load_cvefixes(db_path, max_samples=20, strip_comments=True)
    print(f"Loaded {len(samples)} samples")

    buggy = sum(1 for s in samples if s['has_bug'])
    print(f"  Buggy: {buggy}, Clean: {len(samples) - buggy}")

    for s in samples[:5]:
        status = "BUG" if s['has_bug'] else "CLEAN"
        print(f"  [{status}] {s['id'][:50]}... ({s.get('cwe', 'N/A')})")
