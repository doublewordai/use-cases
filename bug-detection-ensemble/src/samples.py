"""
Sample code with known bugs for testing the ensemble.

Each sample includes:
- id: Unique identifier
- code: The code snippet
- language: Programming language
- has_bug: Ground truth
- bug_type: Category of bug (if has_bug)
- bug_description: What the bug is (if has_bug)
- cwe: CWE ID if applicable
"""

SAMPLES = [
    # SQL Injection (CWE-89)
    {
        "id": "sql_injection_1",
        "code": '''def get_user(username):
    query = f"SELECT * FROM users WHERE username = '{username}'"
    return db.execute(query)''',
        "language": "python",
        "has_bug": True,
        "bug_type": "security",
        "bug_description": "SQL injection via string formatting",
        "cwe": "CWE-89"
    },

    # Command Injection (CWE-78)
    {
        "id": "command_injection_1",
        "code": '''import os

def ping_host(hostname):
    os.system(f"ping -c 1 {hostname}")''',
        "language": "python",
        "has_bug": True,
        "bug_type": "security",
        "bug_description": "Command injection via os.system",
        "cwe": "CWE-78"
    },

    # XSS (CWE-79)
    {
        "id": "xss_1",
        "code": '''function displayMessage(msg) {
    document.getElementById('output').innerHTML = msg;
}''',
        "language": "javascript",
        "has_bug": True,
        "bug_type": "security",
        "bug_description": "XSS via innerHTML without sanitization",
        "cwe": "CWE-79"
    },

    # Path Traversal (CWE-22)
    {
        "id": "path_traversal_1",
        "code": '''def read_file(filename):
    with open(f"/data/{filename}", "r") as f:
        return f.read()''',
        "language": "python",
        "has_bug": True,
        "bug_type": "security",
        "bug_description": "Path traversal via unsanitized filename",
        "cwe": "CWE-22"
    },

    # Off-by-one error (CWE-193)
    {
        "id": "off_by_one_1",
        "code": '''def sum_array(arr):
    total = 0
    for i in range(1, len(arr)):
        total += arr[i]
    return total''',
        "language": "python",
        "has_bug": True,
        "bug_type": "logic",
        "bug_description": "Off-by-one: skips first element",
        "cwe": "CWE-193"
    },

    # Integer overflow (CWE-190)
    {
        "id": "integer_overflow_1",
        "code": '''int calculate_size(int width, int height) {
    return width * height * 4;  // RGBA
}''',
        "language": "c",
        "has_bug": True,
        "bug_type": "types",
        "bug_description": "Integer overflow on multiplication",
        "cwe": "CWE-190"
    },

    # Null pointer dereference (CWE-476)
    {
        "id": "null_deref_1",
        "code": '''public String getUserName(User user) {
    return user.getName().toUpperCase();
}''',
        "language": "java",
        "has_bug": True,
        "bug_type": "types",
        "bug_description": "Null pointer dereference if user or name is null",
        "cwe": "CWE-476"
    },

    # Resource leak (CWE-404)
    {
        "id": "resource_leak_1",
        "code": '''def process_file(path):
    f = open(path, 'r')
    data = f.read()
    return process(data)''',
        "language": "python",
        "has_bug": True,
        "bug_type": "error_handling",
        "bug_description": "File handle never closed",
        "cwe": "CWE-404"
    },

    # Race condition (CWE-362)
    {
        "id": "race_condition_1",
        "code": '''class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1''',
        "language": "python",
        "has_bug": True,
        "bug_type": "concurrency",
        "bug_description": "Race condition: increment not atomic",
        "cwe": "CWE-362"
    },

    # Division by zero (CWE-369)
    {
        "id": "division_by_zero_1",
        "code": '''def calculate_average(numbers):
    return sum(numbers) / len(numbers)''',
        "language": "python",
        "has_bug": True,
        "bug_type": "logic",
        "bug_description": "Division by zero when list is empty",
        "cwe": "CWE-369"
    },

    # Hard-coded credentials (CWE-798)
    {
        "id": "hardcoded_creds_1",
        "code": '''def connect_to_db():
    return mysql.connect(
        host="localhost",
        user="admin",
        password="admin123"
    )''',
        "language": "python",
        "has_bug": True,
        "bug_type": "security",
        "bug_description": "Hard-coded database credentials",
        "cwe": "CWE-798"
    },

    # Incorrect comparison (logic)
    {
        "id": "wrong_comparison_1",
        "code": '''function isAdult(age) {
    if (age = 18) {
        return true;
    }
    return false;
}''',
        "language": "javascript",
        "has_bug": True,
        "bug_type": "logic",
        "bug_description": "Assignment instead of comparison (= vs ==)",
        "cwe": None
    },

    # Buffer overflow (CWE-120)
    {
        "id": "buffer_overflow_1",
        "code": '''void copy_input(char *input) {
    char buffer[64];
    strcpy(buffer, input);
}''',
        "language": "c",
        "has_bug": True,
        "bug_type": "security",
        "bug_description": "Buffer overflow via strcpy",
        "cwe": "CWE-120"
    },

    # Missing return (logic)
    {
        "id": "missing_return_1",
        "code": '''def is_positive(n):
    if n > 0:
        return True
    elif n < 0:
        return False''',
        "language": "python",
        "has_bug": True,
        "bug_type": "logic",
        "bug_description": "Missing return for n == 0",
        "cwe": None
    },

    # Weak crypto (CWE-327)
    {
        "id": "weak_crypto_1",
        "code": '''import hashlib

def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()''',
        "language": "python",
        "has_bug": True,
        "bug_type": "security",
        "bug_description": "Using MD5 for password hashing (weak)",
        "cwe": "CWE-327"
    },

    # CLEAN CODE SAMPLES (no bugs)

    {
        "id": "clean_sql_1",
        "code": '''def get_user(username):
    query = "SELECT * FROM users WHERE username = %s"
    return db.execute(query, (username,))''',
        "language": "python",
        "has_bug": False,
        "bug_type": None,
        "bug_description": None,
        "cwe": None
    },

    {
        "id": "clean_file_1",
        "code": '''def process_file(path):
    with open(path, 'r') as f:
        data = f.read()
    return process(data)''',
        "language": "python",
        "has_bug": False,
        "bug_type": None,
        "bug_description": None,
        "cwe": None
    },

    {
        "id": "clean_average_1",
        "code": '''def calculate_average(numbers):
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)''',
        "language": "python",
        "has_bug": False,
        "bug_type": None,
        "bug_description": None,
        "cwe": None
    },

    {
        "id": "clean_sum_1",
        "code": '''def sum_array(arr):
    total = 0
    for i in range(len(arr)):
        total += arr[i]
    return total''',
        "language": "python",
        "has_bug": False,
        "bug_type": None,
        "bug_description": None,
        "cwe": None
    },

    {
        "id": "clean_hash_1",
        "code": '''import hashlib
import secrets

def hash_password(password):
    salt = secrets.token_hex(16)
    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()''',
        "language": "python",
        "has_bug": False,
        "bug_type": None,
        "bug_description": None,
        "cwe": None
    },
]


def get_sample(sample_id: str) -> dict:
    """Get a sample by ID."""
    for s in SAMPLES:
        if s["id"] == sample_id:
            return s
    raise ValueError(f"Unknown sample: {sample_id}")


def get_samples_by_bug_type(bug_type: str) -> list[dict]:
    """Get all samples with a specific bug type."""
    return [s for s in SAMPLES if s["bug_type"] == bug_type]


def get_buggy_samples() -> list[dict]:
    """Get all samples that have bugs."""
    return [s for s in SAMPLES if s["has_bug"]]


def get_clean_samples() -> list[dict]:
    """Get all samples without bugs."""
    return [s for s in SAMPLES if not s["has_bug"]]
