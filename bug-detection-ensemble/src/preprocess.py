"""
C/C++ comment stripping utility.

Removes comments from C/C++ code while preserving string literals.
This is important for Juliet samples which contain `/* FLAW */` markers
that leak the answer.
"""

import re


def strip_c_comments(code: str) -> str:
    """
    Remove C/C++ comments while preserving string literals.

    Handles:
    - Single-line comments (//)
    - Multi-line comments (/* ... */)
    - String literals with escaped quotes
    - Character literals

    Args:
        code: C/C++ source code

    Returns:
        Code with comments removed
    """
    result = []
    i = 0
    n = len(code)

    while i < n:
        # Check for string literal
        if code[i] == '"':
            # Find end of string, handling escapes
            j = i + 1
            while j < n:
                if code[j] == '\\' and j + 1 < n:
                    j += 2  # Skip escaped character
                elif code[j] == '"':
                    j += 1
                    break
                else:
                    j += 1
            result.append(code[i:j])
            i = j

        # Check for character literal
        elif code[i] == "'":
            j = i + 1
            while j < n:
                if code[j] == '\\' and j + 1 < n:
                    j += 2
                elif code[j] == "'":
                    j += 1
                    break
                else:
                    j += 1
            result.append(code[i:j])
            i = j

        # Check for single-line comment
        elif i + 1 < n and code[i:i+2] == '//':
            # Skip to end of line
            j = i + 2
            while j < n and code[j] != '\n':
                j += 1
            # Keep the newline if present
            if j < n:
                result.append('\n')
                j += 1
            i = j

        # Check for multi-line comment
        elif i + 1 < n and code[i:i+2] == '/*':
            # Find closing */
            j = i + 2
            while j + 1 < n and code[j:j+2] != '*/':
                j += 1
            if j + 1 < n:
                j += 2  # Skip past */
            # Preserve line count by keeping newlines
            newlines = code[i:j].count('\n')
            result.append('\n' * newlines)
            i = j

        else:
            result.append(code[i])
            i += 1

    return ''.join(result)


def strip_juliet_markers(code: str) -> str:
    """
    Remove Juliet-specific markers that leak vulnerability information.

    Markers removed:
    - /* FLAW */
    - /* POTENTIAL FLAW */
    - /* FIX */
    - OMITBAD/OMITGOOD preprocessor directives
    """
    # First strip all comments
    code = strip_c_comments(code)

    # Remove OMIT* preprocessor lines
    lines = code.split('\n')
    filtered = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#ifndef OMIT') or stripped.startswith('#endif'):
            continue
        if 'OMITBAD' in stripped or 'OMITGOOD' in stripped:
            continue
        filtered.append(line)

    return '\n'.join(filtered)


def clean_whitespace(code: str) -> str:
    """
    Clean up excessive whitespace while preserving structure.

    - Removes trailing whitespace on each line
    - Collapses multiple blank lines to single blank line
    - Preserves indentation
    """
    lines = code.split('\n')
    result = []
    prev_blank = False

    for line in lines:
        line = line.rstrip()
        is_blank = not line.strip()

        if is_blank:
            if not prev_blank:
                result.append('')
            prev_blank = True
        else:
            result.append(line)
            prev_blank = False

    # Remove leading/trailing blank lines
    while result and not result[0]:
        result.pop(0)
    while result and not result[-1]:
        result.pop()

    return '\n'.join(result)


def preprocess_code(code: str, strip_comments: bool = True) -> str:
    """
    Full preprocessing pipeline for code samples.

    Args:
        code: Source code
        strip_comments: Whether to remove comments

    Returns:
        Preprocessed code
    """
    if strip_comments:
        code = strip_juliet_markers(code)
    code = clean_whitespace(code)
    return code


if __name__ == "__main__":
    # Test with a Juliet-style sample
    test_code = '''
void CWE121_bad() {
    char data[50];
    char source[100];
    /* FLAW: Possible buffer overflow */
    memcpy(data, source, 100);  // This is dangerous
    printLine(data);
}
'''
    print("Original:")
    print(test_code)
    print("\nStripped:")
    print(preprocess_code(test_code))
