#!/usr/bin/env python3
"""Compare sct_* command invocations between tutorials and batch scripts."""

import logging
import re
import sys
from collections import Counter
from pathlib import Path


# This script lives at $SCT_DIR/.github/workflows/extract_commands.py
PWD = Path(__file__).parent
DOC_PATH = PWD.parent.parent / "documentation" / "source" / "user_section" / "tutorials"
# FIXME: These are just copies of the upstream `sct_tutorial_data` scripts. We still need to figure out a way to
#        have the scripts live in the SCT repo while also packaging the scripts alongside the `sct_tutorial_data` repo
SCRIPT_PATHS = [
    PWD.parent.parent / "batch_single_subject.sh",
    PWD / "sct_tutorial_data" / "multi_subject" / "sample_usage.sh"
]

logger = logging.getLogger(__name__)

# Matches the opening of a code block directive (language is optional).
# Group 1 captures the leading whitespace (directive's own indentation level).
_CODE_BLOCK_RE = re.compile(r'^(\s*)\.\.\s+code(?:-block)?::(?:\s+\S+)?\s*$')

# A command name is sct_ followed by word characters
_SCT_CMD_RE = re.compile(r'^sct_[a-zA-Z0-9_]+')

# Things to skip
_PLACEHOLDER_RE_LST = [re.compile(r'<[A-Z][A-Z0-9_]*>')]     # e.g. <IMAGE>
_SCT_VAR_ASSIGNMENT_RE = re.compile(r'^sct_[a-zA-Z0-9_]+=')  # e.g. sct_python=
_SKIP_TUTORIAL_FILE = "before-starting.rst"                  # page with setup steps
_SKIP_INITIAL_DOWNLOAD = "sct_download_data -d sct_course_data"  # only for full bash script


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def join_continuations(lines):
    """Join backslash-continued lines."""
    joined = ''
    for raw in lines:
        stripped = raw.rstrip('\n').rstrip('\r')
        if stripped.endswith('\\'):
            joined += stripped[:-1] + ' '
        else:
            joined += stripped
            yield joined
            joined = ''
    if joined:
        yield joined


def normalize(cmd):
    """Collapse internal whitespace and trim ends."""
    return re.sub(r'\s+', ' ', cmd).strip()


def has_args(cmd):
    """Return whether a command includes arguments."""
    # After the command name there must be at least one non-space character
    match = _SCT_CMD_RE.match(cmd)
    if not match:
        return False
    rest = cmd[match.end():]
    return bool(rest.strip())


def is_extractable_command(cmd):
    """Return whether a normalized command should be kept."""
    if _SCT_VAR_ASSIGNMENT_RE.match(cmd):
        return False
    if any(p.search(cmd) for p in _PLACEHOLDER_RE_LST):
        return False
    return bool(_SCT_CMD_RE.match(cmd) and has_args(cmd))


# ---------------------------------------------------------------------------
# RST extraction
# ---------------------------------------------------------------------------

def extract_commands_from_tutorials(tutorials_path):
    """Iterate over tutorial RST files and return normalized sct_* commands."""
    all_cmds = []
    rst_files = sorted(tutorials_path.glob("**/*.rst"))
    logger.info(f"Found {len(rst_files)} RST tutorial files in {tutorials_path}")
    for rst_file in rst_files:
        if rst_file.name == _SKIP_TUTORIAL_FILE:
            continue
        cmds = extract_commands_from_rst(rst_file.read_text(encoding='utf-8'))
        all_cmds.extend(cmds)
    return all_cmds


def extract_commands_from_rst(content):
    """Find code blocks in raw RST content and return normalized sct_* commands."""
    results = []
    for block in extract_code_block_lines(content):
        for line in join_continuations(block):
            cmd = normalize(line)
            if is_extractable_command(cmd):
                results.append(cmd)
    return results


def extract_code_block_lines(content):
    """Generator function that extracts code-block lines from RST content."""
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m_code_block = _CODE_BLOCK_RE.match(line)
        if m_code_block:
            directive_indent = m_code_block.group(1)
            # Skip blank lines immediately after directive header
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i >= len(lines):
                return
            # Determine indent string from first content line
            first = lines[i]
            block_indent_len = len(first) - len(first.lstrip())
            block_indent = first[:block_indent_len]
            if not block_indent.startswith(directive_indent) or block_indent == directive_indent:
                raise ValueError(f"Malformed code block at line {i+1}: {first!r}")
            # Extract lines of code (removing the block indent)
            block_lines = []
            while i < len(lines):
                line = lines[i]
                if line.strip() and not line.startswith(block_indent):
                    break
                block_lines.append(line.removeprefix(block_indent))
                i += 1
            yield block_lines
        else:
            i += 1


# ---------------------------------------------------------------------------
# Bash script extraction
# ---------------------------------------------------------------------------

def extract_commands_from_bash(content):
    """Return normalized sct_* commands from a bash script."""
    results = []
    script_lines = content.splitlines(keepends=True)
    # Pre-strip one leading '#' so commented command continuations can be joined and parsed like regular commands.
    # This allows block comments ('##') to fully exclude commands from processing.
    script_lines = [re.sub(r'^(\s*)#\s?', r'\1', line) for line in script_lines]

    for line in join_continuations(script_lines):
        stripped = line.lstrip()
        # Skip blank lines and comment lines
        if not stripped or stripped.startswith('#'):
            continue
        # Skip full dataset download command
        if stripped == _SKIP_INITIAL_DOWNLOAD:
            continue
        cmd = normalize(stripped)
        if is_extractable_command(cmd):
            results.append(cmd)
    return results


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_commands(doc_cmds, script_cmds):
    """Compare sets of commands and return status plus diff text."""
    doc_counts = Counter(doc_cmds)
    script_counts = Counter(script_cmds)

    all_cmds = sorted(set(doc_counts) | set(script_counts))

    red = '\033[31m'
    green = '\033[32m'
    reset = '\033[0m'

    def format_line(prefix, cmd):
        line = f"{prefix} {cmd}"
        if prefix == '-':
            return f"{red}{line}{reset}"
        if prefix == '+':
            return f"{green}{line}{reset}"
        return line

    if doc_counts == script_counts:
        return True, "Commands are in sync"

    parts = ["Commands are out of sync:\n", "Legend: ' ' = same, '-' = tutorials only, '+' = script only\n"]
    for cmd in all_cmds:
        doc_n = doc_counts.get(cmd, 0)
        script_n = script_counts.get(cmd, 0)
        same_n = min(doc_n, script_n)
        for _ in range(same_n):
            parts.append(format_line(' ', cmd))
        for _ in range(doc_n - same_n):
            parts.append(format_line('-', cmd))
        for _ in range(script_n - same_n):
            parts.append(format_line('+', cmd))
    return False, "\n".join(parts)


def main():
    # extract documentation commands
    doc_cmds = extract_commands_from_tutorials(DOC_PATH)
    logger.info(f"Found {len(doc_cmds)} invocations in tutorials")

    # extract script commands
    script_cmds = []
    for source in SCRIPT_PATHS:
        logger.info(f"Reading script from {source}")
        content = Path(source).read_text(encoding='utf-8')
        source_cmds = extract_commands_from_bash(content)
        logger.info(f"Found {len(source_cmds)} invocations in script: {source}")
        script_cmds.extend(source_cmds)
    logger.info(f"Found {len(script_cmds)} invocations across all scripts")

    # compare commands and exit
    synced, message = compare_commands(doc_cmds, script_cmds)
    print(message)
    return 0 if synced else 1


if __name__ == '__main__':
    sys.exit(main())
