import argparse
import json
import re
import stat
import sys
from pathlib import Path


HEADER_RE = re.compile(r'^\s*#\s*(?P<title>.+?)\s*$')
SEPARATOR_RE = re.compile(r'^\s*#\s*={5,}\s*$')
DATASET_IN_TITLE_RE = re.compile(r'\s*\(dataset:\s*(?P<dataset>[^)]+?)\s*\)')

# FIXME: List of dataset zip assets published on the sct_tutorial_data
#   GitHub release page.  Any dataset in this set that is NOT referenced by at
#   least one section in batch_single_subject.sh will be reported so it can be
#   manually reviewed and a header added. We probably want to generate this
#   automatically from the release assets using the GitHub API at some point
KNOWN_DATASETS = {
    # "data_batch-processing-of-subjects",
    #     NB1: this one is intentionally excluded: it is a multi-subject
    #          workflow tutorial (using sct_run_batch) that is handled by a separate script/pipeline
    #          and does not belong in batch_single_subject.sh.
    # "data_normalizing-morphometrics-compression",
    #     NB2: this one is intentionally excluded: it is a special case that only
    #          provides pre-generated intermediate files (segmentation, vertebral labels,
    #          compression labels) that users can download to skip the generation steps
    #          already covered by the data_compression section.  All commands that consume
    #          these files are included in the data_compression section, so no separate section
    #          header is needed in batch_single_subject.sh.
    "data_atlas-based-analysis",
    "data_compression",
    "data_contrast-agnostic-registration",
    "data_coregistration",
    "data_gm-wm-metric-computation",
    "data_gm-wm-segmentation",
    "data_improving-registration-with-gm-seg",
    "data_lesion-analysis",
    "data_lumbar-registration",
    "data_ms-lesion-segmentation",
    "data_mtr-computation",
    "data_processing-dmri-data",
    "data_processing-fmri-data",
    "data_rootlets-registration",
    "data_shape-metric-computation",
    "data_spinalcord-segmentation",
    "data_spinalcord-smoothing",
    "data_template-registration",
    "data_vertebral-labeling",
    "data_visualizing-misaligned-cords",
}

# ==============================================================================
#               Split batch_single_subject.sh into mini-scripts
# ==============================================================================
# NB: Sections are defined by the header:
#
#     # Section Title (dataset: data_[DATASET_NAME])
#     # ============================================
#
# Sections with headers that don't contain `(dataset: ...)` are skipped.


def parse_sections(lines):
    sections = []
    n = len(lines)

    # Find START OF SCRIPT marker present in batch_single_subject.sh to avoid its preamble
    for idx, line in enumerate(lines):
        if 'START OF SCRIPT' in line:
            i = idx + 1
            break
    else:
        raise ValueError('Could not find "START OF SCRIPT" marker in input file. Cannot determine where the actual script starts.')

    while i < n:
        line = lines[i]
        m_header = HEADER_RE.match(line)
        if not (m_header and i + 1 < n and SEPARATOR_RE.match(lines[i + 1])):
            i += 1
            continue

        # collect body until next header+separator or EOF
        j = i + 2
        while j < n:
            m_header_next = HEADER_RE.match(lines[j])
            if m_header_next and j + 1 < n and SEPARATOR_RE.match(lines[j + 1]):
                break
            j += 1

        # construct the section (depending on whether the header had a dataset or not)
        section = {'title': m_header.group('title'), 'body': lines[i+2:j],
                   'start_line': i + 1, 'end_line': j}
        m_dataset = DATASET_IN_TITLE_RE.search(section['title'])
        if m_dataset:  # extract the dataset name and clean it from the title
            section['title'] = DATASET_IN_TITLE_RE.sub('', section['title'])
            section['dataset'] = m_dataset.group('dataset')
            sections.append(section)
        else:
            print(
                f'WARNING: skipping section at line {i + 1} '
                f'"{section["title"]}" — no (dataset: ...) marker in header.',
                file=sys.stderr,
            )

        i = j

    return sections


# ==============================================================================
#            Handle the `cd` commands that may or may not be present
# ==============================================================================
# Each mini-script is run fresh from `$DATA_DIR`.  The original
# `batch_single_subject.sh` chains all sections together, so a later section
# can write e.g. `cd ../mt` because the previous section left the shell inside
# `t2/`.  That navigation breaks when the section runs in isolation. So, we
# replace every `cd RELATIVE` with `cd "$DATA_DIR/RESOLVED"` so that the
# isolated mini-script always navigates correctly regardless of where the shell
# started.

# Assumptions made to simplify the code:
#   * Data directories are always a single subdirectory name, optionally
#     preceded by `../` (e.g. `cd t2`, `cd ../mt`).
#   * Exceptionally, a bare `..` is allowed (which produces an empty
#     subdirectory name).
#   * Data directories are always relative to `$DATA_DIR`.
#   * No spaces, glob characters, variables, or multi-part paths.

# NB: regex that matches a live (non-commented) `cd TARGET` line.
# - Group 1: leading whitespace
# - Group 2: path (which may be quoted)
# - Group 3: trailing whitespace / inline comment (preserved verbatim)
CD_LINE_RE = re.compile(r'^(\s*)cd\s+(\S+)(\s*(?:#.*)?)$')

# NB: regex that validates our path assumptions.
# - Optional leading `../`.
# - A single directory name, with no special characters.
# - Optional trailing slash `/`.
# - Or, literally `..`.
CD_PATH_RE = re.compile(r'^(?:\.\./)?(?P<dir_name>[-_0-9A-Za-z]+)/?$|^\.\.$')


def rewrite_cd_commands(body_lines: list, start_cwd: str = '') -> tuple:
    current_cwd = start_cwd
    result = []
    for line in body_lines:
        # Deconstruct the `cd` command from the regex groups (if present)
        m_cd_line = CD_LINE_RE.match(line)
        if not m_cd_line:
            result.append(line)
            continue
        indent, target, tail = m_cd_line.groups()

        m_cd_path = CD_PATH_RE.match(target)
        if not m_cd_path:
            raise ValueError(f'Could not parse `cd` target "{target}" in input file.')
        dir_name = m_cd_path.group('dir_name') or ''

        # Both `cd t2` and `cd ../t2` resolve to the same subdirectory
        # relative to `$DATA_DIR` given the flat, single-depth layout.
        current_cwd = dir_name
        result.append(f'{indent}cd "$DATA_DIR/{current_cwd}"{tail}\n')

    return result, current_cwd


# ==============================================================================
#                   Write the processed mini-scripts to files
# ==============================================================================


def make_slug(s):
    s = s.lower()                          # lowercase
    s = re.sub(r"[^a-z0-9]+", '-', s)      # replace non-alphanumeric with hyphen
    s = re.sub(r'-+', '-', s).strip('-')   # collapse multiple and strip leading/trailing hyphens
    return s or 'section'


def make_header():
    return """#!/usr/bin/env bash
set -euo pipefail
echo "== Running mini-script generated from batch_single_subject.sh =="
"""


def section_starts_with_cd(body_lines):
    for line in body_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        # If an SCT command is encountered, the section is assumed to not start with a CD command,
        # even if one comes later.
        if stripped.startswith('sct_'):
            return False
        if bool(CD_LINE_RE.match(line)):
            return True
    return False


def write_scripts(sections, path_out):
    path_out.mkdir(parents=True, exist_ok=True)
    manifest = []
    current_cwd = ''  # NB: Will result in `cd "$DATA_DIR/"` if the first section has no `cd` of its own.
    for idx, section in enumerate(sections, start=1):
        # construct the file path from the title and dataset
        # this results in a long filenames with redundant info, but is good for provenance
        # and helps avoid cases where e.g. the same title/dataset are used for multiple sections
        name = section['title']
        dataset = section['dataset']  # always present — sections without dataset are skipped in parse_sections
        fname = f"{idx:02d}_{make_slug(name).upper()}_{make_slug(dataset)}.sh"
        fpath = path_out / fname

        with fpath.open('w', newline='\n') as fh:
            # write the header and section title to the mini-script
            fh.write(make_header())
            fh.write('\n')
            fh.write(f'echo "== Section: {name} (dataset: {dataset}) =="\n')
            fh.write('\n')

            # determine whether the section body opens with its own `cd` (ignoring blank lines and comments).
            # if it doesn't, we need to inject the current cwd to make sure relative paths point to the right files.
            body = section['body']
            if not section_starts_with_cd(body) and current_cwd:
                fh.write('# Starting directory carried over from previous section in original script\n')
                fh.write(f'cd "$DATA_DIR/{current_cwd}"\n')
                fh.write('\n')

            # write the body lines to the script (making sure to rewrite any `cd` commands so that they always point
            # to the correct subdirectory relative to `$DATA_DIR`)
            body, current_cwd = rewrite_cd_commands(body, current_cwd)
            for body_line in body:
                fh.write(body_line.rstrip('\n') + '\n')

        # make the script executable. we only set the owner execute bit because
        # this always runs using the default GHA user in a temporary Ubuntu container,
        # thus group/other execute permissions are unnecessary.
        fpath.chmod(fpath.stat().st_mode | stat.S_IEXEC)

        manifest.append({
            'name': name,
            'dataset': dataset,
            'file': fname,
            'start_line': section['start_line'],
            'end_line': section['end_line'],
        })

    return manifest


# ==============================================================================
#                        main function and CLI entry point
# ==============================================================================


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', help='Path to batch_single_subject.sh')
    p.add_argument('--out', default='.github/workflows/scripts/batch_single_subject', help='Output directory for mini-scripts')
    args = p.parse_args(argv)

    path_in = Path(args.input)
    if not path_in.exists():
        print(f'Error: {path_in} not found', file=sys.stderr)
        return 2

    lines = path_in.read_text(encoding='utf-8').splitlines()
    sections = parse_sections(lines)
    if not sections:
        print('No sections with a (dataset: ...) marker detected. Exiting.', file=sys.stderr)
        return 1

    path_out = Path(args.out)
    manifest = write_scripts(sections, path_out)

    manifest_path = path_out / 'sections.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Wrote {len(manifest)} mini-scripts to {path_out} and manifest to {manifest_path}')

    # Cross-reference datasets found in the script against the known release assets.
    referenced_datasets = {entry['dataset'] for entry in manifest}
    unaccounted = sorted(KNOWN_DATASETS - referenced_datasets)
    if unaccounted:
        print(
            '\nNOTE: The following datasets are published as release assets but have NO '
            'corresponding section (with a dataset: header) in the batch script.\n'
            'These should be reviewed and a new section header added if applicable:',
            file=sys.stderr,
        )
        for ds in unaccounted:
            print(f'  - {ds}', file=sys.stderr)
        print(file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
