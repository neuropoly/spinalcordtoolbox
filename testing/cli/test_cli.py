# pytest unit tests for all cli scripts

import pytest
import sys
import importlib
from importlib.metadata import entry_points
import subprocess

scripts = [cs.name for cs in entry_points().select(group='console_scripts') if cs.value.startswith("spinalcordtoolbox")]

scripts_where_no_args_is_valid = [
    'isct_test_ants',          # No args -> tests ants binaries
    'sct_check_dependencies',  # No args -> checks dependencies
    'sct_version',             # No args -> prints version
    'sct_testing'              # No args -> runs pytest in $SCT_DIR
]

scripts_to_test = [s for s in scripts if s not in scripts_where_no_args_is_valid]


@pytest.mark.parametrize("script", scripts_to_test)
def test_calling_scripts_with_no_args_shows_usage(capsys, script):
    """
    Test that SCT's scripts all return error code 2 and show usage descriptions when called with no arguments.
    """
    completed_process = subprocess.run([script], capture_output=True)
    assert completed_process.returncode == 2
    assert b'usage' in completed_process.stderr


@pytest.fixture
def sandbox_modules():
    """Fixture to keep sys.modules sandboxed by backing up and restoring its state."""
    orig_modules = sys.modules.copy()
    yield
    sys.modules.clear()
    sys.modules.update(orig_modules)


@pytest.mark.parametrize("script", scripts_to_test)
def test_importing_scripts_avoids_expensive_modules(script, sandbox_modules):
    """
    Test that importing SCT's scripts does not import expensive modules like numpy or scipy.
    This ensures that lazy loading is working correctly.
    """
    # NB: The list of expensive modules comes from the modules we're already lazy-loading. The thought process is that
    #     if we're lazy-loading that module somewhere, we should do it everywhere consistently.
    expensive_modules = [
        'dipy',
        'matplotlib',
        'nibabel',
        'nilearn',
        'pandas',
        'scipy.signal',
        'scipy.stats',
        'sklearn',
        'torch',
        'spinalcordtoolbox.registration.vxm',
    ]

    # Wipe `sys.modules` of expensive modules to force reimports
    for m in list(sys.modules.keys()):
        for em in expensive_modules:
            if m.startswith(em):
                del sys.modules[m]

    # Import the script module
    before = set(sys.modules.keys())
    importlib.import_module(f"spinalcordtoolbox.scripts.{script}")
    after = set(sys.modules.keys())

    # Check that expensive modules are not in sys.modules
    imported_modules = after - before
    found_modules = []
    for em in expensive_modules:
        if any(im.startswith(em) for im in imported_modules):
            found_modules.append(em)
    if found_modules:
        pytest.fail(f"Expensive modules found while importing {script}. Please lazy-load these modules: {found_modules}")
