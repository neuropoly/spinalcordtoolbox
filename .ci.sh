#!/usr/bin/env bash
# CI testing script
#  Installs SCT from scratch and runs all the tests we've ever written for it.

# stricter shell mode
# https://sipb.mit.edu/doc/safe-shell/
set -eo pipefail  # exit if non-zero error is encountered (even in a pipeline)
set -u            # exit if unset variables used
shopt -s failglob # error if a glob doesn't find any files, instead of remaining unexpanded

export SCT_PROGRESS_BAR=off # disable SCT's progress bar (e.g. for sct_download_data)
export PIP_PROGRESS_BAR=off # disable pip's progress bar for the duration of CI
export PY_COLORS=1 # Colored pytest output (https://github.com/pytest-dev/pytest/issues/7443#issuecomment-656642591)

install_sct () {
  # NB: We want to test the "standalone install" when on master, so copy the install script outside the srcdir
  if [ "$GITHUB_REF_NAME" = "master" ]; then
    mkdir -p "$HOME/Downloads"
    cp install_sct "$HOME/Downloads"
    cd "$HOME/Downloads"
  fi
  # NB: we only force in-place (-i) installs to avoid pytest running from the source
  #     instead of the installed folder, where the extra detection models are.
  #     Further explanation at https://blog.ionelmc.ro/2014/05/25/python-packaging/#the-structure
  #     TO BE REMOVED during https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3140.
  # NB: '-c' disables sct_check_dependencies so we can check it separately
  ./install_sct -iyc
}

activate_venv_sct(){
  # NB: If "standalone install" is tested when on master, then `python/` won't exist in the PWD
  if [ "$GITHUB_REF_NAME" = "master" ]; then
    SCT_VERSION=$(cat spinalcordtoolbox/version.txt)
    cd "$HOME/sct_$SCT_VERSION"
  fi
  # NB: `python/` won't be available unless `install_sct` is run, hence disabling the check:
  # shellcheck disable=SC1091
  source python/etc/profile.d/conda.sh  # to be able to call conda
  set +u
  conda activate venv_sct
  set -u
}

check_dependencies() {
  activate_venv_sct
  sct_check_dependencies
}

run_tests() {
  activate_venv_sct
  # re-run failed tests, just in case something flaky has happened. if no failures, do nothing.
  pytest testing/api testing/cli || pytest --last-failed --last-failed-no-failures none
  # NB: 'testing/batch_single_subject' is run by a separate CI workflow
}

while getopts ":ict" opt; do
  case $opt in
  i)
    install_sct
    ;;
  c)
    check_dependencies
    ;;
  t)
    run_tests
    ;;
  *)
    exit 99
    ;;
  esac
done
