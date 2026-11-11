@echo off
rem Installation script for SCT on native Windows platforms
rem
rem Copyright (c) 2022 Polytechnique Montreal <www.neuro.polymtl.ca>
rem License: see the file LICENSE
rem
rem Usage: install_sct.bat <version>
rem e.g.
rem        install_sct.bat 5.5

echo:
echo *******************************
echo * Welcome to SCT installation *
echo *******************************

rem This option is needed for expanding !git_ref!, which is set (*and expanded*!) inside the 'if' statement below.
rem See also https://stackoverflow.com/q/9102422 for a further description of this behavior.
setLocal EnableDelayedExpansion

set TMP_DIR=%temp%\tmp-%RANDOM%%RANDOM%
mkdir %TMP_DIR%

rem Try to ensure that Git is available on the PATH prior to invoking `git clone` to avoid 'command not found' errors
rem   - See also: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3912
rem NB: This *should* be handled by the git installer, and we even have reports of people running git --version
rem successfully and still getting an error. However, there are perhaps situations where someone has installed git but
rem hasn't refreshed their terminal. Manually modifying the PATH is a bit of a hacky workaround, especially if Git has
rem been installed somewhere else, but if this mitigates a user post on the forum, this will save us some dev time.
PATH=%PATH%;C:\Program Files\Git
git --version >nul 2>&1 || (
    echo ### git not found. Make sure that git is installed ^(and a fresh Command Prompt window has been opened^) before running the SCT installer.
    goto error
)

rem Default value: 'master', however this value is updated on stable release branches.
set git_ref=master

rem Check to see if the PWD contains the project source files (using `version.txt` as a proxy for the entire source dir)
rem If it exists, then we can reliably access source files (`version.txt`, `requirements-freeze.txt`) from the PWD.
if exist spinalcordtoolbox\version.txt (
  set SCT_SOURCE=%cd%
rem If version.txt isn't present, then the installation script is being run by itself (i.e. without source files).
rem So, we need to clone SCT to a TMPDIR to access the source files, and update SCT_SOURCE accordingly.
) else (
  set SCT_SOURCE=%TMP_DIR%\spinalcordtoolbox
  echo:
  echo ### Source files not present. Downloading source files ^(@ !git_ref!^) to !SCT_SOURCE!...
  git clone -b !git_ref! --single-branch --depth 1 https://github.com/spinalcordtoolbox/spinalcordtoolbox.git !SCT_SOURCE!
  rem Since we're git cloning into a TMPDIR, this can never be an "in-place" installation, so we force "package" instead.
  set SCT_INSTALL_TYPE=package
)

rem Get installation type if not already specified
if [%SCT_INSTALL_TYPE%]==[] (
  rem The file 'requirements-freeze.txt` only exists for stable releases
  if exist %SCT_SOURCE%\requirements-freeze.txt (
    set SCT_INSTALL_TYPE=package
  rem If it doesn't exist, then we can assume that a dev is performing an in-place installation from master
  ) else (
    set SCT_INSTALL_TYPE=in-place
  )
)

rem Fetch the version of SCT from the source file
for /F %%g IN (%SCT_SOURCE%\spinalcordtoolbox\version.txt) do (set SCT_VERSION=%%g)

echo:
echo ### SCT version ......... %SCT_VERSION%
echo ### Installation type ... %SCT_INSTALL_TYPE%

rem if installing from git folder, then becomes default installation folder
if %SCT_INSTALL_TYPE%==in-place (
  echo ### Setting default installation directory to source folder: %SCT_SOURCE%
  set SCT_DIR=%SCT_SOURCE%
) else (
  echo ### Setting default installation directory to home folder: %USERPROFILE%\sct_%SCT_VERSION%
  set SCT_DIR=%USERPROFILE%\sct_%SCT_VERSION%
)

rem Validate the default installation directory 
rem If it's not valid, don't propose it to the user
echo ### Checking default installation directory for potential issues...
call :validate_sct_dir || goto :while_prompt_custom_path

rem Count blank attempts when choosing a custom install path
rem to avoid infinite loops
set attempt=0

:while_prompt_default_path
  echo:
  echo ### SCT will be installed here: [%SCT_DIR%]

  rem The non-interactive default is to accept
  set keep_default_path=y
  set /p keep_default_path="### Do you agree? [y]es/[n]o: "

  rem Either the installation is non-interactive,
  rem or the user accepts the default path
  if /i ["%keep_default_path%"]==["y"] goto :done_sct_dir

  rem The user wants a non-default path
  if /i ["%keep_default_path%"]==["n"] goto :while_prompt_custom_path
goto :while_prompt_default_path

:while_prompt_custom_path
  set /a attempt=attempt+1
  if !attempt! GTR 10 (
    rem The install path was invalid 10 times, so this is probably non-interactive. Halt.
      echo ### ERROR: The chosen installation directory is invalid, and no valid input was passed.
      echo            Please make sure to enter a valid input.
      goto error

  )

  echo:
  echo ### Choose install directory.

  set "SCT_DIR="
  set /p SCT_DIR="### Warning^! Give full path ^(e.g. C:\Users\username\sct_v3.0^): "

  rem Ask again if the given path is invalid
  call :validate_sct_dir && goto :done_sct_dir
goto :while_prompt_custom_path

:validate_sct_dir
rem Validate the user's choice of path in SCT_DIR
rem Check for an empty path
if ["%SCT_DIR%"]==[] exit /b 1
rem Check for spaces
rem TODO: This may no longer be necessary as of a patch made to Mamba in Dec. 2024!
if not "%SCT_DIR%"=="%SCT_DIR: =%" (
  echo ### WARNING: Install directory %SCT_DIR% contains spaces.
  echo ### SCT uses conda, which does not permit spaces in installation paths.
  echo ### More details can be found here: https://github.com/ContinuumIO/anaconda-issues/issues/716
  echo:
  exit /b 1
)
rem Check number of characters in path
set tmpfile_len=%TMP_DIR%\tmpfile_len.txt
echo %SCT_DIR%> !tmpfile_len!
for /F "usebackq" %%a in ('!tmpfile_len!') do set /a size=%%~za - 2
if !size! GTR 113 (
  echo ### WARNING: '%SCT_DIR%' exceeds path length limit ^(!size! ^> 113^). Please choose a shorter path.
  exit /b 1
)
rem We passed all the checks
exit /b 0

:done_sct_dir
rem At this point, SCT_DIR is a valid path accepted by the user

rem Create directory
if not exist %SCT_DIR% (
  mkdir %SCT_DIR% || goto error
) else (
  echo ### WARNING: '%new_install%' already exists. Files will be overwritten.
)

rem Copy files to destination directory
echo:
set tmpfile_exclude=%TMP_DIR%\exclusion.txt
echo %SCT_SOURCE%\bin> %tmpfile_exclude%
echo %SCT_SOURCE%\python>> %tmpfile_exclude%
echo %SCT_SOURCE%\spinalcordtoolbox.egg-info\>> %tmpfile_exclude%
if not %SCT_DIR%==%SCT_SOURCE% (
  echo ### Copying source files from %SCT_SOURCE% to %SCT_DIR%
  xcopy /s /e /q /y %SCT_SOURCE% %SCT_DIR% /EXCLUDE:%tmpfile_exclude% || goto error
) else (
  echo ### Skipping copy of source files ^(source and destination folders are the same^)
)

rem Clean old install setup in bin/ if existing
if exist %SCT_DIR%\bin\ (
  echo ### Removing sct and isct softlink inside the SCT directory...
  del %SCT_DIR%\bin\sct_* || goto error
  del %SCT_DIR%\bin\isct_* || goto error
)
rem Remove old python folder
if exist %SCT_DIR%\python\ (
  echo ### Removing existing 'python' folder inside the SCT directory...
  rmdir /s /q %SCT_DIR%\python\ || goto error
)
rem Remove old '.egg-info` folder created by editable installs
if exist %SCT_DIR%\spinalcordtoolbox.egg-info\ (
  echo ### Removing existing '.egg-info' folder inside the SCT directory...
  rmdir /s /q %SCT_DIR%\spinalcordtoolbox.egg-info\ || goto error
)

rem Move into the SCT installation directory
pushd %SCT_DIR% || goto error

rem Install portable miniforge instance. (Command source: https://github.com/conda/conda/issues/1977)
echo:
echo ### Downloading Miniforge installer...
curl -o %TMP_DIR%\miniforge.exe -L https://github.com/conda-forge/miniforge/releases/download/26.7.2-1/Miniforge3-Windows-x86_64.exe
echo:
echo ### Installing portable copy of Miniforge...
start /wait "" %TMP_DIR%\miniforge.exe /InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /NoRegistry=1 /S /D=%cd%\python

rem Create and activate miniforge environment to install SCT into
echo:
echo ### Using Conda to create virtual environment...
python\Scripts\conda create -y -p python\envs\venv_sct python=3.11 || goto error
echo Virtual environment created successfully!

rem Install SCT and its requirements
if exist requirements-freeze.txt (
  set requirements_file=requirements-freeze.txt || goto error
) else (
  set requirements_file=requirements.txt || goto error
)
echo:
echo ### Installing SCT and its dependencies from %requirements_file%...
rem Avoid eMudhra / emSign certificates with >=26.2.2 (https://discuss.python.org/t/information-for-python-users-regarding-emudhra-emsign-ca-certificate-security-incident/109576/40)
python\envs\venv_sct\python -m pip install -U "pip>=26.2.2" || goto error
python\envs\venv_sct\Scripts\pip install -r %requirements_file% || goto error
python\envs\venv_sct\Scripts\pip install -e . --use-pep517 || goto error

rem Install external dependencies
echo:
echo ### Downloading model files and binaries...
python\envs\venv_sct\Scripts\sct_download_data -d binaries_win -k
python\envs\venv_sct\Scripts\sct_download_data -d default -k
python\envs\venv_sct\python -c "import spinalcordtoolbox.deepseg.models; spinalcordtoolbox.deepseg.models.install_default_models()"

rem Copying SCT scripts to an isolated folder (so we can add scripts to the PATH without adding the entire venv_sct)
echo:
echo ### Copying SCT's CLI scripts to %CD%\bin\
xcopy %CD%\python\envs\venv_sct\Scripts\*sct*.* %CD%\bin\ /v /y /q /i || goto error

echo ### Checking installation...
python\envs\venv_sct\Scripts\sct_check_dependencies

rem Give further instructions that the user add the Scripts directory to their PATH
echo:
echo ### Installation finished!
echo:
echo To use SCT's command-line scripts in Command Prompt, please follow these instructions:
echo:
echo 1. Open the Start Menu -^> Type 'edit environment' -^> Open 'Edit environment variables for your account'
echo 2. Click 'New', then enter 'SCT_DIR' for the variable name. For the value, copy and paste this directory:
echo:
echo    %CD%
echo:
echo 3. Click 'OK', then click on the 'Path' variable, then click the 'Edit...' button.
echo 4. Click 'New', then copy and paste this directory:
echo:
echo    %CD%\bin\
echo:
echo 5. Click 'OK' three times. You can now access SCT's scripts in the Command Prompt.
echo:
echo If you have any questions or concerns, feel free to create a new topic on SCT's forum:
echo   --^> https://forum.spinalcordmri.org/c/sct

rem Return to initial directory and deactivate the virtual environment
goto exit

:error
set cached_errorlevel=%errorlevel%
echo:
echo Installation failed with error code %cached_errorlevel%.
echo Please copy and paste the installation log in a new topic on SCT's forum:
echo   --^> https://forum.spinalcordmri.org/c/sct

:exit
if "%cached_errorlevel%"=="" set cached_errorlevel=0
popd
where deactivate >nul 2>&1
PAUSE
exit /b %cached_errorlevel%
