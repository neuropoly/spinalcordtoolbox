.. _mac_installation:

**********************
Installation for MacOS
**********************

Requirements
============

Supported Operating Systems
---------------------------

* MacOS >= 10.13


Mac computers with Apple silicon
--------------------------------

If your computer uses Apple silicon (M1, M2, M3, etc.), you can either :

- Install `Rosetta 2 <https://en.wikipedia.org/wiki/Rosetta_(software)#Rosetta_2>`__. See below for the installation instructions.
- Use the `Docker installation method <#docker-install-macos>`__ given further down on this page.

Rosetta installation
====================

Open a Terminal and run:

.. code:: sh

   softwareupdate --install-rosetta


Gnu Compiler Collection (gcc)
-----------------------------

You need to have ``gcc`` installed. Check to see if ``gcc`` is installed by opening a Terminal and running:

.. code:: sh

   gcc --version


If it isn't installed, we recommend installing `Homebrew <https://brew.sh/>`__ and then run:

.. code:: sh

   brew install gcc


Installation Options
====================


Option 1: Install from Package (recommended)
--------------------------------------------

The simplest way to install SCT is to do it via a stable release. First, navigate to the `latest release <https://github.com/spinalcordtoolbox/spinalcordtoolbox/releases>`__, currently |version|, then download the install script for SCT (:substitution-code:`install_sct-|version|_macos.sh`). Major changes to each release are listed in the :doc:`/dev_section/CHANGES`.

Once you have downloaded SCT, open a new Terminal in the location of the downloaded script, then launch the installer the ``bash`` command. For example, if the script was downloaded to `Downloads/`, then you would run:

.. code-block:: sh
   :substitutions:

   cd ~/Downloads
   bash install_sct-|version|_macos.sh


Option 2: Install from GitHub (development)
-------------------------------------------

If you wish to benefit from the cutting-edge version of SCT, or if you wish to contribute to the code, we recommend you download the GitHub version.

#. Retrieve the SCT code

   Clone the repository and hop inside:

   .. code:: sh

      git clone https://github.com/spinalcordtoolbox/spinalcordtoolbox

      cd spinalcordtoolbox

#. (Optional) Checkout the revision of interest, if different from `master`:

   .. code:: sh

      git checkout ${revision_of_interest}

#. Run the installer and follow the instructions

   .. code:: sh

      ./install_sct

Option 3: Install with pip (experimental)
-----------------------------------------

SCT can be installed using pip, with some caveats:

- The installation is done in-place, so the folder containing SCT must be kept around

- In order to ensure coexistence with other packages, the dependency specifications are loosened, and it is possible that your package combination has not been tested with SCT.

  So in case of problems, try again with the reference installation, and report a bug indicating the dependency versions retrieved using `sct_check_dependencies`.


Procedure:

#. Retrieve the SCT code to a safe place

   Clone the repository and hop inside:

   .. code:: sh

      git clone https://github.com/spinalcordtoolbox/spinalcordtoolbox

      cd spinalcordtoolbox

#. Checkout the revision of interest, if different from `master`:

   .. code:: sh

      git checkout ${revision_of_interest}

#. If numpy is not already on the system, install it, either using your distribution package manager or pip.

#. Install sct using pip

   If running in a virtualenv:

   .. code:: sh

      pip install -e .

   else:

   .. code:: sh

      pip install --user -e .

.. _docker-install-macos:

Option 4: Install with Docker
-----------------------------

`Docker <https://www.docker.com/what-container/>`__ is a portable (Linux, macOS, Windows) container platform.

In the context of SCT, it can be used to test SCT in a specific OS environment; this is much faster than running a fully fledged virtual machine.

Basic Installation (No GUI)
***************************

First, `install Docker Desktop <https://docs.docker.com/desktop/install/mac-install/>`__ and launch it. Then, follow the examples below to create an OS-specific SCT installation.

From an official release
^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 7.4

Consult the `spinalcordtoolbox Docker registry <https://hub.docker.com/r/neuropoly/sct/tags>`__ to select a specific release and copy its tag (like |version| as in the example below). Then, you only need to pull the image to access the full extent of SCT features in that release :

.. code-block:: bash
   :substitutions:

   # Start from the Terminal
   docker pull neuropoly/sct:|version|
   # If the previous command says 'Cannot connect to the Docker daemon', make sure the docker service is running and you have the necessary permissions to access it

Preinstalling DeepSeg tasks
"""""""""""""""""""""""""""

To inject DeepSeg tasks in an official release, you need to first pull the image, then run the following command, replacing the version (|version| in the example below) with the one you want to use :

.. code-block:: bash
   :substitutions:

   # Pull the Docker image for Ubuntu 22.04
   docker pull neuropoly/sct:|version|
   # Launch interactive mode (command-line inside container)
   sudo docker run -it neuropoly/sct:|version|
   # Now inside Docker container, install the DeepSeg tasks
   sct_deepseg spinalcord -install
   sct_deepseg tumor_t2 -install
   # Save the state of the container as a docker image.
   # Back on the Host machine, open a new terminal and run:
   sudo docker ps -a  # list all containers (to find out the container ID)
   # specify the ID, and also choose a name to use for the docker image, such as "sct_v|version|"
   sudo docker commit <CONTAINER_ID> <IMAGE_NAME>/deepseg:spinalcord-tumor_t2

From a local SCT installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 7.4

Using the `Dockerfile` available in the SCT repository, you can build the container with your local modifications. Simply run the command below, replacing ``{local_sct_repository}`` with the path to your local copy of the SCT repository (e.g. ``/home/user/spinalcordtoolbox``):

.. code:: sh

   docker build -t <IMAGE_NAME>/sct:local {local_sct_repository}

Preinstalling DeepSeg tasks
"""""""""""""""""""""""""""

You can inject DeepSeg tasks at build time by providing a comma-separated list of tasks as a build argument (``--build-arg``). For example, to install the ``spinalcord`` and ``tumor_t2`` tasks:

.. code:: sh

   docker build --build-arg DEEPSEG_TASKS=spinalcord,tumor_t2 -t <IMAGE_NAME>/sct:local {local_sct_repository}

SCT versions before 7.4
^^^^^^^^^^^^^^^^^^^^^^^

First, launch Docker Desktop, then open up a new Terminal window and run the commands below, replacing 7.3 with the version of your choice:

.. code:: bash

   # Start from the Terminal
   docker pull ubuntu:22.04
   # If the previous command says 'Cannot connect to the Docker daemon', make sure you have launched Docker Desktop
   # Launch interactive mode (command-line inside container)
   docker run -it ubuntu:22.04
   # Now, inside Docker container, install dependencies
   apt update
   apt install git bzip2 curl gcc git libdbus-1-3 libgl1-mesa-glx libglib2.0-0 libxkbcommon-x11-0 libxrender1
   # Note for above: libdbus-1-3, libgl1-mesa-glx, libglib2.0-0, libxkbcommon-x11-0, libxrender1 are required by PyQt
   # Install SCT (you can change 7.3 for the version of your choice)
   git clone --branch 7.3 https://github.com/spinalcordtoolbox/spinalcordtoolbox.git sct
   cd sct
   ./install_sct -iy
   # For the previous command, it's normal if the last two checks show [FAIL] in red
   # This will be fixed by doing the "Enable GUI Scripts" optional step in the next section
   source /root/.bashrc
   # Test SCT
   sct_testing
   # Save the state of the container as a docker image.
   # Back on the Host machine, open a new terminal and run:
   docker ps -a  # list all containers (to find out the container ID)
   # specify the ID, and also choose a name to use for the docker image, such as "sct_v7.3"
   docker commit <CONTAINER_ID> <IMAGE_NAME>/ubuntu:ubuntu22.04

Enable GUI Scripts (Optional)
*****************************

In order to run scripts with GUI you need to allow X11 redirection.
First, save your Docker image if you haven't already done so:

1. Open another Terminal
2. List current docker images

   .. code:: bash

      docker ps -a

3. Save container as new image

   .. code:: bash

      docker commit <CONTAINER_ID> <IMAGE_NAME>/<IMAGE_SPEC>

Create an X11 server for handling display:

1. Install `XQuartz X11 <https://www.xquartz.org/>`__ server.
2. Check ‘Allow connections from network clients option in XQuartz\` settings. (Run XQuartz; in the top-of-screen menu bar, choose "XQuartz", then "Preferences..."; the relevant option is under the "Security" tab.)
3. Quit and restart XQuartz.
4. In the XQuartz "xterm" window, run the command:

   .. code:: bash

      xhost + 127.0.0.1

5. In your other Terminal window, run:

   .. code:: bash

      docker run -e DISPLAY=host.docker.internal:0 -it <IMAGE_NAME>/<IMAGE_SPEC>

   (If this command says 'Cannot connect to the Docker daemon', try again after launching Docker Desktop.)

6. You can test whether GUI scripts are available by running the following command in your Docker container:

   .. code:: bash

      sct_check_dependencies

   You should see two green ``[OK]`` symbols at the bottom of the report for "PyQT" and "matplotlib" checks, which represent the GUI features provided by SCT.

Additional Notes
================

If MATLAB is Installed
----------------------

MATLAB took the liberty of setting ``DYLD_LIBRARY_PATH`` and in order for SCT to run, you have to run:

.. code:: matlab

   setenv('DYLD_LIBRARY_PATH', '');

Prior to running SCT commands.
See https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/405
