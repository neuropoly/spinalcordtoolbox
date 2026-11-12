.. _linux_installation:

**********************
Installation for Linux
**********************

.. warning::

    If you use Windows Subsystem for Linux (WSL), please refer to the :ref:`Windows installation section <wsl-installation>`.


Installation Options
====================

Depending on your use case, there are a number of different installation methods available:

- :ref:`Packaged Installation <native-install>`: For general use.
- :ref:`GitHub Installation <github-install>`: For developers.
- :ref:`Docker Installation <docker-install-linux>`: For cross-platform use/testing.
- **[Experimental]** :ref:`PIP Installation <pip-install>` Installation ``pip`` for use as a Python package.


Requirements
============

Supported Operating Systems
---------------------------

* Debian >=9
* Ubuntu >= 16.04
* Fedora >= 19
* RedHat/CentOS >= 7


GNU Compiler Collection (gcc)
-----------------------------

You need to have ``gcc`` installed. We recommend installing it via your package manager.

For example on Debian/Ubuntu:

.. code:: sh

    apt install gcc


On CentOS/RedHat:

.. code:: sh

    yum -y install gcc


.. _native-install:

Install from Package
--------------------

The simplest way to install SCT is to use an in-place, static version of a tested package release. If you do not have any special circumstances, we recommend using this installation method.

First, navigate to the `latest release <https://github.com/spinalcordtoolbox/spinalcordtoolbox/releases>`__, curently |version|, then download the install script for SCT (:substitution-code:`install_sct-|version|_linux.sh`). Major changes to each release are listed in the :doc:`/dev_section/CHANGES`.

Once you have downloaded SCT, open a new Terminal in the location of the downloaded script, then launch the installer using the ``bash`` command. For example, if the script was downloaded to `Downloads/`, then you would run:

.. code-block:: sh
    :substitutions:

    cd ~/Downloads
    bash install_sct-|version|_linux.sh


.. _github-install:

Install from GitHub
-------------------

If you wish to benefit from the cutting-edge version of SCT, or if you wish to contribute to or test changes to the code, we recommend you install SCT using this method.

#. Retrieve the SCT code

    Clone the repository and hop inside:

    .. code:: sh

        git clone https://github.com/spinalcordtoolbox/spinalcordtoolbox

        cd spinalcordtoolbox

#. (Optional) Checkout the revision of interest, if different from `master`:

    .. code:: sh

        git checkout <revision_of_interest>

#. Run the installer and follow the instructions

    .. code:: sh

        ./install_sct

.. _docker-install-linux:

Install within Docker
---------------------

`Docker <https://www.docker.com/what-container/>`__ is a portable container platform.

In the context of SCT, it can be used to test SCT in a specific OS environment; this is much faster than running a fully fledged virtual machine.

Basic Installation (No GUI)
***************************

First, `install Docker <https://docs.docker.com/engine/install/#server>`__. Be sure to install from your distribution's repository.

.. note::
    Docker Desktop for Linux is not recommended if you intend to use the GUI.
    Instead install the `Docker Server Engine <https://docs.docker.com/engine/install/#server>`__, which is separate from the Docker Desktop Engine.
    For example on Ubuntu/Debian, follow the instructions for installing Docker from the `apt repository <https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository>`__.

By default, Docker commands require the use of ``sudo`` for additional permissions. If you want to run Docker commands without needing to add ``sudo``, please follow `these instructions <https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user>`__ to create a Unix group called ``docker``, then add your user account to it.

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

Follow the example below to create an OS-specific SCT installation (in this case, for Ubuntu 22.04), replacing 7.3 with the version of your choice:

.. code:: bash

    # Pull the Docker image for Ubuntu 22.04
    sudo docker pull ubuntu:22.04
    # Launch interactive mode (command-line inside container)
    sudo docker run -it ubuntu:22.04
    # Now inside Docker container, install SCT dependencies
    apt update
    apt install git bzip2 curl gcc git libdbus-1-3 libgl1-mesa-glx libglib2.0-0 libxkbcommon-x11-0 libxrender1
    # Note for above: libdbus-1-3, libgl1-mesa-glx, libglib2.0-0, libxkbcommon-x11-0, libxrender1 are required by PyQt
    # Install SCT (you can change 7.3 for the version of your choice)
    git clone --branch 7.3 https://github.com/spinalcordtoolbox/spinalcordtoolbox.git sct
    cd sct
    ./install_sct -iy
    source /root/.bashrc
    # Test SCT
    sct_testing
    # Save the state of the container as a docker image.
    # Back on the Host machine, open a new terminal and run:
    sudo docker ps -a  # list all containers (to find out the container ID)
    # specify the ID, and also choose a name to use for the docker image, such as "sct_v7.3"
    sudo docker commit <CONTAINER_ID> <IMAGE_NAME>/ubuntu:ubuntu22.04

Enabling GUI Scripts
********************

In order to run scripts with GUI you need to allow X11 redirection.
First, save your Docker image if you haven't already done so:

#. Open another Terminal
#. List current docker images

    .. code:: bash

        sudo docker ps -a

#. If you haven't already, save the container as a new image

    .. code:: bash

        sudo docker commit <CONTAINER_ID> <IMAGE_NAME>/<IMAGE_SPEC>

Then, to forward the X11 server:

.. note::

    The following instructions have been tested with Xorg and xWayland.

    Set up may vary if you are using a different X11 server.

#. Install ``xauth`` and ``xhost`` on the host machine, if not already installed:

    For example on Debian/Ubuntu:

    .. code:: bash

        sudo apt install xauth x11-xserver-utils

#. Permit docker access to the X11 Server

   If you are hosting the container from the local machine:

    .. code:: bash

        xhost +local:docker

#. In your Terminal window, run:

    .. code:: bash

        sudo docker run -it --rm --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix <IMAGE_NAME>/<IMAGE_SPEC>``

You can test whether GUI scripts are available by running the following command in your Docker container:

    .. code:: bash

        sct_check_dependencies

You should see two green ``[OK]`` symbols at the bottom of the report for "PyQT" and "matplotlib" checks, which represent the GUI features provided by SCT are now available.

.. _pip-install:

**[EXPERIMENTAL]** Install as a ``pip`` Package
-----------------------------------------------

You should only install SCT this way if you need to access the internal functions of the package for use in a Python environment. As well, doing so comes with some caveats:

- The installation is done in-place, so the folder containing SCT must be kept around and in the same place it was originally.
- In order to ensure coexistence with other packages, the dependency specifications are loosened. As a result, it is much more likely that you will be running a combination that has not been tested, which may introduce unpredicable bugs or crashing.

If the installation fails, or you run into errors, please report a bug indicating the dependency versions retrieved using "sct_check_dependencies", and try again with a clean ``pip`` installation/environment.

#. [Optional] `Activate <https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#activate-a-virtual-environment>`__ the ``virtualenv`` environment you want to install SCT within.

#. Clone the current SCT repository and enter it.

    .. code:: sh

        git clone https://github.com/spinalcordtoolbox/spinalcordtoolbox

        cd spinalcordtoolbox

#. Checkout the revision of interest, if different from ``master``:

    .. code:: sh

        git checkout <revision_of_interest>

#. Install ``numpy``:

    .. code:: sh

        pip install numpy

#. Install SCT using ``pip``:

    If you're installing within a ``virtualenv``:

    .. code:: sh

        pip install -e .

    Otherwise (you want SCT available in your base environment):

    .. code:: sh

        pip install --user -e .
