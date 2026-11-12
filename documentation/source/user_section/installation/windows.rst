.. _windows_installation:

************************
Installation for Windows
************************

We currently provide 3 different ways to install SCT on Windows machines:

- :ref:`Native Windows installation (recommended) <native-installation>`
- :ref:`Windows Subsystem for Linux (WSL) <wsl-installation>`
- :ref:`Docker <docker-install-windows>`


-----


.. _native-installation:

Native Windows installation (recommended)
-----------------------------------------

This set of instructions will show you how to install SCT directly on Windows.

.. note::

   This method was first introduced to SCT in April 2022 as part of the Version 5.6 release. Previous versions of SCT required the use of :ref:`Windows Subsystem for Linux (WSL) <wsl-installation>` or :ref:`Docker <docker-install-windows>` in order to support Windows.

   Because this method is new to SCT, we would greatly appreciate hearing any feedback you may have about your experiences using SCT on Windows. If you encounter any issues, or have any questions or concerns, feel free to post on the `Spinalcordmri.org forums <https://forum.spinalcordmri.org/c/sct/8>`__, and SCT's development team will be happy to help.

1. Installing Prerequisites
***************************

SCT depends on two pieces of software that must be set up prior to the installation of SCT.

Visual C++ 2019 runtime
^^^^^^^^^^^^^^^^^^^^^^^

SCT depends on `onnxruntime <https://onnxruntime.ai/docs/install/#requirements>`__, which in turn depends on the `Visual C++ 2019 runtime <https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170>`__.

1. It is likely that you already have this software installed, as many Windows applications rely on this software. You can check for "Microsoft Visual C++ 2015-2022 Redistributable" under the "Apps & Features" section of Windows Settings.

2. If you do not have this software installed, you will typically want to install the "X64" version (``https://aka.ms/vs/17/release/vc_redist.x64.exe``) from `this page <https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#visual-studio-2015-2017-2019-and-2022>`__.


Git for Windows
^^^^^^^^^^^^^^^

The easiest way to try out different versions of SCT is using Git.

1. Download Git for Windows from `the Git download page <https://git-scm.com/download/win>`__.

2. Run the installer.

   - You can click "Next" for most of the options in the installer.
   - However, on the "Adjusting your PATH environment" page, we strongly recommend that you choose the "Use Git and optional Unix tools from the Command Prompt". Selecting this option will provide you with useful Unix utilities such as ``bash``, ``cd``, ``ls``, and more that combine nicely with SCT's command-line tools. In particular, ``bash`` will allow :ref:`sct_run_batch` to execute bash scripts for batch processing of subjects.
   - **Note:** If you prefer, you may instead choose to :ref:`install Cygwin<installing_cygwin>` (rather than selecting the "Use Git and optional Unix tools" option) in order to gain access to these same Unix utilities.

3. After the installation has finished, open your Start Menu and type Command Prompt, then run it. In the Command Prompt window, type ``git --version`` and press enter.

   (Make sure that you see the text ``git version <number>`` before continuing.)


2. Installing SCT
*****************

1. Navigate to the `latest release <https://github.com/spinalcordtoolbox/spinalcordtoolbox/releases>`__, curently |version|, then download the install script for SCT (:substitution-code:`install_sct-|version|_win.bat`). Major changes to each release are listed in the :doc:`/dev_section/CHANGES`.

2. Run the script by double-clicking it. The script will fetch the SCT source code, then install the `spinalcordtoolbox` package into a Miniforge environment for you.

3. Once the installer finishes, follow the instructions given at the end of the Command Prompt window, which will instruct you to create the SCT_DIR variable, and to add the SCT installation folder to the PATH variable.

4. Finally, in the Command Prompt window, type :ref:`sct_check_dependencies` and press enter. Make sure that you see a status report containing all "OKs" before continuing.

5. You are now free to use SCT's command line tools to process your data. If you would like to learn how to use SCT, we recommend starting with SCT's :ref:`tutorials` pages.

.. _installing_cygwin:

3. (Optional) Installing Cygwin
*******************************

.. note:: You do not need to install Cygwin if you already selected the "Use Git and optional Unix tools from the Command Prompt" option during the Git installation step.

Cygwin is a collection of useful Unix utilities such as ``bash``, ``cd``, ``ls``, and more that combine nicely with SCT's command-line tools. In particular, ``bash`` will allow :ref:`sct_run_batch` to execute bash scripts for batch processing of subjects.

1. Download the Cygwin installer from `the Cygwin installation page <https://www.cygwin.com/install.html>`__.

2. Run the installer. (You can click "Next" for every section of the installer, as the default settings are sufficient.)

3. After the installer is finished, you will need to add Cygwin's programs to the PATH.

   - Open the Start Menu -> Type 'path' -> Open 'Edit environment variables for your account'
   - Under the section 'User variables for ____', highlight the 'Path' entry, then click the 'Edit...' button.
   - Click 'New', then copy and paste "``C:\cygwin64\bin``".
   - Finally, click "Ok" three times.

4. Finally, open your Start Menu and type Command Prompt, then run it. In the Command Prompt window, type ``cygcheck --version`` and press enter. Make sure that you see the text ``cygcheck (cygwin)`` before continuing.

   - Note: If you see a "not recognized" error, please repeat Step 3, making sure that the directory you added corresponds to the installation directory of Cygwin.


-----


.. _wsl-installation:

Windows Subsystem for Linux (WSL) installation
----------------------------------------------

Windows Subsystem for Linux (WSL) makes it possible to run native Linux programs on Windows 10. Here, WSL is used to install the Linux version of SCT within Windows (as opposed to the :ref:`native Windows version <native-installation>`).

Basic installation (No GUI)
***************************

#. Make sure that your version of Windows 10 is up to date before continuing.

   - In Windows, search for "System Information" and open the app. In the "Version" field, make sure that you are running "Build 19041" or higher.

   - Then, search for "Powershell" in your Start Menu, then right-click and "Run as administrator". Then run the following command:

     .. code::

        wsl --update

   - If this command is successful, then you can proceed to the next step. Otherwise, please try the following troubleshooting steps:

     - Make sure your version of Windows is up to date.
     - Make sure that you have sufficient administrative privileges for your Windows system.
     - If you cannot update Windows, then you can try the instructions from Microsoft's `"Manual installation steps for older versions of WSL" <https://docs.microsoft.com/en-us/windows/wsl/install-manual>`__ page.

#. Install an Ubuntu distribution in Windows Subsystem for Linux (WSL)

   - In Windows, search for "Powershell" in your Start Menu, then right-click and "Run as administrator".

   - In PowerShell, type the following command and press enter:

     .. code::

        wsl --install

   - After this command finishes, you will be prompted to restart your computer.

   - After restarting, the installation should automatically resume, and you will be able to create a user account inside Ubuntu by selecting a username and password.

#. Choose the WSL version (1/2).

   By default, Microsoft's instructions will create an Ubuntu environment using Version 2 of WSL. While version 2 has been tested to work with SCT, our development team tests more thoroughly using Version 1 of WSL (due to `better support from GitHub Actions <https://github.com/actions/runner-images/issues/50>`__).

   Because of this, we recommend that you convert the WSL2 Ubuntu environment to use WSL1 before continuing. To do this, first close Ubuntu, then re-open Powershell and run the following command:

   .. code::

      wsl --list --verbose

   If WSL installed correctly, you should see a list of installed distributions (as well as their WSL versions). Find the name of the distribution you just installed (which should be something like ``Ubuntu`` or ``Ubuntu-24.04``), then specify that name in the following command:

   .. code::

      wsl --set-version Ubuntu 1

   After you run this command, you can then run ``wsl --list --verbose`` again to check that the distribution has changed from WSL2 to WSL1.

#. Environment preparation

   Now that you have set up an Ubuntu environment with WSL, please open Ubuntu and run the following commands to install various packages that will be needed to install SCT.

   .. code-block:: sh

      sudo apt-get update && sudo apt-get upgrade -y
      sudo apt-get -y install gcc unzip python3-pip python3 psmisc net-tools git gfortran libjpeg-dev

#. Install SCT

   First, download SCT by running the following commands in Ubuntu:

   .. code-block:: sh

      cd ~
      git clone https://github.com/spinalcordtoolbox/spinalcordtoolbox.git
      cd spinalcordtoolbox

   To select a `specific release <https://github.com/spinalcordtoolbox/spinalcordtoolbox/releases>`__, replace ``X.Y`` below with the proper release number (latest release is |version|). If you prefer to use the development version, you can skip this step.

   .. code-block:: sh

      git checkout X.Y

   Install SCT:

   .. code:: sh

      ./install_sct -y

   .. note::

      At the end of this installation step, you may see the following warnings:

      .. code::

         Check if figure can be opened with matplotlib.......[FAIL] (Using non-GUI backend 'agg')
         Check if figure can be opened with PyQt.............[FAIL] ($DISPLAY not set on X11-supporting system)

      This is expected, because WSL does not come with the ability to display GUI programs by default. Later on in this page, there will be optional GUI settings you can configure for WSL to address these warnings.

   To complete the installation of SCT, run:

   .. code:: sh

      source ~/.bashrc

   You can now use SCT. Your local C drive is located under ``/mnt/c``. You can access it by running:

   .. code:: sh

      cd /mnt/c


WSL Installation with GUI (Optional)
************************************

If you would like to use SCT's GUI features, or if you would like to try FSLeyes within the same Ubuntu environment, first complete the previous "Basic Installation" section, then continue on to the steps below.

#. Download and install `VcXsrv <https://sourceforge.net/projects/vcxsrv/>`__, a program that makes it possible to run Linux GUI programs installed with WSL.

#. Run the newly installed ``XLaunch`` program, then click the following settings:

   - On the "Display settings" page, click "Next".
   - On the "Client startup" page, click "Next".
   - On the "Extra settings" page, check the "Disable access control" box, then click "Next".
   - Click "Finish", then click "Allow access" when prompted by Windows Firewall.
   - You should now see the X Server icon running in the bottom-right system tray in your taskbar.

   .. note::

      ``XLaunch`` must be running each time you wish to use GUI programs in WSL.

#. Next, run the following commands in Ubuntu, depending on the version of WSL you are using.

   WSL1:

   .. code::

      echo "export DISPLAY=localhost:0.0" >> ~/.bashrc
      echo "export LIBGL_ALWAYS_INDIRECT=0" >> ~/.bashrc
      source ~/.bashrc

   WSL2:

   .. code::

      echo "export DISPLAY=$(awk '/nameserver / {print $2; exit}' /etc/resolv.conf 2>/dev/null):0.0" >> ~/.bashrc
      echo "export LIBGL_ALWAYS_INDIRECT=0" >> ~/.bashrc
      source ~/.bashrc

#. Finally, run the :ref:`sct_check_dependencies` command in Ubuntu to verify that matplotlib and PyQt figures can be opened by SCT.

#. Optionally, you can install FSLeyes using the following commands:

   .. code::

      source ${SCT_DIR}/python/etc/profile.d/conda.sh
      conda create -c conda-forge -p ~/fsleyes_env fsleyes -y
      sudo ln -s ~/fsleyes_env/bin/fsleyes /usr/local/bin/fsleyes

   These instructions will install FSLeyes into a fresh ``conda`` environment, then create a link to FSLeyes so that you can use the ``fsleyes`` command without having to activate the conda environment each time.


-----


.. _docker-install-windows:

Docker installation
-------------------

`Docker <https://www.docker.com/what-container/>`__ is a portable (Linux, macOS, Windows) container platform.

In the context of SCT, it can be used to test SCT in a specific OS environment; this is much faster than running a fully fledged virtual machine.

Basic Installation (No GUI)
***************************

First, `install Docker Desktop <https://docs.docker.com/desktop/install/windows-install/>`__ using the WSL 2 backend. Then, follow the example below to create an OS-specific SCT installation.

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

First, launch Docker Desktop, then open up a new Powershell or Command Prompt window and run the commands below, replacing 7.3 with the version of your choice:

.. code:: bash

   # Start from Powershell or the Command Prompt
   docker pull ubuntu:22.04
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
First, save your Docker image if you have not already done so:

1. Open another Terminal

2. List current docker images

   .. code:: bash

      docker ps -a

3. Save container as new image

   .. code:: bash

      docker commit <CONTAINER_ID> <IMAGE_NAME>/<IMAGE_SPEC>

4. Install `VcXsrv <https://sourceforge.net/projects/vcxsrv/>`__.

5. Launch an X11 Server with XLaunch

   - Run XLaunch, which should have been installed by default.
   - Check 'Multiple Windows' and set the **display number** to ``0``, which you will need later. (The default display number ``-1`` will automatically detect the display number, unless you are running a setup with multiple monitors it will typically use ``0``)
   - Then, you can click Next, select 'Start no Client' then click Next
   - **Uncheck** 'Native opengl' and **check** 'Disable Access Control' then click Next, then click Finish.

6. Determine the IPv4 address of the virtual Ethernet Adapter by running 'ipconfig' in Powershell or the Command Prompt, then looking at the ``Ethernet adapter vEthernet (WSL)`` entry.

7. In your Terminal window, run the following command, filling in the IP address, display number, and image name noted earlier:
   
   .. code:: bash 
   
      docker run -it --rm -e DISPLAY=<IPv4_ADDRESS>:<DISPLAY_NUMBER> -e XDG_RUNTIME_DIR=/tmp/runtime-root <IMAGE_NAME>/<IMAGE_SPEC>


8. You can test whether GUI scripts are available by running the following command in your Docker container:

   .. code:: bash

      mkdir /tmp/runtime-root
      sct_check_dependencies

   You should see two green ``[OK]`` symbols at the bottom of the report for "PyQT" and "matplotlib" checks, which represent the GUI features provided by SCT.
