Transforming the template using warping fields
##############################################

Once we have the warping field, we can use it to warp the entire template to the MT space (including vertebral levels, WM/GM atlas, and more).

.. code:: sh

   sct_warp_template -d mt1.nii.gz -w warp_template2mt.nii.gz -a 0 -qc ~/qc_singleSubj

:Input arguments:
   - ``-d`` : Destination image the template will be warped to.
   - ``-w`` : Warping field (template space to anatomical space).
   - ``-a`` : Because ``-a 0`` is specified, the white and gray matter atlas will not be warped to save time. For more details on ``-a 1``, we recommend the :ref:`atlas-based-analysis` tutorial.
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.

:Output:
   - ``label/template/`` : This directory contains the entirety of the PAM50 template, transformed into the MT space.
   - ``label/atlas/`` : This direct contains 36 NIFTI volumes for WM/GM tracts, transformed into the MT space.

Once the command has finished, at the bottom of your terminal there will be instructions for inspecting the results using :ref:`Quality Control (QC) <qc>` reports. Optionally, If you have :ref:`fsleyes-instructions` installed, a ``fsleyes`` command will printed as well.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/registering-additional-contrasts/io-sct_warp_template.png
   :align: center

   Input/output images for :ref:`sct_warp_template`