.. _sct_deepseg_sc: 

sct_deepseg_sc
==============

Command-line usage
------------------

.. argparse::
   :ref: spinalcordtoolbox.scripts.sct_deepseg_sc.get_parser
   :prog: sct_deepseg_sc
   :markdownhelp:


Algorithm details
-----------------

As its name suggests, :ref:`sct_deepseg_sc` is based on deep learning. It is a newer algorithm, having been introduced to SCT in 2018. The steps of the algorithm are as follows:

1. Spinal cord detection
************************

First, a convolutional neural network is used to generate a probablistic heatmap for the location of the spinal cord.

2. Centerline detection
***********************

The heatmap is then fed into the `OptiC <https://doi.org/10.1007/978-3-319-66185-8_80>`__ algorithm to detect the spinal cord centerline.

3. Patch extraction
*******************

The spinal cord centerline is used to extract a patch from the image. (This is done to exclude regions that we are certain do not contain the spinal cord.)

4. Segmentation
***************

Lastly, a second convolutional neural network is applied to the extracted patch to segment the spinal cord.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord-segmentation/sct_deepseg_sc_steps.png
   :align: center
   :figwidth: 65%

Fixing a failed ``sct_deepseg_sc`` segmentation
-----------------------------------------------

Due to contrast variations in MR imaging protocols, the contrast between the spinal cord and the cerebro-spinal fluid (CSF) can differ between MR volumes. Therefore, the segmentation method may fail sometimes in presence of artifacts, low contrast, etc.

You have several options if the segmentation fails:

- Change the kernel size from 2D to 3D using the ``-kernel`` argument.
- Change the centerline detection method using the ``-centerline`` argument.
- Try the newest segmentation method that is robust to many contrasts: (:ref:`sct_deepseg`)
- Try the legacy segmentation method based on mesh propagation: (:ref:`sct_propseg`)
- Check the specialized models in :ref:`sct_deepseg` to see if one fits your use case.
- Manually correct the segmentation.
- Ask for help on the `SCT forum <https://forum.spinalcordmri.org/c/sct/8>`__.