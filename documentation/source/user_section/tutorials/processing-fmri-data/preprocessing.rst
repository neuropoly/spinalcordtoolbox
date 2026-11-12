Preprocessing steps to highlight the spinal cord
################################################

Similarly to the preprocessing found in the :ref:`dMRI tutorial <preprocessing-dmri>`, it is useful to highlight the spinal cord region using a mask prior to motion correction, as it helps to improve accuracy and speed up processing.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/processing-fmri-data/preprocessing.png
   :align: center


Computing a mean image
----------------------

First we compute the mean of the 4D fMRI data to obtain a coarse 3D approximation. This step is necessary because the relevant SCT tools are designed for individual 3D volumes, rather than 4D images.

.. code::

   sct_maths -i fmri.nii.gz -mean t -o fmri_mean.nii.gz

:Input arguments:
   - ``-i`` : The input image.
   - ``-mean`` : The dimension to compute the mean across. In this case, :ref:`sct_maths` will assume that the fMRI image is a 4D stack of 3D volumes, with dimension ``[x, y, z, t]``. Therefore, ``-mean t`` will average the 3D volumes across the temporal dimension.
   - ``-o``: The filename of the output image.

:Output files/folders:
   - ``fmri_mean.nii.gz`` : A 3D image containing the mean of the individual volumes in the 4D dMRI image.


Generating a spinal cord segmentation
-------------------------------------

Due to the low contrast between spinal cord and cerebrospinal fluid of fMRI data, it is difficult to directly obtain a spinal cord segmentation for fMRI data using :ref:`sct_deepseg`. So, as a workaround, we will instead obtain a spinal cord segmentation for another contrast (T2), then transform it to the space of the fMRI data.


Generating a T2 segmentation
============================

.. code::

   cd ../t2
   sct_deepseg spinalcord -i t2.nii.gz

:Input arguments:
   - ``spinalcord``: Task to perform. Here, we are using ``spinalcord`` to segment the spinal cord. This task is contrast-agnostic, meaning it can be used on any type of image (T1, T2, T2*, etc.)
   - ``-i`` : Input image

:Output files/folders:
   - ``t2_seg.nii.gz`` : 3D binary mask of the segmented spinal cord


Transforming the T2 segmentation to the fMRI space
==================================================

Since the segmentation image will only be used to coarsely highlight the region of interest, a complex transformation is not necessary, so we supply the ``-identity`` to :ref:`sct_register_multimodal` to speed up processing.

.. code::

   cd ../fmri
   sct_register_multimodal -i ../t2/t2_seg.nii.gz -d fmri_mean.nii.gz -identity 1


:Input arguments:
   - ``-i`` : Input image
   - ``-d`` : Destination image
   - ``-identity`` : Supplying this option will skip registration optimization (e.g. translations, rotations, deformations) and will only rely on the qform (from the NIfTI header) of the source and destination images. Use this option if you wish to put the source image into the space of the destination image (i.e. match dimension, resolution and orientation).

:Output files/folders:
   - ``t2_seg_reg.nii.gz`` : The T2 segmentation, transformed to the space of the fMRI mean image. This file is what will be used to create the spinal cord mask for fMRI motion correction.
   - ``fmri_mean_reg.nii.gz`` : The fMRI mean image, transformed to the space of the T2 segmentation.
   - ``warp_t2_seg2fmri_mean.nii.gz`` : The warping field to transform the T2 segmentation to the fMRI space.
   - ``warp_fmri_mean2t2_seg.nii.gz`` : The warping field to transform the fMRI mean image to the T2 space.

Creating a mask around the spinal cord
--------------------------------------

Now that we have a spinal cord segmentation in the space of the fMRI data, we can use it to create a mask around the fMRI spinal cord.

.. code::

   sct_create_mask -i fmri.nii.gz -p centerline,t2_seg_reg.nii.gz -size 35mm -f cylinder

:Input arguments:
   - ``-i`` : The input image to create the mask from.
   - ``-p`` : The process used to position the mask. The ``centerline`` process will compute the center of mass for each slice of ``t2_seg_reg.nii.gz``, then use those locations to center of the mask at each slice.
   - ``-size``: The diameter of the mask.
   - ``-f``: Shape of the mask. This is ``cylinder`` by default, but other shapes can be specified instead.

:Output files/folders:
   - ``mask_fmri.nii.gz`` : An imagine containing a mask surrounding the spinal cord.


..
   FIXME: Find a place in the tutorials to insert these commands. Perhaps rework the
   tutorial to use the `sc_epi` seg rather than the warped T2 seg?

   .. code::

      # Crop extraneous tissue using the t2-based mask generated earlier
      sct_crop_image -i fmri_moco_mean.nii.gz -m mask_fmri.nii.gz -b 0
      # Segment the cord using the cropped image
      sct_deepseg sc_epi -i fmri_moco_mean_crop.nii.gz -qc ~/qc_singleSubj
