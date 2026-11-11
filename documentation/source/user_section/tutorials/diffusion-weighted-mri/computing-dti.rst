Computing DTI for motion corrected dMRI data
############################################

Here, we compute the diffusion tensor metric images. SCT relies on the excellent ``dipy`` library for computing dMRI metrics `[Garyfallidis et al., Front Neuroinform 2014] <https://doi.org/10.3389/fninf.2014.00008>`_.

.. code::

   sct_dmri_compute_dti -i dmri_moco.nii.gz -bval bvals.txt -bvec bvecs.txt

:Input arguments:
   - ``-i`` : The input dMRI image.
   - ``-bval`` : A text file containing a b-value for each volume in the dMRI image, indicating the diffusion weightings for each of the volumes in the dMRI image.
   - ``-bvec`` : A text file with three lines, each containing a value for each volume in the input image. Together, the the three sets of values represent the ``(x, y, z)`` coordinates of the b-vectors, which indicate the direction of the diffusion encoding for each volume of the dMRI image.

.. note::

   You can also supply the ``-method restore`` option to estimate the tensors using **"RESTORE: robust estimation of tensors by outlier rejection"** `[Chang, Magn Reson Med 2005] <https://doi.org/10.1002/mrm.20426>`_.

:Output files/folders:
   - ``dti_FA.nii.gz`` : Fractional anisotropy (FA) diffusion tensor image.
   - ``dti_AD.nii.gz`` : Axial diffusivity (AD) diffusion tensor image.
   - ``dti_MD.nii.gz`` : Mean diffusivity (MD) diffusion tensor image.
   - ``dti_RD.nii.gz`` : Radial diffusivity (RD) diffusion tensor image.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/processing-dmri-data/io-sct_dmri_compute_dti.png
   :align: center

Now that we have the diffusion tensor images, we can move on to extracting DTI metrics from specific regions of the image.
