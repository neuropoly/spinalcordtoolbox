Motion correction for fMRI images
#################################

Now that we have a mask highlighting the spinal cord, we can apply motion correction to the volumes of the fMRI data.

The motion correction algorithm
-------------------------------

SCT features a complex motion correction algorithm, which is inspired by `[Xu et al., Neuroimage 2013] <https://doi.org/10.1016/j.neuroimage.2012.11.014>`__. The key aspects of this algorithm are as follows:

* **SliceReg:** Slice-wise registration regularized along the Z direction (based on the function antsSliceRegularizedRegistration from ANTs, and described in `[De Leener et al., Neuroimage 2017] <https://doi.org/10.1016/j.neuroimage.2016.10.009>`__).
* **Grouping:** The algorithm performs group-wise registration between groups of successive dMRI volumes. If your data has a very low signal-to-noise ratio (SNR), you can use the flag ``-g`` to increase the number of successive images that are averaged into a group in order to have sufficient SNR to estimate a reliable transformation.
* **Iterative average:** After registering a new group to the target image (which is usually the first DWI group), the target is averaged with the newly registered group in order to increase the SNR of the target image.
* **Outlier detection:** If a detected transformation is too large, it is ignored and the previous transformation is used instead.
* **Masking:** You can use the flag ``-m`` to provide a spinal cord mask, in order to estimate motion of the cord while ignoring the rest of the tissue.

Applying the algorithm
----------------------

To apply the algorithm, we use the :ref:`sct_fmri_moco` command:


.. code::

   sct_fmri_moco -i fmri.nii.gz -m mask_fmri.nii.gz -qc ~/qc_singleSubj -qc-seg t2_seg_reg.nii.gz

:Input arguments:
   - ``-i`` : The input fMRI image.
   - ``-m`` : A mask used to limit the voxels considered by the motion correction algorithm.
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.
   - ``-qc-seg`` :  Segmentation of spinal cord to improve cropping in QC report.

:Output files/folders:
   - ``fmri_moco.nii.gz`` : The motion-corrected 4D fMRI image.
   - ``fmri_moco_mean.nii.gz`` : The time-average of the motion-corrected 3D volumes of the fMRI image.
   - ``moco_params.tsv`` : A text file that provides a simplified overview of the motion correction, to be used for quality control. It contains the slicewise average of the magnitude of the axial (X-Y) translations for each 3D volume. (In reality, though, each slice of each volume will have had a different translation applied to it.)
   - ``moco_params_x.nii.gz`` : A 4D image with dimensions ``[1, 1, z, t]``. Each voxel contains the ``x`` translation corresponding to each ``z`` slice across each ``t`` volume.
   - ``moco_params_y.nii.gz`` : A 4D image with dimensions ``[1, 1, z, t]``. Each voxel contains the ``y`` translation corresponding to each ``z`` slice across each ``t`` volume.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/processing-fmri-data/io-sct_fmri_moco.png
   :align: center


Checking QC for Temporal Signal-to-Noise Ratio (tSNR)
-----------------------------------------------------

Here, we can generate a `tSNR` map for the original and motion-corrected fMRI data. This map is a measure of the signal-to-noise ratio in the temporal domain. The higher the tSNR, the better the quality of the data.

Then, we can use the :ref:`sct_qc` command to compare the tSNR maps before and after motion correction to see how motion correction can improve the quality of the data.

.. code::

   # Cord segmentation on motion-corrected averaged time series
   sct_deepseg spinalcord -i fmri_moco_mean.nii.gz -qc ~/qc_singleSubj
   # TSNR before/after motion correction with QC report
   sct_fmri_compute_tsnr -i fmri.nii.gz
   sct_fmri_compute_tsnr -i fmri_moco.nii.gz
   sct_qc -i fmri_tsnr.nii.gz -d fmri_moco_tsnr.nii.gz -s fmri_moco_mean_seg.nii.gz -p sct_fmri_compute_tsnr -qc ~/qc_singleSubj

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/processing-fmri-data/tsnr_qc.png
   :align: center