
.. _command-line-tools:

Command-Line Tools
##################

The command-line tools are named ``sct_*``; to see all the commands available from SCT, start a new Terminal and type ``sct`` then press "tab".

Each command-line tool has a dedicated documentation page where you will find its "help text", with each page linked below. To quickly search the help text of all of the tools at once, please navigate to the :ref:`all_tools` page and type Ctrl+F to search.


Segmentation
============

- :ref:`sct_create_mask` - Create mask along z direction.
- :ref:`sct_deepseg` - Segment anatomical structures or pathologies using deep learning models created with different frameworks (`ivadomed <https://ivadomed.org>`__,  `nnUNet <https://github.com/MIC-DKFZ/nnUNet>`__, `monai <https://project-monai.github.io/>`__).
- :ref:`sct_deepseg_gm` - Segment spinal cord gray matter using deep learning.
- :ref:`sct_get_centerline` - Extracts the spinal cord centerline.
- :ref:`sct_propseg` - Segment spinal cord using propagation of deformation model (PropSeg).
- :ref:`sct_deepseg_sc` - Segment spinal cord using deep learning. [DEPRECATED] Please use `sct_deepseg spinalcord` instead.
- :ref:`sct_deepseg_lesion` - Segment multiple sclerosis lesions. [DEPRECATED] Please use `sct_deepseg ms_lesion` instead.

Segmentation Analysis
=====================

- :ref:`sct_analyze_lesion` - Compute statistics on segmented lesions.
- :ref:`sct_compute_hausdorff_distance` - Compute the Hausdorff's distance between two binary images.
- :ref:`sct_compute_compression` - Compute spinal cord compression morphometrics.
- :ref:`sct_detect_compression` - Predict compression probability using spinal cord morphometrics.
- :ref:`sct_dice_coefficient` - Compute the Dice Coefficient to estimate overlap between two binary images.
- :ref:`sct_process_segmentation` - Perform various types of processing from the spinal cord segmentation.

Labeling
========

- :ref:`sct_detect_pmj` - Detection of the Ponto-Medullary Junction (PMJ).
- :ref:`sct_label_vertebrae` - Label vertebral levels
- :ref:`sct_label_utils` - Collection of tools to create or process labels

Registration
============

- :ref:`sct_apply_transfo` - Apply transformations.
- :ref:`sct_get_centerline` - Reconstruct spinal cord centerline.
- :ref:`sct_register_multimodal` - Register two images together (non-linear, constrained in axial plane)
- :ref:`sct_register_to_template` - Register an image with an anatomical template (eg. the `PAM50 template <https://doi.org/10.1016/j.neuroimage.2017.10.041>`_).
- :ref:`sct_straighten_spinalcord` - Straighten spinal cord from centerline
- :ref:`sct_warp_template` - Warps the template and all atlases to a destination image.

Diffusion MRI
=============

- :ref:`sct_dmri_compute_bvalue` - Calculate b-value (in mm^2/s).
- :ref:`sct_dmri_concat_bvals` - Concatenate bval files in time.
- :ref:`sct_dmri_concat_bvecs` - Concatenate bvec files in time.
- :ref:`sct_dmri_compute_dti` - Compute Diffusion Tensor Images (DTI) using `dipy <https://dipy.org/>`_.
- :ref:`sct_dmri_denoise_patch2self` - Denoise images using `dipy <https://dipy.org/>`_.
- :ref:`sct_dmri_display_bvecs` - Display scatter plot of gradient directions from bvecs file.
- :ref:`sct_dmri_moco` - Slice-wise motion correction of DWI data.
- :ref:`sct_dmri_separate_b0_and_dwi` - Separate b=0 and DW images from diffusion dataset.
- :ref:`sct_dmri_transpose_bvecs` - Transpose bvecs file.

Magnetization transfer
======================

- :ref:`sct_compute_mtr` - Compute magnetization transfer ratio (MTR).
- :ref:`sct_compute_mtsat` - Compute MTsat and T1map `[Helms et al. Magn Reson Med 2008] <https://doi.org/10.1002/mrm.21732>`_.

Functional MRI
==============

- :ref:`sct_fmri_compute_tsnr` - Compute the temporal signal-to-noise ratio from fMRI nifti files.
- :ref:`sct_fmri_moco` - Correct fMRI data for motion.

Metric processing
=================

- :ref:`sct_analyze_texture` - Extraction of grey level co-occurence matrix (GLCM) texture features from an image within a given mask.
- :ref:`sct_extract_metric` - Estimate metric value within tracts, taking into account partial volume effect.

Image manipulation
==================

- :ref:`sct_convert` - Convert image file to another type.
- :ref:`sct_crop_image` - Tools to crop an image, either via command line or via a Graphical User Interface (GUI).
- :ref:`sct_denoising_onlm` - Utility function to denoise images.
- :ref:`sct_flatten_sagittal` - Flatten the spinal cord in the sagittal plane (to make nice pictures).
- :ref:`sct_image` - Performs various operations on images (split, pad, etc.).
- :ref:`sct_maths` - Performs mathematical operations on images (threshold, smooth, etc.).
- :ref:`sct_merge_images` - Merge images to the same space.
- :ref:`sct_resample` - Anisotropic resampling of 3D or 4D data.
- :ref:`sct_smooth_spinalcord` - Smooth the spinal cord along its centerline.

Miscellaneous
=============

- :ref:`sct_compute_ernst_angle` - Compute Ernst angle.
- :ref:`sct_compute_snr` - Compute SNR using methods described in `[Dietrich et al. JMRI 2007]
  <https://doi.org/10.1002/jmri.20969>`_.
- :ref:`sct_download_data` - Download binaries from the web.
- :ref:`sct_qc` - Generate Quality Control (QC) report following SCT processing.
- :ref:`sct_run_batch` - Wrapper to processing scripts, which loops across subjects.

System tools
============

- :ref:`sct_check_dependencies` - Check installation and compatibility of SCT.
- :ref:`sct_version` - Display SCT version.


.. Note: The toctree below is required by Sphinx for the sidebar. However, the automatically generated sidebar isn't ideal, because ":maxdepth: 2" shows too many sections, but ":maxdepth: 1" doesn't show enough. To get around this, we set the toctree as `:hidden:`, then manually create a secondary TOC using bullet point lists (see above). This manual method produces e a good-looking hybrid of both of the 'max-depth' options.

.. Note 2: Both the hidden toctree (below) and the manual TOC (above) should be updated together. Make sure to use short titles in each section's page (since these will automatically be shown in the sidebar). But, feel free to use longer titles in the manual TOC, where there is more space.


.. toctree::
   :hidden:
   :maxdepth: 2

   command-line/index/segmentation
   command-line/index/segmentation-analysis
   command-line/index/labeling
   command-line/index/registration
   command-line/index/diffusion-mri
   command-line/index/magnetization-transfer
   command-line/index/functional-mri
   command-line/index/metric-processing
   command-line/index/image-manipulation
   command-line/index/miscellaneous
   command-line/index/system
   command-line/all_tools
