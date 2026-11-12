.. _generate-necessary-inputs:

Generate the necessary input files
##################################

The :ref:`sct_compute_compression` command uses the shape metric of the spinal cord at the level of compression and computes a ratio with non-compressed levels above and below the compression site.

To use :ref:`sct_compute_compression`, we need the 3 following input files:

1. Spinal cord segmentation
2. Vertebral labels
3. Compression labels (``t2_compressed_label-compression.nii.gz``)

These input files can be generated using the commands below. However, if you would like to save some time, you can also acquire the generated files directly by downloading :sct_tutorial_data:`data_normalizing-morphometrics-compression.zip`.


1. Spinal cord segmentation
----------------------------

.. code:: sh

   sct_deepseg spinalcord -i t2_compressed.nii.gz -qc ~/qc_singleSubj

:Input arguments:
   - ``spinalcord``: Task to perform. Here, we are using ``spinalcord`` to segment the spinal cord. This task is contrast-agnostic, meaning it can be used on any type of image (T1, T2, T2*, etc.)
   - ``-i`` : Input image.
   - ``-qc`` : Directory for Quality Control reporting.


:Output files/folders:
   - ``t2_compressed_seg.nii.gz`` : Spinal cord segmentation.


2. Label the vertebral levels
-----------------------------
.. code:: sh

   sct_label_vertebrae -i t2_compressed.nii.gz -s t2_compressed_seg.nii.gz -c t2 -qc ~/qc_singleSubj

:Input arguments:
   - ``-i`` : The input image file.
   - ``-s`` : Spinal cord segmentation.
   - ``-c`` : Contrast of the input image.
   - ``-qc`` : Directory for Quality Control reporting.

:Output files/folders:
   - ``t2_compressed_seg_labeled.nii.gz`` : Spinal cord vertebral labels.


3a. Label spinal cord compression sites (interactive viewer)
------------------------------------------------------------

1. Open the image in FSLeyes (or your favorite viewer).

.. code:: sh

   fsleyes t2_compressed.nii.gz &

2. If clinical data with information about compression level(s) is available, look at it to determine how many compressions a subject has and at which levels the compressions are located.
3. Locate the first compression site in the sagittal view. (The location does not need to be precise, as we will fine-tune the selection in the next step.) All 4 compression sites are indicated here.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/sb/4158-add-tutorial-sct-compute-compression/spinalcord-compresssion-norm/localizing_compression_sag.png
   :align: center

4. In the axial view, toggle to the maximum compressed slice around the compression.
5. Click on alt+E (Linux/Windows) or option+E (macOS) to open edit mode. Select size 1 and click on the pencil.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/sb/4158-add-tutorial-sct-compute-compression/spinalcord-compresssion-norm/edit_mode.png
   :align: center

6. Create an empty mask alt+N (Linux/Windows) or cmd+N (macOS).

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/sb/4158-add-tutorial-sct-compute-compression/spinalcord-compresssion-norm/create_mask.png
   :align: center

7. Place the label at the center of the spinal cord of the axial image.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/sb/4158-add-tutorial-sct-compute-compression/spinalcord-compresssion-norm/labeling_compression.png
   :align: center

8. In the same mask, repeat steps 3, 4 and 7 for the remaining three compression sites.
9. Save with the filename ``t2_compressed_labels-compression.nii.gz`` and quit.

If you need to label multiple patients, you can use the ``manual_correction.py`` script from the `manual-correction repository <https://github.com/spinalcordtoolbox/manual-correction>`__; see the example `here <https://github.com/spinalcordtoolbox/manual-correction/wiki#manual-labeling-of-spinal-cord-compression>`__.


3b. Label spinal cord compression sites (direct coordinates)
------------------------------------------------------------

Alternatively, if you know the precise coordinates of the compression sites already, you can easily reproduce the label file using the ``-create`` argument of :ref:`sct_label_utils`:

.. code:: sh

   sct_label_utils -i t2_compressed.nii.gz -create 30,152,99,1.0:30,156,118,1.0:30,157,140,1.0:31,160,159,1.0 -o t2_compressed_labels-compression.nii.gz
