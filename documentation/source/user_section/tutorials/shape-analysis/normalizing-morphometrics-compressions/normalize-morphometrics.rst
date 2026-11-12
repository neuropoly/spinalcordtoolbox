.. _normalize-morphometrics:

Compute normalized morphometrics
################################

The :ref:`sct_compute_compression` function computes a ratio of the morphometric measures at the level(s) of compression with the measures at the levels above and below all compression sites.

Adding the option ``-normalize-hc`` normalizes the morphometric measures with a database of adult healthy participants before computing the ratio.


Compute ratio between **AP-diameter** at level of compression vs. above/below
-----------------------------------------------------------------------------

We will compute the ratio of the anteroposterior diameter (AP-diameter) at the level of compression and levels above and below all compresssions.
This is equivalent to the MSCC (maximum spinal cord compression) metric (`Miyanji et al., 2007 <https://doi.org/10.1148/radiol.2433060583>`_).

.. code:: sh

   sct_compute_compression -i t2_compressed_seg.nii.gz -vertfile t2_compressed_seg_labeled.nii.gz -l t2_compressed_labels-compression.nii.gz -metric diameter_AP -normalize-hc 0 -o ap_ratio.csv

:Input arguments:
   - ``-i`` : The input segmentation file.
   - ``-vertfile`` : Vertebral labeling file.
   - ``-l`` : Compression labels file.
   - ``-metric``: Metric to compute ratio: diameter_AP (default).
   - ``-normalize-hc``: Set to 1 to normalize the metrics using a database of healthy controls. Set to 0 to not normalize.
   - ``-o`` : The output CSV file.
:Output files/folders:
   - ``ap_ratio.csv`` : A file containing the ratio values for each. This file is partially replicated in the table below.


.. csv-table:: Anterposterior diameter ratio with levels above and below all compressions.
   :file: ap_ratio.csv
   :header-rows: 1

:Legend:
   - **diameter_AP_ratio**: Ratio computed in the subject's native space.
   - **diameter_AP_ratio_PAM50**: Ratio computed in the PAM50 space.
   - **diameter_AP_ratio_PAM50_normalized**: Ratio computed in the PAM50 space and normalized with adult healthy participants.

.. note::
   - The flag ``-metric`` can be used to specify the morphometric to compute the ratio.
   - The flag ``-distance`` can be used to select the distance (mm) in the superior-inferior direction along the cord to average healthy slices.
   - The flag ``-extent`` can be used to specify the extent (mm) to average metrics of healthy levels.


Compute ratio of **AP-diameter**  normalized with healthy controls
------------------------------------------------------------------
We will add the flag ``-normalize-hc`` to use a database of adult healthy participants to normalize the anteroposterior diameters.

.. code:: sh

   sct_compute_compression -i t2_compressed_seg.nii.gz -vertfile t2_compressed_seg_labeled.nii.gz -l t2_compressed_labels-compression.nii.gz -metric diameter_AP -normalize-hc 1 -o ap_ratio_norm_PAM50.csv

:Input arguments:
   - ``-i`` : The input segmentation file.
   - ``-vertfile`` : Vertebral labeling file.
   - ``-l`` : Compression labels file.
   - ``-metric``: Metric to compute ratio: diameter_AP (default).
   - ``-normalize-hc``: Set to 1 to normalize the metrics using a database of healthy controls. Set to 0 to not normalize.
   - ``-o`` : The output CSV file.

:Output files/folders:
   - ``ap_ratio_norm_PAM50.csv`` : A file containing the ratio values for each. This file is partially replicated in the table below.

.. csv-table:: Anterposterior diameter ratio with levels above and below all compressions normalized with healthy controls.
   :file: ap_ratio_norm_PAM50.csv
   :header-rows: 1

:Legend:   
   - **diameter_AP_ratio**: Ratio computed in the subject's native space.
   - **diameter_AP_ratio_PAM50**: Ratio computed in the PAM50 space.
   - **diameter_AP_ratio_PAM50_normalized**: Ratio computed in the PAM50 space and normalized with adult healthy participants.

.. note::
   - The flag ``-sex`` can be used to select the sex of adult healthy participants to use for the normalization.
   - The flag ``-age`` can be used to select the age range of adult healthy participants to use for normalization.


Compute adapted Spinal Cord Occupation Ratio (aSCOR)
-----------------------------------------------------

You can also quantify spinal canal occupancy by first segmenting the canal, then computing aSCOR.

.. code:: sh

   sct_deepseg canal -i t2_compressed.nii.gz -o t2_compressed_canal_seg.nii.gz -qc ~/qc_singleSubj
   sct_compute_ascor -i-SC t2_compressed_seg.nii.gz -i-canal t2_compressed_canal_seg.nii.gz -perlevel 1 -o ascor.csv

:Input arguments:
   - ``-i-SC`` : Spinal cord segmentation input for ``sct_compute_ascor``.
   - ``-i-canal`` : Spinal canal segmentation input for ``sct_compute_ascor``.
   - ``-perlevel`` : Set to ``1`` to report per-level values.
   - ``-o`` : Output CSV filename.

:Output files/folders:
   - ``ascor.csv`` : aSCOR values for the selected region/levels.
