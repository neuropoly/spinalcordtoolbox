Using the atlas to extract MTR in white matter
##############################################

Now that the atlas has been warped to the MT space, it can be used to extract MTR for specific white matter regions. First, we will extract MTR for the white matter as a whole.

.. code::

   sct_extract_metric -i mtr.nii.gz -f label/atlas -method map -l 51 -vert 2:5 -o mtr_in_wm.csv

:Input arguments:
   - ``-i`` : Image to extract values from.
   - ``-f`` : File or folder used to pick out specific regions from the input image. In this case, we supply the folder of the white matter atlas that was transformed to the space of the MTR data.
   - ``-method`` : We specify ``map`` to choose the Maximum a Posteriori method, which helps to account for the partial volume effect. This method is most suitable for small regions. This method should only be used with the PAM50 white/gray matter atlas (or with any custom atlas) as long as the sum across all overlapping labels equals 1 in each voxel part of the atlas.
   - ``-l`` : The IDs of the label (or group of labels) to compute metrics for. In this case, label 51 is a combined label that represents all of the WM tracts together. You can see the full list of labels to choose from by running :ref:`sct_extract_metric` ``-list-labels``.
   - ``-o`` : The name of the output file.

.. csv-table:: ``mtr_in_wm.csv``: MTR values in white matter
   :file: mtr_in_wm.csv
   :header-rows: 1

The label volume fraction is indicated as “``Size [vox]``”, which gives you a sense of the reliability of the measure. In this example, for each slice, the metric was computed based on 50-70 voxels. As demonstrated in `[De Leener et al., Neuroimage 2017; Appendix] <https://doi.org/10.1016/j.neuroimage.2016.10.009>`_, having at least 30 voxels results in an error smaller than 2%, while having at least 240 voxels results in an error smaller than 1% (assuming an SNR of 10).

.. warning::

   Be careful to always check the associated volume fraction of your metrics (indicated by the Size [vox] column). This is especially relevant if you are restricting the metric computation to a subset of vertebral levels or axial slices; if the number of voxels included in the computation is too low, your quantifications will be unreliable. If you publish, we recommend including the volume fraction associated with all estimated metrics.


Using the atlas to extract MTR in CST
-------------------------------------

This command showcases some alternate options you can choose, e.g. different labels, quantification methods, and slice ranges.

.. code::

   sct_extract_metric -i dti_FA.nii.gz -f label/atlas -l 4,5 -method wa -z 2:14 -o fa_in_cst.csv