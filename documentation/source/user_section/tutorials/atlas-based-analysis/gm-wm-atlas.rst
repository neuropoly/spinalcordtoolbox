Atlas-based analysis
####################

While it can be helpful to extract metrics using the white and gray matter segmentations alone (as was shown in the :ref:`gm-wm-metric-computation` tutorial), there is additional value in extracting metrics from more specific regions of the white and gray matter (e.g. individual tracts). This can be done using an atlas.

In SCT, a digitalized version of the Gray’s Anatomy spinal cord white matter atlas [Standring, Gray's Anatomy 2008] was merged with the :ref:`pam50`. For more information on the procedure, see `[Lévy et al., Neuroimage 2015] <https://doi.org/10.1016/j.neuroimage.2015.06.040>`_.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/pam50/white_matter_atlas_detailed.png
   :align: center

The atlas is composed of 30 WM tracts, 6 GM regions and the surrounding CSF. The reason for including the CSF is to be able to account for the :ref:`partial volume effect <partial-volume-effect>` during metric estimation.