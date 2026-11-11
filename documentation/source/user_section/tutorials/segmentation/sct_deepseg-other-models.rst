``sct_deepseg``: Other specialized models
#########################################

The :ref:`sct_deepseg` script also provides access to specialized models that focus on tasks outside of basic spinal cord segmentation. In recent versions, new models for the segmentation of spinal tumors, human multi-class SC & GM at 7T, MS lesion segmentation on MP2RAGE, mouse SC, etc. have been added.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord-segmentation/sct_deepseg_models.png
   :align: center

   Sample models. (**Tumor segmentation**: `Lemay et al. NeuroImage Clinical (2021) <https://doi.org/10.1016/j.nicl.2021.102766>`__,
   **WM/GM cord segmentation at 7T**: `Laines Medina et al. arXiv (2021) <https://doi.org/10.48550/arXiv.2110.06516>`__,
   **MP2RAGE SC and lesion segmentation**: `Cohen-Adad et al. Zenodo release (2023) <https://doi.org/10.5281/zenodo.8376754>`__)

You can learn more about all of the available models by:

* Running :ref:`sct_deepseg` ``-h`` to view a basic summary
* Running :ref:`sct_deepseg` ``-task-details`` to view detailed descriptions of each model.
* Visiting the visual gallery of models by going to the :ref:`sct_deepseg` page.
