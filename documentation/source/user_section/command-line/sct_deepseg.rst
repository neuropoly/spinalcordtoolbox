.. _sct_deepseg:

sct_deepseg
===========

Here we provide a gallery of each model available in the ``sct_deepseg`` CLI tool.

Spinal cord
-----------

.. |spinalcord| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/contrast_agnostic.png
   :target: deepseg/spinalcord.html

.. |sc_lumbar_t2| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/lumbar_t2.png
   :target: deepseg/sc_lumbar_t2.html

.. |sc_epi| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/epi_bold.png
   :target: deepseg/sc_epi.html

.. |sc_mouse_t1| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/mouse_t1.png
   :target: deepseg/sc_mouse_t1.html

Spinal cord segmentation can be performed by running the following sample command:

.. code::

   sct_deepseg spinalcord -i input.nii.gz

You can replace "``spinalcord``" with any of the task names in the table below to perform different tasks. Click on a task below for more information.

.. list-table::
   :align: center
   :widths: 25 25 25

   * - |spinalcord| ``spinalcord``
     - |sc_lumbar_t2| ``sc_lumbar_t2``
     - |sc_epi| ``sc_epi``
   * - |sc_mouse_t1| ``sc_mouse_t1``
     -
     -

Gray matter
-----------
   
.. |graymatter| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/graymatter.png
   :target: deepseg/graymatter.html

.. |gm_sc_7t_t2star| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/gm_sc_7t_t2star.png
   :target: deepseg/gm_sc_7t_t2star.html

.. |gm_wm_exvivo_t2| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/exvivo_gm_t2.png
   :target: deepseg/gm_wm_exvivo_t2.html

.. |gm_wm_mouse_t1| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/gm_wm_mouse_t1.png
   :target: deepseg/gm_wm_mouse_t1.html

.. |gm_mouse_t1| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/gm_mouse_t1.png
   :target: deepseg/gm_mouse_t1.html

Gray matter segmentation can be performed by running the following sample command:

.. code::

   sct_deepseg graymatter -i input.nii.gz

You can replace "``graymatter``" with any of the task names in the table below to perform different tasks. Click on a task below for more information.

.. list-table::
   :align: center
   :widths: 25 25 25

   * - |graymatter| ``graymatter``
     - |gm_sc_7t_t2star| ``gm_sc_7t_t2star``
     - |gm_wm_exvivo_t2| ``gm_wm_exvivo_t2``
   * - |gm_wm_mouse_t1| ``gm_wm_mouse_t1``
     - |gm_mouse_t1| ``gm_mouse_t1``
     -


Pathologies
-----------

.. |lesion_sci_t2| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/sci_lesion_sc_t2.png
   :target: deepseg/lesion_sci_t2.html

.. |lesion_ms_mp2rage| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/ms_lesion_mp2rage.png
   :target: deepseg/lesion_ms_mp2rage.html

.. |lesion_ms_axial_t2| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/ms_lesion_sc_axial_t2.png
   :target: deepseg/lesion_ms_axial_t2.html

.. |lesion_ms| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/ms_lesion.png
   :target: deepseg/lesion_ms.html

.. |tumor_edema_cavity_t1_t2| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/tumor_edema_cavity_t1_t2.png
   :target: deepseg/tumor_edema_cavity_t1_t2.html

.. |tumor_t2| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/tumor_t2.png
   :target: deepseg/tumor_t2.html

Pathology segmentation can be performed by running the following sample command:

.. code::

   sct_deepseg lesion_sci_t2 -i input.nii.gz

You can replace "``lesion_sci_t2``" with any of the task names in the table below to perform different tasks. Click on a task below for more information.


.. list-table::
   :align: center
   :widths: 25 25 25

   * - |lesion_sci_t2| ``lesion_sci_t2``
     - |lesion_ms| ``lesion_ms``
     - |lesion_ms_axial_t2| ``lesion_ms_axial_t2``
   * - |lesion_ms_mp2rage| ``lesion_ms_mp2rage``
     - |tumor_edema_cavity_t1_t2| ``tumor_edema_cavity_t1_t2``
     - |tumor_t2| ``tumor_t2``


Other structures
----------------

.. |rootlets| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/spinal_rootlets_t2.png
   :target: deepseg/rootlets.html

.. |spine| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/totalspineseg.png
   :target: deepseg/spine.html

Multiple structures may be segmented by running the following sample command:

.. code::

   sct_deepseg rootlets -i input.nii.gz

You can replace "``rootlets``" with any of the task names in the table below to perform different tasks. Click on a task below for more information.


.. list-table::
   :align: center
   :widths: 25 25 25

   * - |rootlets| ``rootlets``
     - |spine| ``spine``
     - |canal| ``canal``
   * - |brachial_plexus| ``brachial_plexus``
     -
     -

.. |canal| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/canal.png
   :target: deepseg/canal.html

.. |brachial_plexus| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/brachial-plexus.png
   :target: deepseg/brachial_plexus.html


Retired models
--------------

.. |seg_sc_ms_lesion_stir_psir| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/ms_lesion_sc_stir_psir.png
   :target: deepseg/seg_sc_ms_lesion_stir_psir.html

.. |ms_sc_mp2rage| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/ms_sc_mp2rage.png
   :target: deepseg/seg_ms_sc_mp2rage.html

.. |sc_t2star| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/sc_t2star.png
    :target: deepseg/sc_t2star.html

.. |sc_canal_t2| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/spinal_canal_t2.png
   :target: deepseg/sc_canal_t2.html

These models have been replaced by newer, more advanced models. We recommend switching to the model listed in the table below.

If you absolutely require these models, you can downgrade to version of SCT listed in the table below. If you do this, please let us know on the SCT Forum so we can better understand your use-case, and potentially reinstate the model if necessary.

.. list-table::
   :align: center
   :widths: 33 33 33

   * - **Model**
     - **Last available**
     - **Superseded by**
   * - |seg_sc_ms_lesion_stir_psir| ``seg_sc_ms_lesion_stir_psir``
     - SCT Version ``6.4``
     - ``lesion_ms`` (contrast-agnostic MS lesion segmentation)
   * - |ms_sc_mp2rage| ``ms_sc_mp2rage``
     - SCT Version ``6.4``
     - ``spinalcord`` (contrast-agnostic SC segmentation)
   * - |sc_t2star| ``sc_t2star``
     - SCT Version ``6.5``
     - ``spinalcord`` (contrast-agnostic SC segmentation) and ``sc_epi`` (for EPI-BOLD fMRI SC segmentation)
   * - |sc_canal_t2| ``sc_canal_t2``
     - SCT Version ``7.3``
     - ``canal`` (contrast-agnostic canal segmentation)

.. toctree::
   :hidden:
   :maxdepth: 2

   deepseg/spinalcord
   deepseg/sc_t2star
   deepseg/seg_ms_sc_mp2rage
   deepseg/sc_epi
   deepseg/sc_mouse_t1
   deepseg/sc_lumbar_t2
   deepseg/graymatter
   deepseg/gm_wm_exvivo_t2
   deepseg/gm_sc_7t_t2star
   deepseg/gm_mouse_t1
   deepseg/gm_wm_mouse_t1
   deepseg/lesion_sci_t2
   deepseg/seg_sc_ms_lesion_stir_psir
   deepseg/lesion_ms_axial_t2
   deepseg/lesion_ms_mp2rage
   deepseg/lesion_ms
   deepseg/tumor_edema_cavity_t1_t2
   deepseg/tumor_t2
   deepseg/rootlets
   deepseg/canal
   deepseg/sc_canal_t2
   deepseg/spine
   deepseg/brachial_plexus
