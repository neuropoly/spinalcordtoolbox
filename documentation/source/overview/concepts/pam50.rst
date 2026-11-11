.. _pam50:

PAM50 Template
**************

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/pam50/pam50_template.png
    :figwidth: 40%
    :align: right

Introduction
============

Template-based analysis of multi-parametric MRI data of the spinal cord sets the foundation for 
standardization and reproducibility. Particularly, it allows researchers to process their data 
with minimum bias and perform multi-center studies, with the goal of improving patient diagnosis 
and prognosis and discovering new biomarkers of spinal-related diseases.

To perform template-based analysis, a template is needed. Templates act as a common reference 
space to realign each subject to. PAM50 is one such template for MRI of the full spinal cord 
and brainstem, and is included in your installation of SCT. "PAM" stands for “Polytechnique”, 
“Aix-Marseille University” and “Montreal Neurological Institute”, the three institutions that 
collaborated to develop the template. The PAM50 template has the following features:

* It is available for T1-, T2-and T2*-weighted MRI contrast.
* It is compatible with the ICBM152 brain template (a.k.a. the MNI coordinate system), allowing researchers to 
  conduct simultaneous brain/spine studies within the same coordinate system.
* It includes a probabilistic atlas for white matter, gray matter, and CSF regions.
* It is a straightened template rather than a curved one.

The template is stored in the ``data/PAM50/`` directory in your SCT installation directory. 
``PAM50/`` contains several subfolders, each corresponding to different aspects of PAM50.

.. note:: For more details on the creation of the template, see the `neuropoly/template GitHub repository <https://github.com/neuropoly/template>`_. 


.. _pam50-template-section:

Template
========

**Location:** ``PAM50/template``

The template folder contains the PAM50 template for various MRI contrast types. It also contains binary 
and probabilistic masks for different spinal features, as well as point-wise labels for vertebral 
levels and intervertebral discs.

The template folder also contains an ``info_label.txt`` file to explain what each file represents:

.. include:: info_label-template.txt
   :code:

Please note that the white and gray matter masks found in ``template/`` are for the entire structures. 
If you are looking for individual tracts, the relevant volumes are provided within the :ref:`pam50-atlas-section` section below.


.. _pam50-atlas-section:

White and Gray Matter Atlas
===========================

**Location:** ``PAM50/atlas``

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/pam50/white_matter_atlas_illustration.png
    :align: right

The White Matter atlas will be a useful tool for your studies of specific spinal cord tracts. It consists of 36 NIfTI volumes named ``PAM50_atlas_<tract_number>.nii.gz`` where ``<tract_number>`` is the number identifying the tract. Fifteen WM tracts and three GM regions are available for each side. The values of each voxel of the files ``PAM50_atlas_<tract_number>.nii.gz`` are the voxel volume proportions occupied by the corresponding tract.

Notably, each volume is a probabilistic mask, or "soft mask". The voxel values in these volumes range from 0 to 1; voxels in the center of each tract will equal 1, while voxels near the edges of each tract will be weighted closer to 0. This is to account for `partial volume effects <https://mriquestions.com/partial-volume-effects.html>`_.

The atlas folder also contains an ``info_label.txt`` file to explain what each file represents:

.. include:: info_label-atlas.txt
   :code:

.. note:: For more details on the implementation of atlas, see the `White matter atlas script <https://github.com/spinalcordtoolbox/spinalcordtoolbox-dev/tree/master/dev/atlas>`_. 


.. _pam50-spinal-levels-section:

Spinal levels
=============

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/pam50/spinal_levels.png
    :figwidth: 100%
    :align: right

The file ``PAM50/template/PAM50_spinal_levels.nii.gz`` represents the spinal cord segmentation with parcellation 
across 30 spinal levels ranging from C1 to S5. Like for the vertebral level file, each level is encoded with 
an integer value (C1: **1**, C2: **2**, ..., C8: **8**, T1: **9**, etc.). 

The file ``PAM50/template/PAM50_spinal_midpoint.nii.gz`` represents the mid-point of the spinal level. Each level is 
encoded by one voxel centered in the spinal cord, and the value of the voxel corresponds to the spinal level. This 
file can be useful for registration to the PAM50 using the spinal levels instead of the intervertebral discs, if the 
former are available (e.g., via nerve rootlets segmentation).

The spinal levels are estimated from the relative length of each spinal level with respect to the length of the 
full spinal cord (expressed as a percentage), found in Table 3 of the article by 
`Frostell et al. (2016) <https://doi.org/10.3389/fneur.2016.00238>`_.
The figure below (extracted from Frostell et al.) shows the spatial correspondance between the spinal vs. vertebral levels.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/d39d9e5775a178a125c9479df43ee3f34fb9bdef/pam50/fneur-07-00238-g001.jpg
    :figwidth: 100%
    :align: right

.. note:: For more details on the implementation of the spinal levels in SCT, see the `Pull Request on GitHub <https://github.com/spinalcordtoolbox/PAM50/pull/18>`_. 



References
==========

* `De Leener B, Fonov VS, Collins DL, Callot V, Stikov N, Cohen-Adad J. PAM50: Unbiased multimodal template of the brainstem and spinal cord aligned with the ICBM152 space. Neuroimage, 2018, 165:170-179. <https://doi.org/10.1016/j.neuroimage.2017.10.041>`_
* `Levy S, Benhamou M, Naaman C, Rainville P, Callot V, Cohen-Adad J. White matter atlas of the human spinal cord with estimation of partial volume effect. Neuroimage, 2015, 119:262-271. <https://doi.org/10.1016/j.neuroimage.2015.06.040>`_
* `Frostell A, Hakim R, Thelin EP, Mattsson P, Svensson M. A Review of the Segmental Diameter of the Healthy Human Spinal Cord. Front Neurol. 2016 Dec 23;7:238.. <https://doi.org/10.3389/fneur.2016.00238>`_
