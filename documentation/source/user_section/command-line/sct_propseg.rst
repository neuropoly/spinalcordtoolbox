.. _sct_propseg: 

sct_propseg
===========

.. image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/command-line/sct_propseg_example.png

Command-line Help
-----------------

.. argparse::
   :ref: spinalcordtoolbox.scripts.sct_propseg.get_parser
   :prog: sct_propseg
   :markdownhelp:


Algorithm details
-----------------

The ``sct_propseg`` algorithm is a three step process defined as follows:

1. Centerline detection
***********************

``sct_propseg`` starts by using a machine learning-based method (`OptiC <https://doi.org/10.1007/978-3-319-66185-8_80>`__) to automatically detect the approximate center of the spinal cord.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord-segmentation/optic_steps.png
  :align: center
  :figwidth: 500px

(The centerline detection step is also provided in a standalone script called :ref:`sct_get_centerline`.)

2. Mesh propagation
*******************

Next, a coarse 3D mesh is created by propagating along the spinal cord (PropSeg).

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord-segmentation/mesh_propagation.png
  :align: center
  :figwidth: 500px

3. Surface refinement
*********************

Finally, the surface of the mesh is refined using small adjustments.


When to use ``sct_propseg``
---------------------------

When choosing between ``sct_propseg`` and ``sct_deepseg spinalcord``, it is important to know that no one algorithm is strictly superior in all cases; whether one works better than the other is data-dependent.

As a rule of thumb:

- :ref:`sct_deepseg` will generally perform better on "real world" scans of adult humans, including both healthy controls and subjects with conditions such as multiple sclerosis (MS), degenerative cervical myelopathy (DCM), and others. This is because :ref:`sct_deepseg` was trained on human subjects, including those with a range of common and representative spinal cord pathologies.
- :ref:`sct_propseg`, on the other hand, will generally perform better on non-standard scans, including exvivo spinal cords, pediatric subjects, and non-human species. This is because :ref:`sct_propseg` uses a mesh propagation-based approach that is more agnostic to details such as the shape and size of the spinal cord, the presence of surrounding tissue, etc.

That said, given the variation in imaging data (imaging centers, sizes, ages, coil strengths, contrasts, scanner vendors, etc.), SCT recommends to try both algorithms with your pilot scans to evaluate the merit of each on your specific dataset, then stick with a single method throughout your study.

Note: Development of these approaches is an iterative process, and the data used to develop these approaches evolves over time. If you have input regarding what has worked (or hasn't worked) for you, we would be happy to hear your thoughts in the `SCT forum <https://forum.spinalcordmri.org/c/sct>`__, as it could help to improve the toolbox for future users.