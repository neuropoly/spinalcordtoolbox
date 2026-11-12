.. _contrast-agnostic-registration:

Contrast-agnostic registration with deep learning
#################################################

This tutorial will demonstrate how to coregister two images together that have different contrasts using deep
learning. The algorithm is based on `SynthMorph <https://doi.org/10.48550/arXiv.2004.10282>`__. More details of its
implementation in SCT can be found `here <https://github.com/ivadomed/multimodal-registration>`__.

.. toctree::
   :maxdepth: 1

   contrast-agnostic-registration/before-starting
   contrast-agnostic-registration/preprocessing-t2
   contrast-agnostic-registration/preprocessing-t1
   contrast-agnostic-registration/coregistering-t1-t2