"""
Deals with models for deepseg module. Available models are listed under MODELS.
Important: model names (onnx or pt files) should have the same name as the enclosing folder.

Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""


import os
import json
import logging
import textwrap
import shutil
import glob
import re
from pathlib import Path
from importlib.metadata import metadata

from spinalcordtoolbox import download
from spinalcordtoolbox.utils.sys import stylize, __deepseg_dir__, LazyLoader

tss_init = LazyLoader("tss_init", globals(), 'totalspineseg.init_inference')
auglab_add_trainer = LazyLoader("auglab_add_trainer", globals(), 'auglab.add_trainer')

logger = logging.getLogger(__name__)

# List of models. The convention for model names is: (species)_(university)_(contrast)_region
# Regions could be: sc, gm, lesion, tumor
# NB: The 'url' field should either be:
#     1) A <mirror URL list> containing different mirror URLs for the model
#     2) A dict of <mirror URL lists>, where each list corresponds to a different seed (for model ensembling), and
#        each dictionary key corresponds to the seed's name (seed names are used to create subfolders per-seed)
MODELS = {
    "mice_uqueensland_sc": {
        "url": [
            "https://github.com/ivadomed/mice_uqueensland_sc/releases/download/r20200622/r20200622_mice_uqueensland_sc.zip",
            "https://osf.io/nu3ma/download?version=6",
        ],
        "description": "Cord segmentation model on mouse MRI. Data from University of Queensland.",
        "contrasts": ["t1"],
        "framework": "ivadomed",
        "thr": 0.0,  # Only for display in argparse help (postprocessing.binarize_prediction is not present in model json)
        "default": False,
    },
    "mice_uqueensland_gm": {
        "url": [
            "https://github.com/ivadomed/mice_uqueensland_gm/releases/download/r20200622/r20200622_mice_uqueensland_gm.zip",
            "https://osf.io/mfxwg/download?version=6",
        ],
        "description": "Gray matter segmentation model on mouse MRI. Data from University of Queensland.",
        "contrasts": ["t1"],
        "framework": "ivadomed",
        "thr": 0.0,  # Only for display in argparse help (postprocessing.binarize_prediction is not present in model json)
        "default": False,
    },
    "t2_tumor": {
        "url": [
            "https://github.com/ivadomed/t2_tumor/archive/r20201215.zip"
        ],
        "description": "Cord tumor segmentation model, trained on T2-weighted contrast.",
        "contrasts": ["t2"],
        "framework": "ivadomed",
        "thr": 0.5,  # Only for display in argparse help (mirrors postprocessing.binarize_prediction, which is 0.5 in model json)
        "default": False,
    },
    "findcord_tumor": {
        "url": [
            "https://github.com/ivadomed/findcord_tumor/archive/r20201215.zip"
        ],
        "description": "Cord localisation model, trained on T2-weighted images with tumor.",
        "contrasts": ["t2"],
        "framework": "ivadomed",
        "thr": 0.5,  # Only for display in argparse help (mirrors postprocessing.binarize_prediction, which is 0.5 in model json)
        "default": False,
    },
    "model_seg_sctumor-edema-cavity_t2-t1_unet3d-multichannel": {
        "url": [
            "https://github.com/ivadomed/model_seg_sctumor-edema-cavity_t2-t1_unet3d-multichannel/archive/r20201215.zip"
        ],
        "description": "Multiclass cord tumor segmentation model.",
        "contrasts": ["t2", "t1"],
        "framework": "ivadomed",
        "thr": 0.5,  # Only for display in argparse help (mirrors postprocessing.binarize_prediction, which is 0.5 in model json)
        "default": False,
    },
    "model_seg_exvivo_gm-wm_t2_unet2d-multichannel-softseg": {
        "url": [
            "https://github.com/ivadomed/model_seg_exvivo_gm-wm_t2_unet2d-multichannel-softseg/archive/r20210401_v2.zip"
        ],
        "description": "Grey/white matter seg on exvivo human T2w.",
        "contrasts": ["t2"],
        "framework": "ivadomed",
        "thr": 0.0,  # Only for display in argparse help (postprocessing.binarize_prediction is not present in model json)
        "default": False,
    },
    "model_7t_multiclass_gm_sc_unet2d": {
        "url": [
            "https://github.com/ivadomed/model_seg_gm-wm_t2star_7t_unet3d-multiclass/archive/refs/tags/r20211012.zip"
        ],
        "description": "SC/GM multiclass segmentation on T2*-w contrast at 7T. The model was created by N.J. Laines Medina, "
                       "V. Callot and A. Le Troter at CRMBM-CEMEREM Aix-Marseille University, France",
        "contrasts": ["t2star"],
        "framework": "ivadomed",
        "thr": 0.5,  # Only for display in argparse help (mirrors postprocessing.binarize_prediction, which is 0.5 in model json)
        "default": False,
    },
    "model_seg_epfl_t2w_lumbar_sc": {
        "url": [
            "https://github.com/ivadomed/lumbar_seg_EPFL/releases/download/r20231004/model_seg_epfl_t2w_lumbar_sc.zip"
        ],
        "description": "Lumbar SC segmentation on T2w contrast with 3D UNet",
        "contrasts": ["t2"],
        "framework": "ivadomed",
        "thr": 0.5,  # Only for display in argparse help (mirrors postprocessing.binarize_prediction, which is 0.5 in model json)
        "default": False,
    },
    # NB: Handling image binarization threshold for ivadomed vs. non-ivadomed models:
    #   - All models:
    #       - SCT provides a `-thr` argument for overriding default threshold values
    #       - `-thr` will only be shown to the user if 'model': 'thr' is not None.
    #       - The argparse help for `-thr` will display the default 'model': 'thr' value
    #   - ivadomed models (above):
    #       - Threshold value is actually fetched **by the ivadomed package** from a `.json` sidecar file
    #       - Binarization is applied within the ivadomed package
    #       - Meaning, the 'thr' values above are simply for display in the `-thr` argparse help
    #   - non-ivadomed models (below)
    #       - Models do not have a `.json` sidecar file, since they were not developed with ivadomed
    #       - So, threshold value is stored here, within the model dict
    #       - Binarization is applied within SCT code
    # v4 model: trained on sc-crop cropped volumes.
    "model_seg_sc_contrast_agnostic_nnunet": {
        "url": [
            "https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord/releases/download/v4.0/model_contrast_agnostic_20260628.zip"
        ],
        "description": "Spinal cord segmentation agnostic to MRI contrasts",
        "contrasts": ["any"],
        "framework": "nnunetv2",
        "thr": None,  # We're now using an nnUNet model, which does not need a threshold
        "default": True,
        "cropped_image": True,
    },
    "model_seg_sci_multiclass_sc_lesion_nnunet": {
        "url": [
            "https://github.com/ivadomed/model_seg_sci/releases/download/r20240729/model_SCIsegV2_r20240729.zip"
        ],
        "description": "Intramedullary SCI lesion and cord segmentation in T2w MRI",
        "contrasts": ["t2"],
        "framework": "nnunetv2",
        "thr": None,  # Images are already binarized when splitting into sc-seg + lesion-seg
        "default": False,
    },
    "model_seg_spinal_rootlets_nnunet": {
        "url": [
            "https://github.com/ivadomed/model-spinal-rootlets/releases/download/r20250318/model-spinal-rootlets-multicon-r20250318.zip"
        ],
        "description": "Segmentation of spinal nerve rootlets for T2w and MP2RAGE contrasts (T1w-INV1, T1w-INV2, and UNIT1) using nnUNet",
        "contrasts": ["t2", "UNIT1", "T1w-INV1", "T1w-INV2"],
        "framework": "nnunetv2",
        "thr": None,  # Multiclass rootlets model (1.0, 2.0, 3.0...) -> no thresholding
        "default": False,
    },
    "model_seg_gm_wm_mouse_nnunet": {
         "url": [
             "https://github.com/ivadomed/model_seg_mouse-sc_wm-gm_t1/releases/download/v0.4/model.zip"
         ],
         "description": "White and grey matter segmentation on T1-weighted exvivo mouse spinal cord using nnUNet",
         "contrasts": ["t1"],
         "framework": "nnunetv2",
         "thr": None,  # Images are already binarized when splitting into gm-seg and wm-seg
         "default": False,
     },
    "model_seg_sc_epi_nnunet": {
         "url": [
             "https://github.com/sct-pipeline/fmri-segmentation/releases/download/v0.2/model-fmri-segmentation-v0.2_nnUNetTrainer.zip"
         ],
         "description": "Spinal cord segmentation for EPI data (single 3D volume)",
         "contrasts": ["bold"],
         "framework": "nnunetv2",
         "thr": None,  # Images are already binarized
         "default": False,
     },
    "model_seg_ms_lesion_mp2rage": {
         "url": [
             "https://github.com/ivadomed/model_seg_ms_mp2rage/releases/download/r20240610/nnUNetTrainer_seg_ms_lesion_mp2rage__nnUNetPlans__3d_fullres.zip"
         ],
         "description": "Segmentation of spinal cord MS lesions on MP2RAGE UNIT1 contrast",
         "contrasts": ["UNIT1"],
         "framework": "nnunetv2",
         "thr": None,  # Images are already binarized
         "default": False,
     },
    "model_seg_ms_sc_lesion_bavaria_quebec_nnunet": {
        "url": [
            "https://github.com/ivadomed/model-seg-ms-axial-t2w/releases/download/r20241111/model_bavaria_quebec_axial_t2w_ms.zip"
        ],
        "description": "Intramedullary MS lesion and spinal cord segmentation in Axial T2w MRI",
        "contrasts": ["t2"],
        "framework": "nnunetv2",
        "thr": None,  # Images are already binarized when splitting into sc-seg + lesion-seg
        "default": False,
    },
    "model_seg_ms_lesion": {
         "url": {
            "model_fold0": ["https://github.com/ivadomed/seg-sc-ms-lesion-multicontrast/releases/download/r20260703/model_fold0.zip"],
            "model_fold1": ["https://github.com/ivadomed/seg-sc-ms-lesion-multicontrast/releases/download/r20260703/model_fold1.zip"],
            "model_fold2": ["https://github.com/ivadomed/seg-sc-ms-lesion-multicontrast/releases/download/r20260703/model_fold2.zip"],
            "model_fold3": ["https://github.com/ivadomed/seg-sc-ms-lesion-multicontrast/releases/download/r20260703/model_fold3.zip"],
            "model_fold4": ["https://github.com/ivadomed/seg-sc-ms-lesion-multicontrast/releases/download/r20260703/model_fold4.zip"]
         },
         "description": "Segmentation of spinal cord MS lesions",
         "contrasts": ["any"],
         "framework": "nnunetv2",
         "thr": None,  # Images are already binarized
         "default": False,
         "cropped_image": True,
     },
    "model_seg_canal": {
        "url": [
            "https://github.com/ivadomed/model-canal-seg/releases/download/r20260406/model-canal-seg-r20260406.zip"
        ],
        "description": "Segmentation of spinal canal on all contrasts",
        "contrasts": ["all"],
        "framework": "nnunetv2",
        "thr": None,  # Images are already binarized
        "default": False,
    },
    "model_seg_spine_contrast_agnostic": {
         # NB: Rather than hardcoding the URLs ourselves, use the URLs from the totalspineseg package.
         # This means that when the totalspineseg package is updated, the URLs will be too, thus triggering
         # a re-installation of the model URLs
         "url": {
             k: re.findall(r'https?://\S+', v)
             for k, v in (meta.split(', ', 1) for meta in metadata('totalspineseg').get_all('Project-URL'))
             if k.startswith('Dataset')
         },
         "description": "Instance segmentation of vertebrae, intervertebral discs (IVDs), spinal cord, and spinal canal on multi-contrasts MRI scans.",
         "contrasts": ["any"],
         "framework": "nnunetv2",
         "thr": None,  # Images are already binarized
         "default": False,
     },
    "model_seg_gm_contrast_region_agnostic": {
        "url": [
            "https://github.com/ivadomed/model-gm-contrast-region-agnostic/releases/download/r20250420/Dataset820_gm-seg.zip"
        ],
        "description": "Segmentation of spinal cord gray matter on any region and contrast MRI",
        "contrasts": ["any"],
        "framework": "nnunetv2",
        "thr": None,
        "default": False,
     },
}


# List of tasks. The convention for task names is: action_(animal)_region_(contrast)
#   Regions could be: sc, gm, lesion, tumor
#   If the animal is human, omit it from the task label.
# Some optional tags to help with groupin and sorting during help display are as follows:
#   Group: The sub-header the tasks should be displayed under. If not specified, the model is placed under "OTHER"
#   Priority: Higher priority tasks are displayed higher within their own group, and vice versa. Default priority is 0,
#     and ties are sorted alphabetically.

CROP_MESSAGE = (
    'To crop the data you can first segment the spinal cord using the contrast agnostic model. This could be '
    'done using: "sct_deepseg spinalcord -i IMAGE -o IMAGE_sc", then crop the '
    'image with 30 mm of dilation on axial orientation around the spinal cord. This could be done using: '
    '"sct_crop_image -i IMAGE -m IMAGE_sc -dilate 30x30x5". Note that 30 is only for 1mm isotropic '
    'resolution, for images with another resolution divide 30/your_axial_resolution.'
)
TASKS = {
    'sc_mouse_t1':
        {'description': 'Cord segmentation on mouse MRI',
         'long_description': 'This segmentation model for T1w mouse spinal cord segmentation uses the UNet '
                             'architecture, and was created with the `ivadomed` package. Training data was provided by '
                             'Dr. A. Althobathi of the University of Queensland, with the files consisting of T1w '
                             'scans (along with manual spinal cord + gray matter segmentations) from 10 mice in total. '
                             'The dataset was comprised of both non-pathological (healthy) and pathological (diseased) '
                             'mice.',
         'url': 'https://github.com/ivadomed/mice_uqueensland_sc/',
         'models': ['mice_uqueensland_sc'],
         'citation': None,
         'group': 'spinal_cord',
         'priority': -1
         },
    'gm_mouse_t1':
        {'description': 'Gray matter segmentation on mouse MRI',
         'long_description': 'This segmentation model for T1w mouse spinal gray matter segmentation uses the UNet '
                             'architecture, and was created with the `ivadomed` package. Training data was provided by '
                             'Dr. A. Althobathi of the University of Queensland, with the files consisting of T1w '
                             'scans (along with manual spinal cord + gray matter segmentations) from 10 mice in total. '
                             'The dataset was comprised of both non-pathological (healthy) and pathological (diseased) '
                             'mice.',
         'url': 'https://github.com/ivadomed/mice_uqueensland_gm/',
         'models': ['mice_uqueensland_gm'],
         'citation': None,
         'group': 'gray_matter',
         'priority': -1  # Mouse segmentation tends to be less common
         },
    'tumor_t2':
        {'description': 'Cord tumor segmentation on T2-weighted contrast',
         'long_description': 'This segmentation model for T2w spinal tumor segmentation uses the UNet '
                             'architecture, and was created with the `ivadomed` package. Training data consisted of '
                             '380 pathological subjects in total: 120 with tumors of type Astrocytoma, 129 with '
                             'Ependymoma, and 131 with Hemangioblastoma. This model is used in tandem with another '
                             'model for specialized cord localisation of spinal cords with tumors '
                             '(https://github.com/ivadomed/findcord_tumor).',
         'url': 'https://github.com/sct-pipeline/tumor-segmentation',
         'models': ['findcord_tumor', 't2_tumor'],
         'citation': None,
         'group': 'pathology'
         },
    'tumor_edema_cavity_t1_t2':
        {'description': 'Multiclass cord tumor/edema/cavity segmentation',
         'long_description': 'This segmentation model for T1w and T2w spinal tumor, edema, and cavity segmentation '
                             'uses a 3D UNet architecture, and was created with the `ivadomed` package. Training data '
                             'consisted of a subset of the dataset used for the model `tumor_t2`, with 243 '
                             'subjects in total: 49 with tumors of type Astrocytoma, 83 with Ependymoma, and 111 with '
                             'Hemangioblastoma. For each subject, the requisite parts of the affected region (tumor, '
                             'edema, cavity) were segmented individually for training purposes. This model is used in '
                             'tandem with another model for specialized cord localisation of spinal cords with tumors '
                             '(https://github.com/ivadomed/findcord_tumor).',
         'url': 'https://github.com/ivadomed/model_seg_sctumor-edema-cavity_t2-t1_unet3d-multichannel',
         'models': ['findcord_tumor', 'model_seg_sctumor-edema-cavity_t2-t1_unet3d-multichannel'],
         'citation': textwrap.dedent(r"""
             ```bibtex
             @article{LEMAY2021102766,
                      title={Automatic multiclass intramedullary spinal cord tumor segmentation on MRI with deep learning},
                      journal={NeuroImage: Clinical},
                      volume={31},
                      pages={102766},
                      year-2021},
                      issn-2213-1582},
                      doi-https://doi.org/10.1016/j.nicl.2021.102766},
                      url-https://doi.org/10.1016/j.nicl.2021.102766},
                      author-Andreanne Lemay and Charley Gros and Zhizheng Zhuo and Jie Zhang and Yunyun Duan and Julien Cohen-Adad and Yaou Liu},
                      keywords-Deep learning, Automatic segmentation, Spinal cord tumor, MRI, Multiclass, CNN}
             }
             ```
         """),
         'group': 'pathology'
         },
    'gm_wm_exvivo_t2':
        {'description': 'Grey/white matter seg on exvivo human T2w',
         'long_description': 'This segmentation model for T2w human spinal gray and white matter uses a 2D Unet '
                             'architecture, and was created with the `ivadomed` package. Training data consisted '
                             'of 149 2D slices taken from the scan of an ex vivo spinal cord of a single subject, with '
                             'the gray and white matter from each slice manually segmented for training purposes. The '
                             'data was provided by the University of Queenland, and through this collaboration, the '
                             'model was subsequently applied to the development of an ex vivo spinal cord template '
                             '(https://archive.ismrm.org/2020/1171.html).',
         'url': 'https://github.com/ivadomed/model_seg_exvivo_gm-wm_t2_unet2d-multichannel-softseg',
         'models': ['model_seg_exvivo_gm-wm_t2_unet2d-multichannel-softseg'],
         'citation': None,
         'group': 'gray_matter'
         },
    'gm_sc_7t_t2star':
        {'description': 'SC/GM seg on T2*-weighted contrast at 7T',
         'long_description': 'This multiclass model (SC/GM) was developed by N.J. Laines Medina, V. Callot and A. Le '
                             'Troter at the Center for Magnetic Resonance in Biology and Medicine (CRMBM-CEMEREM, UMR '
                             '7339, CNRS, Aix-Marseille University, France). Training data consisted of T2*w scans '
                             'acquired at 7T from 72 subjects: 34 healthy controls, 25 patients with ALS, 13 patients '
                             'with MS. The model was validated by comparing with single-class models using 9-fold '
                             'Cross-Validation. It was enriched by integrating a hybrid data augmentation (composed of '
                             'classical geometric transformations, MRI artifacts, and real GM/WM contrasts distorted '
                             'with anatomically constrained deformation fields). Finally, it was tested with an '
                             'external multicentric database. For more information, see the following URL.',
         'url': 'https://github.com/ivadomed/model_seg_gm-wm_t2star_7t_unet3d-multiclass',
         'models': ['model_7t_multiclass_gm_sc_unet2d'],
         'citation': textwrap.dedent(r"""
             ```bibtex
             @misc{medina20212d,
                   title={2D Multi-Class Model for Gray and White Matter Segmentation of the Cervical Spinal Cord at 7T},
                   author={Nilser J. Laines Medina and Charley Gros and Julien Cohen-Adad and Virginie Callot and Arnaud Le Troter},
                   year={2021},
                   eprint={2110.06516},
                   archivePrefix={arXiv},
                   primaryClass={eess.IV}
             }
             ```
         """),
         'group': 'gray_matter'
         },
    'sc_lumbar_t2':
        {'description': 'Lumbar cord segmentation with 3D UNet',
         'long_description': 'This segmentation model for T2w spinal cord segmentation uses a 3D UNet architecture, '
                             'and was created with the ivadomed package. Training data was provided by Nawal Kinany '
                             'and Dimitry Van De Ville of EPFL, with the files consisting of lumbar T2w scans (and '
                             'manual spinal cord segmentations) of 11 healthy (non-pathological) patients.',
         'url': 'https://github.com/ivadomed/lumbar_seg_EPFL',
         'models': ['model_seg_epfl_t2w_lumbar_sc'],
         'citation': None,
         'group': 'spinal_cord'
         },
    'spinalcord':
        {'description': 'Spinal cord segmentation agnostic to MRI contrasts',
         'long_description': 'The contrast agnostic spinal cord segmentation uses a 3D CNN model based on the nnUNet '
                             'framework. Training data consists of healthy controls from the open-source Spine Generic '
                             'Multi Subject database and pathologies from private datasets including DCM, MS, '
                             'SCI (Acute, Intermediate and Chronic; Pre/Post-operative) patients. Segmentations have been '
                             'tested with the following contrasts: '
                             '[T1w, T2w, T2star, MTon_MTS, GRE_T1w, DWI, mp2rage_UNIT1, PSIR, STIR, EPI], but '
                             'other contrasts that are close visual matches may also work well with this model.',
         'url': 'https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord/',
         'models': ['model_seg_sc_contrast_agnostic_nnunet'],
         'citation': textwrap.dedent(r"""
             ```bibtex
             @article{bedard2025towards,
                  title={Towards contrast-agnostic soft segmentation of the spinal cord},
                  author={Bédard, Sandrine and Karthik, Enamundram Naga and Tsagkas, Charidimos and Pravatà, Emanuele and Granziera, Cristina and Smith, Andrew and Weber II, Kenneth Arnold and Cohen-Adad, Julien},
                  journal={Medical Image Analysis},
                  pages={103473},
                  year={2025},
                  publisher={Elsevier}
            }
            ```"""),  # noqa E501 (line too long)
         'group': 'spinal_cord',
         'priority': 100  # Force this, and its group, to be displayed first, as it's our main model
         },
    'lesion_sci_t2':
        {'description': 'Intramedullary SCI lesion and cord segmentation in T2w MRI',
         'long_description': 'This segmentation model for spinal cord injury segmentation uses a 3D U-Net '
                             'architecture, and was trained with the nnUNetV2 framework. It is a multiclass model, '
                             'outputting segmentations for both the hyperintense SCI lesions and spinal cord. Training '
                             'data consisted of T2w images from 7 sites with traumatic (acute pre-operative, intermediate, '
                             'chronic), non-traumatic (DCM) and ischemic SCI lesions spanning numerous resolutions, '
                             'orientations, as well as multiple scanner manufacturers and field strengths.',
         'url': 'https://github.com/ivadomed/model_seg_sci',
         'models': ['model_seg_sci_multiclass_sc_lesion_nnunet'],
         'citation': textwrap.dedent(r"""
             ```bibtex
             @InProceedings{10.1007/978-3-031-82007-6_19,
                            author="Karthik, Enamundram Naga and Valo{\v{s}}ek, Jan and Farner, Lynn and Pfyffer, Dario and Schading-Sassenhausen, Simon and Lebret, Anna and David, Gergely and Smith, Andrew C. and Weber II, Kenneth A. and Seif, Maryam and Freund, Patrick and Cohen-Adad, Julien",
                            editor="Wu, Shandong and Shabestari, Behrouz and Xing, Lei",
                            title="SCIsegV2: A Universal Tool for Segmentation of Intramedullary Lesions in Spinal Cord Injury",
                            booktitle="Applications of Medical Artificial Intelligence",
                            year="2025",
                            publisher="Springer Nature Switzerland",
                            address="Cham",
                            pages="198--209",
                            abstract="Spinal cord injury (SCI) is a devastating incidence leading to permanent paralysis and loss of sensory-motor functions potentially resulting in the formation of lesions within the spinal cord. Imaging biomarkers obtained from magnetic resonance imaging (MRI) scans can predict the functional recovery of individuals with SCI and help choose the optimal treatment strategy. Currently, most studies employ manual quantification of these MRI-derived biomarkers, which is a subjective and tedious task. In this work, we propose (i) a universal tool for the automatic segmentation of intramedullary SCI lesions, dubbed SCIsegV2, and (ii) a method to automatically compute the width of the tissue bridges from the segmented lesion. Tissue bridges represent the spared spinal tissue adjacent to the lesion, which is associated with functional recovery in SCI patients. The tool was trained and validated on a heterogeneous dataset from 7 sites comprising patients from different SCI phases (acute, sub-acute, and chronic) and etiologies (traumatic SCI, ischemic SCI, and degenerative cervical myelopathy). Tissue bridges quantified automatically did not significantly differ from those computed manually, suggesting that the proposed automatic tool can be used to derive relevant MRI biomarkers. SCIsegV2 and the automatic tissue bridges computation are open-source and available in Spinal Cord Toolbox (v6.4 and above) via the sct{\_}deepseg -task seg{\_}sc{\_}lesion{\_}t2w{\_}sci and sct{\_}analyze{\_}lesion functions, respectively.",
                            isbn="978-3-031-82007-6"
            }
             ```"""),  # noqa E501 (line too long)
         'group': 'pathology'
         },
    'rootlets':
        {'description': 'Segmentation of spinal nerve rootlets for T2w and MP2RAGE contrasts (T1w-INV1, T1w-INV2, and UNIT1)',
         'long_description': 'This segmentation model for spinal nerve rootlets segmentation uses a 3D U-Net '
                             'architecture, and was trained with the nnUNetV2 framework. It is a multiclass model, '
                             'outputting a single segmentation image containing 8 classes representing the C2-T1 '
                             'dorsal and ventral spinal cord nerve rootlets. Training data included images from healthy '
                             'subjects across three datasets: spine-generic multi-subject (3T T2w, n=21), OpenNeuro '
                             'ds004507 (3T T2w, n=7, 10 images), and private data (7T MP2RAGE, n=15, 3 contrasts per '
                             'subject, 45 images).',
         'url': 'https://github.com/ivadomed/model-spinal-rootlets',
         'models': ['model_seg_spinal_rootlets_nnunet'],
         'citation': textwrap.dedent(r"""
             ```bibtex
             @misc{krejci2025rootletsegdeeplearningmethod,
                   title={RootletSeg: Deep learning method for spinal rootlets segmentation across MRI contrasts},
                   author={Katerina Krejci and Jiri Chmelik and Sandrine Bédard and Falk Eippert and Ulrike Horn and Virginie Callot and Julien Cohen-Adad and Jan Valosek},
                   year={2025},
                   eprint={2509.16255},
                   archivePrefix={arXiv},
                   primaryClass={q-bio.TO},
                   url={https://doi.org/10.48550/arXiv.2509.16255},
             }
             ```"""),  # noqa E501 (line too long)
         },
    'gm_wm_mouse_t1':
        {'description': 'Exvivo mouse GM/WM segmentation for T1w contrast',
         'long_description': 'This segmentation model for gray and white matter segmentation of exvivo mice spinal '
                             'cords uses an NNunet architecture, and was created with the nnUNetV2 package. It is a '
                             'multiclass model, outputting segmentations for both the grey matter and white matter.'
                             'Training data consisted of 22 mice with different numbers of chunks, for a total of 72 '
                             'MRI 3D images. Each training image was T2-weighted, had a size of 200x200x500, and had '
                             'a resolution of 0.05mm isotropic. Training data was provided by the Balgrist Center at'
                             'the University of Zurich.',
         'url': 'https://github.com/ivadomed/model_seg_mouse-sc_wm-gm_t1',
         'models': ['model_seg_gm_wm_mouse_nnunet'],
         'citation': textwrap.dedent(r"""
             ```bibtex
             @software{cohen_adad_2024_10819207,
                       author={Cohen-Adad, Julien},
                       title={{Segmentation model of ex vivo mouse spinal cord white and gray matter}},
                       month=mar,
                       year=2024,
                       publisher={Zenodo},
                       version={v0.5},
                       doi={10.5281/zenodo.10819207},
                       url={https://doi.org/10.5281/zenodo.10819207}
             }
             ```"""),
         'group': 'gray_matter',
         'priority': -1  # Mouse segmentation tends to be less common
         },
    'sc_epi':
        {'description': 'Spinal cord segmentation for EPI-BOLD fMRI data',
         'long_description': 'This segmentation model for spinal cord on EPI data (single 3D volume) uses a 3D UNet model built from '
                             'the nnUNetv2 framework. The training data consists of 3D images (n=192) spanning numerous resolutions '
                             'from multiple sites like Max Planck Institute for Human Cognitive and Brain Sciences - Leipzig, '
                             'University of Geneva, Stanford University, Kings College London, Universitätsklinikum Hamburg. The '
                             'dataset has healthy control subjects. The model has been trained in a human-in-the-loop active learning fashion.',
         'url': 'https://github.com/sct-pipeline/fmri-segmentation',
         'models': ['model_seg_sc_epi_nnunet'],
         'citation': textwrap.dedent(r"""
             ```
             @article{Banerjee2025.01.07.631402,
                      author={Banerjee, Rohan and Kaptan, Merve and Tinnermann, Alexandra and Khatibi, Ali and Dabbagh, Alice and B{\"u}chel, Christian and K{\"u}ndig, Christian W. and Law, Christine S.W. and Pfyffer, Dario and Lythgoe, David J. and Tsivaka, Dimitra and Van De Ville, Dimitri and Eippert, Falk and Muhammad, Fauziyya and Glover, Gary H. and David, Gergely and Haynes, Grace and Haaker, Jan and Brooks, Jonathan C. W. and Finsterbusch, J{\"u}rgen and Martucci, Katherine T. and Hemmerling, Kimberly J. and Mobarak-Abadi, Mahdi and Hoggarth, Mark A. and Howard, Matthew A. and Bright, Molly G. and Kinany, Nawal and Kowalczyk, Olivia S. and Freund, Patrick and Barry, Robert L. and Mackey, Sean and Vahdat, Shahabeddin and Schading, Simon and McMahon, Stephen B. and Parish, Todd and Marchand-Pauvert, V{\'e}ronique and Chen, Yufen and Smith, Zachary A. and Weber, Kenneth A. and De Leener, Benjamin and Cohen-Adad, Julien},
                      title={EPISeg: Automated segmentation of the spinal cord on echo planar images using open-access multi-center data},
                      elocation-id{2025.01.07.631402},
                      year{2025},
                      doi{10.1101/2025.01.07.631402},
                      publisher{Cold Spring Harbor Laboratory},
                      abstract{Functional magnetic resonance imaging (fMRI) of the spinal cord is relevant for studying sensation, movement, and autonomic function. Preprocessing of spinal cord fMRI data involves segmentation of the spinal cord on gradient-echo echo planar imaging (EPI) images. Current automated segmentation methods do not work well on these data, due to the low spatial resolution, susceptibility artifacts causing distortions and signal drop-out, ghosting, and motion-related artifacts. Consequently, this segmentation task demands a considerable amount of manual effort which takes time and is prone to user bias. In this work, we (i) gathered a multi-center dataset of spinal cord gradient-echo EPI with ground-truth segmentations and shared it on OpenNeuro (https://doi.org/10.18112/openneuro.ds005143.v1.3.0), and (ii) developed a deep learning-based model, EPISeg, for the automatic segmentation of the spinal cord on gradient-echo EPI data. We observe a significant improvement in terms of segmentation quality compared to other available spinal cord segmentation models. Our model is resilient to different acquisition protocols as well as commonly observed artifacts in fMRI data. The training code is available at https://github.com/sct-pipeline/fmri-segmentation/, and the model has been integrated into the Spinal Cord Toolbox as a command-line tool.Competing Interest StatementSince January 2024, Dr. Barry has been employed by the National Institute of Biomedical Imaging and Bioengineering at the National Institutes of Health. This work was co-authored by Robert Barry in his personal capacity. The opinions expressed in this study are his own and do not necessarily reflect the views of the National Institutes of Health, the Department of Health and Human Services, or the United States government. The other authors declared no potential conflicts of interest with respect to the research, authorship, and/or publication of this article.},
                      URL{https://doi.org/10.1101/2025.01.07.631402},
                      eprint{https://doi.org/10.1101/2025.01.07.631402},
                      journal{bioRxiv}
             }
             ```"""),  # noqa E501 (line too long)
         'group': 'spinal_cord'
         },
    'lesion_ms_mp2rage':
        {'description': 'MS lesion segmentation on cropped MP2RAGE data',
         'long_description': f'This segmentation model for multiple sclerosis lesion segmentation on cropped MP2RAGE-UNIT1 spinal cord data. '
                             f'Uses a 3D U-Net, trained with the nnUNetV2 framework. It is a single-class model outputting binary MS lesions '
                             f'segmentations. Training consisted of MP2RAGE data on UNIT1 contrast at 1.0 mm3 isotropic (322 subjects from '
                             f'3 sites: National Institutes of Health, Bethesda, USA, University Hospital Basel and University of Basel, Basel, '
                             f'Switzerland and Center for Magnetic Resonance in Biology and Medicine, CRMBM-CEMEREM, UMR 7339, CNRS,  '
                             f'Aix-Marseille University, Marseille, France). {CROP_MESSAGE}',
         'url': 'https://github.com/ivadomed/model_seg_ms_mp2rage',
         'models': ['model_seg_ms_lesion_mp2rage'],
         'citation': textwrap.dedent(r"""
             ```bibtex
             @inproceedings{laines2024automatic,
                 author={Laines Medina, N. and Mchinda, S. and Testud, B. and Demorti{\`e}re, S. and Chen, M. and Granziera, G. and Reich, D. and Tsagkas, C. and Cohen-Adad,
                 J. and Callot, V.},
                 title={Automatic Multiple Sclerosis Lesion Segmentation in the Spinal Cord on 3T and 7T MP2RAGE Images},
                 booktitle={Proceedings of the 40th Annual Scientific Meeting of the ESMRMB},
                 year={2024},
                 address={Barcelona, Spain},
                 pages={171}
             }
             ```"""),  # noqa E501 (line too long)
         'group': 'pathology'
         },
    'lesion_ms':
        {'description': 'MS lesion segmentation on spinal cord MRI images',
         'long_description': 'This segmentation model for spinal cord MS lesion segmentation uses a 3D U-Net architecture. It outputs a binary '
                             'segmentation of MS lesions. The model was trained and evaluated on a large-scale dataset comprising 4,428 annotated '
                             'images from 1,849 persons with MS across 23 imaging centers. The dataset included images acquired on GE, Siemens or '
                             'Philips MRI systems, at 1.5T, 3T or 7T, using six distinct MRI contrasts: T2w (n=3,060), T2*w (n=548), PSIR '
                             '(n=363), UNIT1 (reconstructed uniform image from MP2RAGE sequence, n=343), STIR (n=92), and T1w (n=22), and spans '
                             '2D axial (n=2,895), 2D sagittal (n=1,160), and 3D (n=373) acquisition planes. The field-of-view coverage varied '
                             'across sites (brain and upper SC, or SC only). Image resolution exhibited high variability, with an average '
                             '(± standard deviation) of 1.10±1.13 x 0.51±0.24 x 3.27±1.95 mm³ reported in “RPI-” orientation ',
         'url': 'https://github.com/ivadomed/ms-lesion-agnostic',
         'models': ['model_seg_ms_lesion'],
         'citation': textwrap.dedent(r"""
             ```bibtex
            @article{doi:10.1177/13524585261427333,
                author = {Pierre-Louis Benveniste and Laurent Létourneau-Guillon and David Araujo and Lydia Chougar and Dumitru Fetco and Masaaki Hori and Kouhei Kamiya and Steven Messina and Charidimos Tsagkas and Bertrand Audoin and Rohit Bakshi and Elise Bannier and Daniel Blezek and Jean-Christophe Brisset and Virginie Callot and Erik Charlson and Michelle Chen and Olga Ciccarelli and Sarah Demortière and Gilles Edan and Massimo Filippi and Tobias Granberg and Cristina Granziera and Christopher C. Hemond and B. Mark Keegan and Anne Kerbrat and Jan Kirschke and Shannon Kolind and Pierre Labauge and Lisa Eunyoung Lee and Yaou Liu and Caterina Mainero and Julian McGinnis and Nilser Laines Medina and Mark Mühlau and Govind Nair and Kristin P. O’Grady and Jiwon Oh and Russell Ouellette and Alexandre Prat and Daniel S. Reich and Maria A. Rocca and Timothy M. Shepherd and Seth A. Smith and Leszek Stawiarz and Jason Talbott and Roger Tam and Shahamat Tauhid and Anthony Traboulsee and Constantina Andrada Treaba and Paola Valsasina and Zachary Vavasour and Marios Yiannakas and Hervé Lombaert and Julien Cohen-Adad},
                title ={Generalizable spinal cord multiple sclerosis lesion segmentation across MRI contrasts, protocols, and centers},
                journal = {Multiple Sclerosis Journal},
                volume = {32},
                number = {6},
                pages = {598-613},
                year = {2026},
                doi = {10.1177/13524585261427333},
                note ={PMID: 42028790},
                URL = {https://doi.org/10.1177/13524585261427333},
                eprint = {https://doi.org/10.1177/13524585261427333},
                abstract = { Background/Objectives: Characterizing spinal cord multiple sclerosis (MS) lesions in MRI is critical for diagnosis, monitoring, and treatment evaluation. However, current automated approaches for lesion detection and segmentation are typically designed for specific MRI contrasts or acquisition sites, limiting their generalizability in real-world clinical settings where imaging protocols vary widely. This work proposes a robust multi-site, multi-contrast segmentation framework for spinal cord lesions.Methods: The segmentation model was trained and evaluated on a large-scale dataset comprising 4428 annotated images from 1849 persons with MS across 23 imaging centers, encompassing six MRI contrasts (T1w, T2w, T2*w, PSIR, STIR, and UNIT1) acquired at 1.5 tesla (T), 3 T, and 7 T.Results: Likert-type assessment performed by neuroradiologist ratings demonstrated superior generalization of the model compared to existing contrast-specific pipelines (p < 0.01). Additional experiments evaluated robustness across spinal levels, acquisition resolutions, binarization thresholds, and quantitative evaluation on external labeled datasets.Conclusions: The proposed model can achieve accurate and reliable spinal cord MS lesion segmentation across heterogeneous MRI data, addressing a key barrier to clinical translation. The model is available in the Spinal Cord Toolbox v7.2 and higher.Code repository: https://github.com/ivadomed/seg-sc-ms-lesion-multicontrast }
            }
             ```"""),  # noqa E501 (line too long)
         'group': 'pathology'
         },
    'canal':
        {'description': 'Segmentation of spinal canal on all contrasts',
         'long_description': 'This model segments the spinal canal, or in an anatomic definition the dural sac, on all contrasts. '
                             'Uses a 3D U-Net, trained with the nnUNetV2 framework. It is a single-class model outputting the binary canal segmentation. '
                             'Training consisted of an active learning procedure, correcting segmentations with ITK Snap. Last training '
                             'procedure can be found here : https://github.com/ivadomed/model-canal-seg/issues/20 '
                             'Images used present different resolutions, FOV, and pathologies. A script is added as post-processing to '
                             'keep the largest connected component of the segmentation, since spinal canal is connected, to avoid '
                             'false positives segmentations of other anatomical structures.',
         'url': 'https://github.com/ivadomed/model-canal-seg',
         'models': ['model_seg_canal'],
         'citation': None
         },
    'spine':
        {'description': 'Intervertebral discs labeling and vertebrae segmentation',
         'long_description': 'TotalSpineSeg is a tool for automatic instance segmentation of all vertebrae, intervertebral discs (IVDs), '
                             'spinal cord, and spinal canal in MRI images. It is robust to various MRI contrasts, acquisition orientations, '
                             'and resolutions. The model used in TotalSpineSeg is based on nnU-Net as the backbone for training and inference.',
         'url': 'https://github.com/neuropoly/totalspineseg',
         'models': ['model_seg_spine_contrast_agnostic'],
         'citation': None
         },
    'lesion_ms_axial_t2':
        {'description': 'Intramedullary MS lesion and spinal cord segmentation in Axial T2w MRI',
         'long_description': 'This MS lesion segmentation uses a 2D U-Net '
                             'architecture, and was trained with the nnUNetV2 framework. The model outputs '
                             'lesion mask along with the spinal cord segmentation mask. Training and evaluation'
                             'data consisting of highly-anisotropic axial T2w chunks was gathered from 4 sites: Klinikum Rechts der Isar, '
                             'Technical University of Munich, Germany, NYU Langone Medical Center, NY, USA, '
                             'Zuckerberg SF General Hospital, UCSF, CA, USA, and Brigham and Womens Hospital, '
                             'Harvard Medical School, Boston, USA . Unlike typical MS lesion segmentation models, this models works equally well on '
                             'cervical, thoracic and lumbar spinal cord regions.',
         'url': 'https://github.com/ivadomed/model-seg-ms-axial-t2w',
         'models': ['model_seg_ms_sc_lesion_bavaria_quebec_nnunet'],
         'citation': None,
         'group': 'pathology'
         },
    'graymatter':
        {'description': 'Segmentation of gray matter agnostic to MRI contrasts and regions',
         'long_description': 'This model for spinal cord gray matter (GM) segmentation uses a 2D nnU-Net architecture. It outputs a binary '
                             'segmentation. The model was trained and tested on datasets including >20 sites, 3 magnetic field strengths, 9 sequences, '
                             '1367 subjects included: 1.5T-PDw(N = 8), 3T-MGE-T2starw(N = 509), 3T-MTR(N = 21), 3T-PDw(N = 145), 3T-PSIR(N = 176), '
                             '3T-rAMIRA(N = 48), 3T-TSE-T1w(N = 64), 7T-MGE-T2starw(N = 89), 7T-MP2RAGE-T1map(N = 144), 7T-MP2RAGE-UNI(N = 144), '
                             '7T-QSM(N = 14), 7T-SWI(N = 5), acquired in the cervical, thoracic and lumbar regions from healthy controls, pediatrics, '
                             'multiple sclerosis, spinal muscular atrophy, cervical degenerative myelopathy, spinal cord injury, '
                             'amyotrophic lateral sclerosis post-polio syndrome and stroke.',
         'url': 'https://github.com/ivadomed/model-gm-contrast-region-agnostic',
         'models': ['model_seg_gm_contrast_region_agnostic'],
         'citation': None,
         'group': 'gray_matter',
         'priority': 1  # Push gray matter to the top of its eponymous category
         },
}


def get_required_contrasts(task):
    """
    Get required contrasts according to models in tasks.

    :return: list: List of required contrasts
    """
    contrasts_required = set()
    for model in TASKS[task]['models']:
        for contrast in MODELS[model]['contrasts']:
            contrasts_required.add(contrast)

    return list(contrasts_required)


def folder(name_model):
    """
    Return absolute path of deep learning models.

    :param name: str: Name of model.
    :return: str: Folder to model
    """
    return os.path.join(__deepseg_dir__, name_model)


# sc_crop.detect() padding kwargs; see sc_crop.crop.detect() for the meaning of each face.
CROP_PAD_KEYS = ('pad_superior', 'pad_inferior', 'pad_left', 'pad_right', 'pad_anterior', 'pad_posterior')


def _read_source_json(path_model):
    """
    Read an installed model's provenance file, if present.

    :param path_model: str: Path to the installed model folder.
    :return: dict: parsed `source.json` content, or {} if missing.
    """
    source_path = os.path.join(path_model, "source.json")
    if not os.path.isfile(source_path):
        return {}
    with open(source_path, "r") as fp:
        return json.load(fp)


def load_crop_metadata(name_model, path_model):
    """
    Determine an installed model's sc-crop padding, if it uses the sc-crop pipeline at all.

    `MODELS[name_model]['cropped_image']` says whether this model is *expected* to use
    sc-crop -- used before the model is even installed (e.g. to decide whether to expose
    `-box-*` at all). The actual decision, and the padding, come from whether a
    `crop_metadata.json` file is present at the root of the model's release .zip -- so both
    travel with the actual artifact on disk, rather than being hardcoded here (which would be
    wrong for a model installed via `-custom-url`, since that can point to a differently-trained
    artifact). A release only ever bundles this file for a model that actually uses sc-crop, so
    its mere presence is the signal -- there's no separate flag inside the file for this.

    If the model isn't expected to use sc-crop, nothing else is checked. If it is, but no
    `crop_metadata.json` is found:
    - for an official install (no `-custom-url`), this means the release is missing a file it
      should have -- raise, since silently skipping cropping would produce out-of-distribution
      predictions with the model none the wiser.
    - for a `-custom-url` install, the artifact may simply not be a cropped model at all (e.g.
      an older release); treat it as not using sc-crop instead of raising.

    Padding is required to be fully specified in `crop_metadata.json` whenever the model does
    use sc-crop -- callers must not fall back to `sc_crop.detect()`'s own default padding, since
    those defaults can change between `sc-crop` versions, silently shifting the crop a model was
    actually trained on.

    :param name_model: str: Name of model (key into MODELS).
    :param path_model: str: Path to the installed model folder.
    :return: dict: `sc_crop.detect()` padding kwargs, one entry per key in `CROP_PAD_KEYS`, or
        `{}` if this model doesn't use the sc-crop pipeline. Callers should treat a non-empty
        return as "sc-crop is active" (i.e. use `bool(load_crop_metadata(...))`).
    """
    if not MODELS.get(name_model, {}).get('cropped_image', False):
        return {}

    path_json = os.path.join(path_model, "crop_metadata.json")
    if not os.path.isfile(path_json):
        is_custom = _read_source_json(path_model).get('custom', False)
        if not is_custom:
            raise RuntimeError(
                f"Model '{name_model}' should use the sc-crop pipeline, but no "
                f"crop_metadata.json was found in its installed folder ('{path_model}'). "
                f"The release is likely missing this file -- reinstall with `-install`, "
                f"or check the model's release assets."
            )
        return {}

    with open(path_json, "r") as fp:
        crop_metadata = json.load(fp)

    missing_keys = [key for key in CROP_PAD_KEYS if key not in crop_metadata]
    if missing_keys:
        raise RuntimeError(
            f"Model '{name_model}'s crop_metadata.json ('{path_json}') is missing padding "
            f"key(s) {missing_keys} -- all of {list(CROP_PAD_KEYS)} must be specified "
            f"explicitly, so cropping doesn't silently depend on sc_crop's own defaults."
        )
    return {key: crop_metadata[key] for key in CROP_PAD_KEYS}


def is_list_of_urls(url_field):
    """Check if an object is a list of URLs."""
    return isinstance(url_field, list) and all(
        isinstance(url, str) and url.startswith("http")
        for url in url_field
    )


def is_dict_of_lists_of_urls(url_field):
    """Check if an object is a dict of lists of URLs."""
    return isinstance(url_field, dict) and all(
        isinstance(k, str) and is_list_of_urls(v)
        for k, v in url_field.items()
    )


def install_model(name_model, custom_url=None):
    """
    Download and install specified model under SCT installation dir.

    :param name: str: Name of model.
    :param custom_url: list of URLs, either 1 for single-model / single-mirror case or one per fold/seed
        (for models made up of several folds/seeds, e.g. `model_seg_ms_lesion`).
    :return: None
    """
    logger.info(f"\nINSTALLING MODEL: {name_model}")
    url_field = default_url = MODELS[name_model]['url']
    if custom_url is not None:
        if not isinstance(custom_url, list):
            raise ValueError(f"Expected a list of model URLs for custom_url, got: {custom_url}")
        # Single-fold/seed model
        elif len(custom_url) == 1:
            url_field = custom_url
        # Multiple URLs passed, see how many are expected
        else:
            n_urls_expected = len(default_url) if isinstance(default_url, dict) else 1
            # Multi-fold/multi-seed model: one custom URL is expected per fold, in the same order as the default URLs
            if len(custom_url) == n_urls_expected:
                url_field = {seed_name: [url] for seed_name, url in zip(default_url.keys(), custom_url)}
            # Multi-fold model with the wrong number of URLs passed, or
            # single-fold model with multiple URLs passed
            else:
                raise ValueError(
                    f"Expected {n_urls_expected} custom URL(s) for model '{name_model}' "
                    f"but got {len(custom_url)} instead.")
    # List of mirror URLs corresponding to a single model
    if is_list_of_urls(url_field):
        model_urls = url_field
        # Make sure to preserve the internal folder structure for nnUNet-based models (to allow re-use with 3D Slicer)
        urls_used = [download.install_data(model_urls, folder(name_model), dirs_to_preserve=("nnUNetTrainer",))]
    # Dict of lists, with each list corresponding to a different model seed for ensembling
    elif is_dict_of_lists_of_urls(url_field):
        # totalspineseg handles data downloading itself, so just pass the urls along
        if name_model in TASKS['spine']['models']:
            # Totalspineseg expects exactly 1 URL string per model (rather than a list of mirrors)
            dict_urls = {seed_name: (urls[0] if isinstance(urls, list) else urls)
                         for seed_name, urls in url_field.items()}
            tss_init.init_inference(data_path=Path(folder(name_model)), quiet=False, dict_urls=dict_urls,
                                    store_export=False)  # Avoid having duplicate .zip files stored on disk
            urls_used = url_field
            # For totalspineseg, for now we need to copy its custom trainer to the `nnunetv2` folder
            # Long-term, this may not be necessary if we end up using the fork of `nnunetv2`:
            #    See: https://github.com/neuropoly/totalspineseg/issues/126
            # Since `install_model`` will only be invoked on first run or if model is outdated, it's safe to always overwrite
            auglab_add_trainer.add_trainer(trainer_name="nnUNetTrainerDAExt", overwrite=True)
        else:
            urls_used = {}
            for i, (seed_name, model_urls) in enumerate(url_field.items()):
                # For the ms_lesion model, we need to regroup the folds together
                # For lesion_ms, we can extract all the folds to the same `nnunetTrainer` directory
                if name_model == "model_seg_ms_lesion":
                    target_directory = folder(name_model)
                    dirs_to_preserve = ("nnUNetTrainer",)
                # For other multi-seed models, create subfolders for each seed
                else:
                    target_directory = folder(os.path.join(name_model, seed_name))
                    dirs_to_preserve = ()
                logger.info(f"\nInstalling '{seed_name}'...")
                urls_used[seed_name] = [download.install_data(model_urls, target_directory, keep=(i > 0),
                                                              dirs_to_preserve=dirs_to_preserve)]
    else:
        raise ValueError(f"Invalid url field in MODELS: {url_field}")
    # Write `source.json` (for model provenance / updating)
    source_dict = {
        'model_name': name_model,
        'model_urls': urls_used,
        # NB: If a custom URL is used, then it would just get overwritten as "out of date" when running the task
        #     So, we add a flag to tell `sct_deepseg` *not* to reinstall the model if a custom URL was used.
        'custom': bool(custom_url)
    }
    with open(os.path.join(folder(name_model), "source.json"), "w") as fp:
        json.dump(source_dict, fp, indent=4)


def install_default_models():
    """
    Download all default models and install them under SCT installation dir.

    :return: None
    """
    for name_model, value in MODELS.items():
        if value['default']:
            install_model(name_model)


def is_up_to_date(path_model):
    """
    Determine whether an on-disk model is up-to-date by comparing
    the URL used to download the model with the latest mirrors.

    :return: bool: whether the model is up-to-date
    """
    source_dict = _read_source_json(path_model)
    if not source_dict:
        logger.warning("Provenance file 'source.json' missing!")
        return False  # NB: This will force a reinstall
    model_name = source_dict["model_name"]
    if model_name not in MODELS:
        logger.warning(f"Model name '{model_name}' from source.json does not match model names in SCT source code.")
        return False

    expected_model_urls = MODELS[model_name]['url'].copy()
    actual_model_urls = source_dict["model_urls"]
    # old format for cached/on-disk model may have been saved as a single string
    if isinstance(actual_model_urls, str):
        actual_model_urls = [actual_model_urls]

    if "custom" in source_dict and source_dict["custom"] is True:
        logger.warning(f"Using custom model from URL '{actual_model_urls}'.")
        return True  # Don't reinstall the model if the 'custom' flag is set (since custom URLs would fail comparison)

    # Single-seed models (list of URLs)
    if is_list_of_urls(expected_model_urls) and is_list_of_urls(actual_model_urls):
        if not any(url in expected_model_urls for url in actual_model_urls):
            return False

    # Multi-seed, ensemble models (dict of lists of URLs, one list per seed)
    elif is_dict_of_lists_of_urls(expected_model_urls) and is_dict_of_lists_of_urls(actual_model_urls):
        expected_seeds = set(expected_model_urls)
        actual_seeds = set(actual_model_urls)

        if unexpected_seeds := actual_seeds - expected_seeds:
            logger.warning(f"unexpected seeds: {list(unexpected_seeds)}")
            return False
        if missing_seeds := expected_seeds - actual_seeds:
            logger.warning(f"missing seeds: {list(missing_seeds)}")
            return False

        for seed in actual_seeds:
            if not set(actual_model_urls[seed]).issubset(expected_model_urls[seed]):
                logger.warning(f"wrong version for {seed}: {actual_model_urls[seed]}")
                return False

    else:
        logger.warning("Mismatch between 'source.json' URL format and SCT source code URLs")
        return False

    logger.info(f"Model '{model_name}' is up to date (Source: {actual_model_urls})")
    return True


def is_valid(path_models):
    """
    Check if model paths have the necessary files and follow naming conventions:
    - Folder should have the same name as the enclosed files.

    :param path_models: str or list: Absolute path(s) to folder(s) that enclose the model files.
    """
    def _is_valid(path_model):
        return has_ivadomed_files(path_model) or has_ckpt_files(path_model) or has_pth_files(path_model)
    # Adapt the function so that it can be used on single paths (str) or lists of paths
    if not isinstance(path_models, list):
        path_models = [path_models]
    return all(_is_valid(path) for path in path_models)


def has_ivadomed_files(path_model):
    """
    Check if model path contains A) a named .pt/.onnx model file and B) a named ivadomed json configuration file
    """
    name_model = Path(path_model).name
    path_pt = os.path.join(path_model, f'{name_model}.pt')
    path_onnx = os.path.join(path_model, f'{name_model}.onnx')
    path_json = os.path.join(path_model, f'{name_model}.json')
    return (os.path.exists(path_pt) or os.path.exists(path_onnx)) and os.path.exists(path_json)


def has_ckpt_files(path_model):
    """
    Check if model path contains any checkpoint files (used by non-ivadomed MONAI models)
    """
    return bool(glob.glob(os.path.join(path_model, '**', '*.ckpt'), recursive=True))


def has_pth_files(path_model):
    """
    Check if model path contains any serialized PyTorch state dictionary files (used by non-ivadomed nnUNet models)
    """
    return bool(glob.glob(os.path.join(path_model, '**', '*.pth'), recursive=True))


def check_model_software_type(path_model):
    """
    Determine the software used to train the model based on the types of files in the model folder
    """
    if has_ivadomed_files(path_model):
        return 'ivadomed'
    elif has_ckpt_files(path_model):
        return 'monai'
    elif has_pth_files(path_model):
        return 'nnunet'
    else:
        raise ValueError("Model type cannot be determined.")


def _priority_then_alpha(dict_key: str) -> (int, str):
    # NOTE: The negation is required to allow for an "alternating" ascending/descending sort
    return -TASKS[dict_key].get('priority', 0), dict_key


def _group_tasks(sorted_task_keys: list) -> dict[str, list[str]]:
    """
    Group our tasks by their task labels. The order of tasks provided determines the order of groups as well, so you
    should probably sort it how you like before passing it into this function.
    """
    groups = dict()

    # Keep track of tasks without a group separate, as they need to be appended last
    other_set = []

    # Iterate through our tasks to find their groups
    for k in sorted_task_keys:
        task_group = TASKS[k].get('group', None)
        if task_group is None:
            other_set.append(k)
        else:
            group_vals = groups.get(task_group, [])
            group_vals.append(k)
            groups[task_group] = group_vals

    # Add the "other" group last, if any models fell within it
    if len(other_set) > 0:
        groups['other'] = other_set

    # Return the result
    return groups


def list_tasks_string():
    # Some set-up to make uniformity easier to maintain
    table_width = 80
    task_width = 30  # Description width does not need to be defined, as we are left-justified

    # Display coloured output
    color = {True: 'LightGreen', False: 'LightRed'}
    table = [
        "TASK".ljust(task_width), "DESCRIPTION\n",
        "=" * table_width, "\n",
    ]

    sorted_groups = _group_tasks(sorted(TASKS, key=_priority_then_alpha))

    for group_name, group_tasks in sorted_groups.items():
        # Format the group name in all-caps w/ underscores replaced with spaces
        formatted_group_name = group_name.replace("_", " ").upper()
        # Add the header for this group
        table.extend([
            "\n", formatted_group_name, "\n",
            "-" * table_width, "\n",
        ])
        # Print out the task details
        for task_name in group_tasks:
            # Grab the details for this task from our dict
            task_details = TASKS[task_name]
            # Extract and format the metadata for the task
            path_models = [folder(name_model) for name_model in task_details['models']]
            path_models = [find_model_folder_paths(path) for path in path_models]
            are_models_valid = [is_valid(path_model) for path_model in path_models]
            task_status = stylize(task_name.ljust(task_width), color[all(are_models_valid)])
            description_status = stylize(task_details['description'], color[all(are_models_valid)])
            # Add it to the task list
            table.extend([task_status, description_status, "\n"])

    # Add a legend to denote which tools are installed or not and the end
    table.extend([
        "\nLegend: ", stylize("installed", color[True]), " | ", stylize("not installed", color[False]),
        "\n\nTo read in-depth descriptions of the training data, model architecture, etc. "
        "used for these tasks, type the following command:"
        "\n\n    ", stylize("sct_deepseg -task-details", ['LightBlue', 'Bold']),
    ])
    return ''.join(table)


def display_list_tasks():
    # Define the labels which can be used for each entry
    task_label = "TASK:"
    contrast_label = "CONTRAST:"
    model_label = "MODELS:"
    description_label = "DESCRIPTION:"
    url_label = "URL:"
    installed_label = "INSTALLED:"

    # Get the left-justified indent required for the longest of these labels
    padded_len = max([len(x) for x in [task_label, contrast_label, model_label, description_label, url_label]])

    # Sort the tasks by priority, then alphanumerically
    sorted_tasks = sorted(TASKS, key=_priority_then_alpha)

    # Iterate through each task and add its corresponding entry to the output
    for task_name in sorted_tasks:
        # Grab the details of the task from the TASKS dict
        task_details = TASKS[task_name]

        # Lead with the task name, bolded to draw the user's attention
        print(f"{task_label.ljust(padded_len)} {stylize(task_name, 'Bold')}")

        # List out the valid contrasts for this model
        contrast_str = ', '.join(get_required_contrasts(task_name))
        print(f"{contrast_label.ljust(padded_len)} {contrast_str}")

        # List out the model(s) used by this task
        task_models = task_details['models']
        model_paths = [find_model_folder_paths(folder(path)) for path in task_models]
        # Filter out models which are missing necessary files to avoid misleading the user
        model_validity = [is_valid(path_model) for path_model in model_paths]
        model_str = ', '.join([model_name for model_name, validity in zip(task_models, model_validity)])
        print(f"{model_label.ljust(padded_len)} {model_str}")

        # Write out the "long" description, wrapped to fit within the console the user is using
        formatted_description = '\n'.join(textwrap.wrap(
            text=task_details['long_description'],
            # Account for our padding left padding too!
            width=shutil.get_terminal_size()[0] - padded_len,
            # +1 padding to ensure at least 1 space always separates the labels and contents
            subsequent_indent=' ' * (padded_len + 1),

        ))
        print(f"{description_label.ljust(padded_len)} {formatted_description}")

        # The URL where the model was downloaded from; 'cyan' formatting is forced in case the console doesn't do it
        formatted_url = stylize(task_details['url'], styles='Cyan')
        print(f"{url_label.ljust(padded_len)} {formatted_url}")

        # Whether the model is installed already or not
        formatted_installed = (
            stylize("Yes", 'LightGreen') if all(model_validity)
            else stylize("No", 'LightRed')
        )
        print(f"{installed_label.ljust(padded_len)} {formatted_installed}")

        # Add a separating newline
        print('')


def get_metadata(folder_model):
    """
    Get metadata from json file located in folder_model

    :param path_model: str: Model folder
    :return: dict
    """
    fname_metadata = os.path.join(folder_model, f'{os.path.basename(folder_model)}.json')
    with open(fname_metadata, "r") as fhandle:
        metadata = json.load(fhandle)
    return metadata


def find_model_folder_paths(path_model):
    """
    Search for the presence of model subfolders within the main model folder. If they exist,
    then the model folder is actually an ensemble of models, so return a list of folders.
    If they don't exist, then return the original `path_model` (but as a list, to ensure code compatibility).

    :param path_model: Absolute path to folder that encloses the model files.
    :return: list: Either a list of ensemble subfolders, or a list containing the original model folder path.
    """
    name_model = path_model.rstrip(os.sep).split(os.sep)[-1]
    # Check to see if model folder contains subfolders with the model name (i.e. ensembling)
    model_subfolders = [folder[0] for folder in os.walk(path_model)  # NB: `[0]` == folder name for os.walk
                        if folder[0].endswith(name_model) and folder[0] != path_model]
    # If it does, then these are the "true" model subfolders. Otherwise, return the original path as a list.
    return model_subfolders if model_subfolders else [path_model]
