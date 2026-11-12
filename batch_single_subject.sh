#!/bin/bash
#
# Example of commands to process multi-parametric data of the spinal cord
# For information about acquisition parameters, see: www.spinalcordmri.org/protocols
#
# Notes:
#   - Many of the commands in this script are commented out (start with "# "). These commands won't be run by default,
#     as many of them involve manual steps (e.g. via an interactive pop-up interface), or may involve data-specific
#     configuration. Often these commands correspond to the "try at home" slides in the SCT Course slide deck,
#     (denoted by a "red circle" in the top-right corner of the corresponding slide). Feel free to "uncomment" the
#     commands by removing the "# " symbol at the start of the line, and experiment with them on your own data.
#   - The parameters were chosen to suit SCT's sample tutorial data. With your data,
#     it is worthwhile to explore the various parameters and tweak them to your situation.
#
# tested with Spinal Cord Toolbox (v6.5)

# Script utilities
# ======================================================================================================================

# If a command fails, set -e will make the whole script exit, instead of just resuming on the next line
set -ve

# For full verbose, uncomment the next line
# set -x

# get starting time:
start=$(date +%s)

# Fetch OS type
if uname -a | grep -i  darwin > /dev/null 2>&1; then
  # OSX
  open_command="open"
elif uname -a | grep -i  linux > /dev/null 2>&1; then
  # Linux
  open_command="xdg-open"
fi

# download example data and enter our data directory
sct_download_data -d sct_course_data
cd "$SCT_DIR/data/sct_course_data/single_subject/data/"

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# display starting time:
echo -e "\nStarted at: $(date +%x_%r)"

# If fsleyes hasn't been installed, provide a subsitute function to avoid crashing the script
# Source: https://unix.stackexchange.com/a/497540
if ! command -v fsleyes > /dev/null; then
  fsleyes() {
    printf 'WARNING: FSLeyes is not installed, so the following command was skipped:\nfsleyes %s\n' "${*@Q}";
  };
fi



# ======================================================================================================================
# START OF SCRIPT
# ======================================================================================================================

# Spinal cord segmentation (dataset: data_spinalcord-segmentation)
# ======================================================================================================================

# Go to T2 folder
cd t2
# Spinal cord segmentation (using the new 2024 contrast-agnostic method)
sct_deepseg spinalcord -i t2.nii.gz -qc ~/qc_singleSubj
# The default output is t2_seg.nii.gz
# You can also choose your own output filename using the “-o” argument
sct_deepseg spinalcord -i t2.nii.gz -o test/t2_seg_2.nii.gz

# To check the QC report, use your web browser to open the file qc_singleSubj/qc/index.html, which has been created in
# your home directory

# View the rest of the `sct_deepseg` tasks
sct_deepseg -h
# See also: https://spinalcordtoolbox.com/stable/user_section/command-line/sct_deepseg.html



# Vertebral labeling (dataset: data_vertebral-labeling)
# ======================================================================================================================

# Vertebral disc labeling
sct_deepseg spine -i t2.nii.gz -label-vert 1 -qc ~/qc_singleSubj

# Full spinal segmentation (Vertebrae, Intervertebral discs, Spinal cord and Spinal canal)
# Segment using totalspineseg
sct_deepseg spine -i t2.nii.gz -qc ~/qc_singleSubj
# Check results using FSLeyes
fsleyes t2.nii.gz -cm greyscale t2_step1_canal.nii.gz -cm YlOrRd -a 70.0 t2_step1_cord.nii.gz -cm YlOrRd -a 70.0 t2_totalspineseg_discs.nii.gz -cm subcortical -a 70.0 t2_step1_output.nii.gz -cm subcortical -a 70.0 t2_step2_output.nii.gz -cm subcortical -a 70.0 &
# Check QC report: Go to your browser and do "refresh".

# Optionally, you can use the generated disc labels to create a labeled segmentation
# sct_label_vertebrae -i t2.nii.gz -s t2_seg.nii.gz -c t2 -discfile t2_totalspineseg_discs.nii.gz

# If you wish to bypass the disc labeling model entirely and instead wish to use the legacy
# vertebral labeling method (`sct_label_vertebrae`), you can do so.
# Note: This approach is no longer recommended. Instead, use the disc labels directly in subsequent commands (e.g. `sct_process_segmentation`).
# sct_label_vertebrae -i t2.nii.gz -s t2_seg.nii.gz -c t2 -qc ~/qc_singleSubj

# Optionally, if the legacy `sct_label_vertebrae` fails as well, you can initialize the command
# by manually labeling the c2-c3 disc.
# sct_label_utils -i t2.nii.gz -create-viewer 3 -o label_c2c3.nii.gz -msg "Click at the posterior tip of C2/C3 inter-vertebral disc"
# sct_label_vertebrae -i t2.nii.gz -s t2_seg.nii.gz -c t2 -initlabel label_c2c3.nii.gz -qc ~/qc_singleSubj

# If generating a labeled segmentation using `sct_label_vertebrae`, you can then extract vertebral
# body labels from the segmentation using `-vert-body`.
# sct_label_utils -i t2_seg_labeled.nii.gz -vert-body 3,9 -o t2_labels_vert.nii.gz

# Create labels at C3 and T2 mid-vertebral levels. These labels are needed for template registration.
sct_label_utils -i t2_totalspineseg_discs.nii.gz -keep 3,9 -o t2_labels_vert.nii.gz
# Generate a QC report to visualize the two selected labels on the anatomical image
sct_qc -i t2.nii.gz -s t2_labels_vert.nii.gz -p sct_label_utils -qc ~/qc_singleSubj

# OPTIONAL: You might want to completely bypass sct_label_vertebrae and do the labeling manually. In that case, we
# provide a viewer to do so conveniently. In the example command below, we will create labels at the inter-vertebral
# discs C2-C3 (value=3), C3-C4 (value=4) and C4-C5 (value=5).
# sct_label_utils -i t2.nii.gz -create-viewer 3,4,5 -o labels_disc.nii.gz -msg "Place labels at the posterior tip of each inter-vertebral disc. E.g. Label 3: C2/C3, Label 4: C3/C4, etc."


# Shape-based analysis (dataset: data_shape-metric-computation)
# ======================================================================================================================

# Compute cross-sectional area (CSA) of spinal cord and average it across levels C2 and C3
sct_process_segmentation -i t2_seg.nii.gz -vert 2:3 -discfile t2_totalspineseg_discs.nii.gz -o csa_c2c3.csv
# Aggregate CSA value per level (including new anat-based symmetry metrics)
sct_process_segmentation -i t2_seg.nii.gz -anat t2.nii.gz -vert 2:3 -discfile t2_totalspineseg_discs.nii.gz -perlevel 1 -o csa_perlevel.csv
# Aggregate CSA value per slices
sct_process_segmentation -i t2_seg.nii.gz -z 30:35 -discfile t2_totalspineseg_discs.nii.gz -perslice 1 -o csa_perslice.csv

# A drawback of vertebral level-based CSA is that it doesn’t consider neck flexion and extension.
# To overcome this limitation, the CSA can instead be computed using the distance to a reference point.
# Here, we use the Pontomedullary Junction (PMJ), since the distance from the PMJ along the centerline
# of the spinal cord will vary depending on the position of the neck.
sct_detect_pmj -i t2.nii.gz -c t2 -qc ~/qc_singleSubj
# Check the QC to make sure PMJ was properly detected, then compute CSA using the distance from the PMJ:
sct_process_segmentation -i t2_seg.nii.gz -pmj t2_pmj.nii.gz -pmj-distance 60 -pmj-extent 30 -o csa_pmj.csv -qc ~/qc_singleSubj -qc-image t2.nii.gz

# The above commands will output the metrics in the subject space (with the original image's slice numbers)
# However, you can get the corresponding slice number in the PAM50 space by using the flag `-normalize-PAM50 1`
sct_process_segmentation -i t2_seg.nii.gz -discfile t2_totalspineseg_discs.nii.gz -perslice 1 -normalize-PAM50 1 -o csa_pam50.csv



# Quantifying spinal cord compression using maximum spinal cord compression (MSCC) and normalizing with database of healthy controls (dataset: data_compression)
# ======================================================================================================================

cd ../t2_compression
# Segment the spinal cord of the compressed spine
sct_deepseg spinalcord -i t2_compressed.nii.gz -qc ~/qc_singleSubj
# Label the vertebrae using the compressed spinal cord segmentation
sct_label_vertebrae -i t2_compressed.nii.gz -s t2_compressed_seg.nii.gz -c t2 -qc ~/qc_singleSubj
# Generate labels for each spinal cord compression site.
# Note: Normally this would be done manually using fsleyes' "Edit mode -> Create mask" functionality. (Uncomment below)
#
# fsleyes t2_compressed.nii.gz &
#
# However, since this is an automated script with example data, we will place the labels at known locations for the
# sake of reproducing the results in the tutorial.
sct_label_utils -i t2_compressed.nii.gz -create 30,152,99,1.0:30,156,118,1.0:30,157,140,1.0:31,160,159,1.0 -o t2_compressed_labels-compression.nii.gz
# Compute ratio between AP-diameter at level of compression vs. above/below
sct_compute_compression -i t2_compressed_seg.nii.gz -vertfile t2_compressed_seg_labeled.nii.gz -l t2_compressed_labels-compression.nii.gz -metric diameter_AP -normalize-hc 0 -o ap_ratio.csv
# Compute ratio of AP diameter, normalized with healthy controls using `-normalize-hc 1`.
sct_compute_compression -i t2_compressed_seg.nii.gz -vertfile t2_compressed_seg_labeled.nii.gz -l t2_compressed_labels-compression.nii.gz -metric diameter_AP -normalize-hc 1 -o ap_ratio_norm_PAM50.csv

# Canal segmentation
sct_deepseg canal -i t2_compressed.nii.gz -o t2_compressed_canal_seg.nii.gz -qc ~/qc_singleSubj
# Check results using FSLeyes
fsleyes t2.nii.gz -cm greyscale t2_canal_seg_seg.nii.gz -cm red -a 70.0 &
# Compute aSCOR (Adapted Spinal Cord Occupation Ratio)
# i.e. Spinal cord to canal ratio using the canal seg
sct_compute_ascor -i-SC t2_compressed_seg.nii.gz -i-canal t2_compressed_canal_seg.nii.gz -perlevel 1 -o ascor.csv



# Registration to template (dataset: data_template-registration)
# ======================================================================================================================

cd ../t2

# Register t2->template.
sct_register_to_template -i t2.nii.gz -s t2_seg.nii.gz -ldisc t2_labels_vert.nii.gz -c t2 -qc ~/qc_singleSubj
# Note: By default the PAM50 template is selected. You can also select your own template using flag -t.

# Register t2->template with modified parameters (advanced usage of `-param`)
sct_register_to_template -i t2.nii.gz -s t2_seg.nii.gz -ldisc t2_labels_vert.nii.gz -qc ~/qc_singleSubj -ofolder advanced_param -c t2 -param step=1,type=seg,algo=rigid:step=2,type=seg,metric=CC,algo=bsplinesyn,slicewise=1,iter=3:step=3,type=im,metric=CC,algo=syn,slicewise=1,iter=2

# Register t2->template with large FOV (e.g. C2-L1) using `-ldisc` option
# sct_register_to_template -i t2.nii.gz -s t2_seg.nii.gz -ldisc t2_totalspineseg_discs.nii.gz -c t2

# Register t2->template in compressed cord (example command)
# In case of highly compressed cord, the algo columnwise can be used, which allows for more deformation than bsplinesyn.
# NB: In the example below, the registration is done in the subject space (no straightening) using a single label point at disc C3-C4 (<LABEL_DISC>).
## sct_register_to_template -i "$IMAGE" -s "$SEGMENTATION" -ldisc "$LABEL_DISC" -ref subject -param step=1,type=seg,algo=centermassrot:step=2,type=seg,algo=columnwise

# Warp template objects (T2, cord segmentation, vertebral levels, etc.). Here we use -a 0 because we don’t need the
# white matter atlas at this point.
sct_warp_template -d t2.nii.gz -w warp_template2anat.nii.gz -a 0 -qc ~/qc_singleSubj
# Note: A folder label/template/ is created, which contains template objects in the space of the subject. The file
#       info_label.txt lists all template files.

# Check results using FSLeyes
fsleyes t2.nii.gz -cm greyscale -a 100.0 label/template/PAM50_t2.nii.gz -cm greyscale -dr 0 4000 -a 100.0 label/template/PAM50_gm.nii.gz -cm red-yellow -dr 0.4 1 -a 50.0 label/template/PAM50_wm.nii.gz -cm blue-lightblue -dr 0.4 1 -a 50.0 &



# Registering additional contrasts (MT registration to T2 template) (dataset: data_coregistration)
# ======================================================================================================================

# Go to mt folder
cd ../mt
# Segment cord
sct_deepseg spinalcord -i mt1.nii.gz -qc ~/qc_singleSubj

# Create a close mask around the spinal cord for more accurate registration (i.e. does not account for surrounding
# tissue which could move independently from the cord)
sct_create_mask -i mt1.nii.gz -p centerline,mt1_seg.nii.gz -size 35mm -f cylinder -o mask_mt1.nii.gz

# Register template->mt1. The flag -initwarp ../t2/warp_template2anat.nii.gz initializes the registration using the
# template->t2 transformation which was previously estimated
sct_register_multimodal -i "${SCT_DIR}/data/PAM50/template/PAM50_t2.nii.gz" -iseg "${SCT_DIR}/data/PAM50/template/PAM50_cord.nii.gz" -d mt1.nii.gz -dseg mt1_seg.nii.gz -m mask_mt1.nii.gz -initwarp ../t2/warp_template2anat.nii.gz -param step=1,type=seg,algo=centermass:step=2,type=seg,algo=bsplinesyn,slicewise=1,iter=3 -owarp warp_template2mt.nii.gz -qc ~/qc_singleSubj
# Tips: Here we only use the segmentations (type=seg) to minimize the sensitivity of the registration procedure to
#       image artifacts.
# Tips: Step 1: algo=centermass to align source and destination segmentations, then Step 2: algo=bpslinesyn to adapt the
#       shape of the cord to the mt modality (in case there are distortions between the t2 and the mt scan).

# OPTIONAL: Registration with -initwarp requires pre-registration, but in some cases you won't have an anatomical image to do a pre-registration. So, as an alternative, you can register the template directly to a metric image. For that, you just need to create one or two labels in the metric space. For example, if you know that your FOV is centered at C3/C4 disc, then you can create a label automatically with:
# sct_label_utils -i mt1_seg.nii.gz -create-seg-mid 4 -o label_c3c4.nii.gz
# Then, you can register to the template.
# Note: In case the metric image has axial resolution with thick slices, we recommend to do the registration in the subject space (instead of the template space), without cord straightening.
# sct_register_to_template -i mt1.nii.gz -s mt1_seg.nii.gz -ldisc label_c3c4.nii.gz -ref subject -param step=1,type=seg,algo=centermassrot:step=2,type=seg,algo=bsplinesyn,slicewise=1

# Warp template
sct_warp_template -d mt1.nii.gz -w warp_template2mt.nii.gz -a 0 -qc ~/qc_singleSubj
# Check results using FSLeyes
fsleyes mt1.nii.gz -cm greyscale -a 100.0 label/template/PAM50_t2.nii.gz -cm greyscale -dr 0 4000 -a 100.0 label/template/PAM50_gm.nii.gz -cm red-yellow -dr 0.4 1 -a 50.0 label/template/PAM50_wm.nii.gz -cm blue-lightblue -dr 0.4 1 -a 50.0 &



# Registering additional contrasts (MT0/MT1 coregistration to compute MTR) (dataset: data_mtr-computation)
# ======================================================================================================================

# Segment cord
sct_deepseg spinalcord -i mt1.nii.gz -qc ~/qc_singleSubj

# Create a close mask around the spinal cord for more accurate registration (i.e. does not account for surrounding
# tissue which could move independently from the cord)
sct_create_mask -i mt1.nii.gz -p centerline,mt1_seg.nii.gz -size 35mm -f cylinder -o mask_mt1.nii.gz

# Register mt0->mt1 using z-regularized slicewise translations (algo=slicereg)
sct_register_multimodal -i mt0.nii.gz -d mt1.nii.gz -dseg mt1_seg.nii.gz -m mask_mt1.nii.gz -param step=1,type=im,algo=slicereg,metric=CC -x spline -qc ~/qc_singleSubj
# Check results using FSLeyes
fsleyes mt1.nii.gz mt0_reg.nii.gz &
# Compute MTR
sct_compute_mtr -mt0 mt0_reg.nii.gz -mt1 mt1.nii.gz
# Note: MTR is given in percentage.



# Registering additional contrasts (contrast-agnostic registration with deep learning between T2 and T1) (dataset: data_contrast-agnostic-registration)
# ======================================================================================================================

# Go to T2 folder
cd ../t2

# Segment the spinal cord on T2-weighted data
sct_deepseg spinalcord -i t2.nii.gz -qc ~/qc_singleSubj
# Create a mask around the spinal cord
sct_create_mask -i t2.nii.gz -p centerline,t2_seg.nii.gz -size 35mm -o mask_t2.nii.gz
# Crop around the spinal cord to speed up and improve the subsequent registration
sct_crop_image -i t2.nii.gz -m mask_t2.nii.gz

# Go to T1 folder
cd ../t1

# Segment the spinal cord on T1-weighted data
sct_deepseg spinalcord -i t1.nii.gz -qc ~/qc_singleSubj
# Create a mask around the spinal cord
sct_create_mask -i t1.nii.gz -p centerline,t1_seg.nii.gz -size 35mm -o mask_t1.nii.gz
# Crop around the spinal cord to speed up and improve the subsequent registration
sct_crop_image -i t1.nii.gz -m mask_t1.nii.gz

# Register T1->T2 using the contrast-agnostic deep learning method
# Note: We use the cropped images to focus the registration on the spinal cord region
# Note: The destination segmentation is provided for QC reporting only
sct_register_multimodal -i t1_crop.nii.gz -d ../t2/t2_crop.nii.gz -param step=1,type=im,algo=dl -qc ~/qc_singleSubj -dseg ../t2/t2_seg.nii.gz



# Registering additional contrasts (T2 lumbar data) (dataset: data_lumbar-registration)
# ======================================================================================================================

cd ../t2_lumbar

# Use lumbar-specific `sct_deepseg` model to segment the spinal cord
sct_deepseg sc_lumbar_t2 -install
sct_deepseg sc_lumbar_t2 -i t2_lumbar.nii.gz -qc ~/qc_singleSubj

# Generate labels for the 2 spinal cord landmarks: cauda equinea ('99') and T9-T10 disc ('17')
# Note: Normally this would be done manually using fsleyes' "Edit mode -> Create mask" functionality. (Uncomment below)
#
# fsleyes t2.nii.gz &
#
# However, since this is an automated script with example data, we will place the labels at known locations for the
# sake of reproducing the results in the tutorial.
sct_label_utils -i t2_lumbar.nii.gz -create 27,76,187,17:27,79,80,60 -o t2_lumbar_labels.nii.gz -qc ~/qc_singleSubj

# generate a QC report for the lumbar labels
sct_qc -i t2_lumbar.nii.gz -s t2_lumbar_labels.nii.gz -p sct_label_utils -qc ~/qc_singleSubj

# Register the image to the template using segmentation and labels
sct_register_to_template -i t2_lumbar.nii.gz -s t2_lumbar_seg.nii.gz -ldisc t2_lumbar_labels.nii.gz -c t2 -qc ~/qc_singleSubj -param step=1,type=seg,algo=centermassrot:step=2,type=seg,algo=bsplinesyn,metric=MeanSquares,iter=3,slicewise=0:step=3,type=im,algo=syn,metric=CC,iter=3,slicewise=0



# Gray matter segmentation (GM/WM seg) (dataset: data_gm-wm-segmentation)
# ======================================================================================================================

# Go to T2*-weighted data, which has good GM/WM contrast and high in-plane resolution
cd ../t2s
# Segment gray matter (check QC report afterwards)
sct_deepseg graymatter -i t2s.nii.gz -o t2s_gmseg.nii.gz -qc ~/qc_singleSubj
# Spinal cord segmentation
sct_deepseg spinalcord -i t2s.nii.gz -qc ~/qc_singleSubj
# Subtract GM segmentation from cord segmentation to obtain WM segmentation
# Note that we use the flag -thr 0 in case some voxels in the GM segmentation are *not* included in the cord
# segmentation. That would results in voxels in the WM segmentation having the value “-1”, which would cause issues
# with the registration.
sct_maths -i t2s_seg.nii.gz -sub t2s_gmseg.nii.gz -thr 0 -o t2s_wmseg.nii.gz



# Gray matter segmentation (Shape-based analysis and metric extraction) (dataset: data_gm-wm-metric-computation)
# ======================================================================================================================

# Compute cross-sectional area (CSA) of the gray and white matter for all slices in the volume.
# Note: Here we use the flag -angle-corr 0, because we do not want to correct the computed CSA by the cosine of the
# angle between the cord centerline and the S-I axis: we assume that slices were acquired orthogonally to the cord.
sct_process_segmentation -i t2s_wmseg.nii.gz -o csa_wm_perslice.csv -perslice 1 -angle-corr 0
sct_process_segmentation -i t2s_gmseg.nii.gz -o csa_gm_perslice.csv -perslice 1 -angle-corr 0

# You can also use the binary masks to extract signal intensity from MRI data.
# The example below will show how to use the GM and WM segmentations to quantify T2* signal intensity, as done in
# [Martin et al. PLoS One 2018].
# Quantify average WM and GM signal between slices 2 and 12.
sct_extract_metric -i t2s.nii.gz -f t2s_wmseg.nii.gz -method bin -z 2:12 -o t2s_value.csv
sct_extract_metric -i t2s.nii.gz -f t2s_gmseg.nii.gz -method bin -z 2:12 -o t2s_value.csv -append 1
# Note: the flag -append enables to append a new result at the end of an already-existing csv file.



# Gray matter segmentation (Improving registration results using binary segmentation masks) (dataset: data_improving-registration-with-gm-seg)
# ======================================================================================================================

# Register template->t2s (using warping field generated from template<->t2 registration)
# Tips: Here we use the WM seg for the iseg/dseg fields in order to account for both the cord and the GM shape.
sct_register_multimodal -i "${SCT_DIR}/data/PAM50/template/PAM50_t2s.nii.gz" -iseg "${SCT_DIR}/data/PAM50/template/PAM50_wm.nii.gz" -d t2s.nii.gz -dseg t2s_wmseg.nii.gz -initwarp ../t2/warp_template2anat.nii.gz -initwarpinv ../t2/warp_anat2template.nii.gz -owarp warp_template2t2s.nii.gz -owarpinv warp_t2s2template.nii.gz -param step=1,type=seg,algo=rigid:step=2,type=seg,metric=CC,algo=bsplinesyn,slicewise=1,iter=3:step=3,type=im,metric=CC,algo=syn,slicewise=1,iter=2 -qc ~/qc_singleSubj
# Warp template
sct_warp_template -d t2s.nii.gz -w warp_template2t2s.nii.gz -qc ~/qc_singleSubj
# Compute vertebral level-based metrics using warped template (needed for the template's vertlevel file)
# FIXME: In the dataset for this section, we only include t2s_wmseg. We should
#   also amend the dataset to include t2s_gmseg.
# Segment gray matter (check QC report afterwards)
sct_deepseg graymatter -i t2s.nii.gz -o t2s_gmseg.nii.gz -qc ~/qc_singleSubj
# Spinal cord segmentation
sct_deepseg spinalcord -i t2s.nii.gz -qc ~/qc_singleSubj
sct_process_segmentation -i t2s_gmseg.nii.gz -vert 2:5 -perlevel 1 -o csa_gm.csv -centerline t2s_seg.nii.gz -centerline-exclude-missing 1
sct_process_segmentation -i t2s_wmseg.nii.gz -vert 2:5 -perlevel 1 -o csa_wm.csv -centerline t2s_seg.nii.gz -centerline-exclude-missing 1

# Register another metric while reusing newly-created GM-informed warping fields
cd ../mt
# Register template->mt using `-initwarp` with t2s to account for GM segmentation
sct_register_multimodal -i "${SCT_DIR}/data/PAM50/template/PAM50_t2.nii.gz" -iseg "${SCT_DIR}/data/PAM50/template/PAM50_cord.nii.gz" -d mt1.nii.gz -dseg mt1_seg.nii.gz -param step=1,type=seg,algo=centermass:step=2,type=seg,algo=bsplinesyn,slicewise=1,iter=3 -m mask_mt1.nii.gz -initwarp ../t2s/warp_template2t2s.nii.gz -owarp warp_template2mt.nii.gz -qc ~/qc_singleSubj



# Atlas-based analysis (Extracting metrics (MTR) in gray/white matter tracts) (dataset: data_atlas-based-analysis)
# ======================================================================================================================

# Warp template
sct_warp_template -d mt1.nii.gz -w warp_template2mt.nii.gz -a 1 -qc ~/qc_singleSubj
# Check results
fsleyes mt1.nii.gz -cm greyscale -a 100.0 label/template/PAM50_t2.nii.gz -cm greyscale -dr 0 4000 -a 100.0 label/template/PAM50_gm.nii.gz -cm red-yellow -dr 0.4 1 -a 100.0 label/template/PAM50_wm.nii.gz -cm blue-lightblue -dr 0.4 1 -a 100.0 &

# Extract MTR for each slice within the white matter (combined label: #51)
# Tips: To list all available labels, type: "sct_extract_metric"
sct_extract_metric -i mtr.nii.gz -f label/atlas -method map -l 51 -vert 2:5 -o mtr_in_wm.csv

# Extract MTR within the right and left corticospinal tract and aggregate across specific slices
sct_extract_metric -i mtr.nii.gz -f label/atlas -method map -l 4,5 -z 5:15 -o mtr_in_cst.csv
# You can specify the vertebral levels to extract MTR from. For example, to quantify MTR between C2 and C4 levels in the
# dorsal column (combined label: #53) using weighted average:
sct_extract_metric -i mtr.nii.gz -f label/atlas -method map -l 53 -vert 2:4 -vertfile label/template/PAM50_levels.nii.gz -o mtr_in_dc.csv



# Diffusion-weighted MRI (dataset: data_processing-dmri-data)
# ======================================================================================================================

cd ../dmri
# Preprocessing steps
# Compute mean dMRI from dMRI data
sct_dmri_separate_b0_and_dwi -i dmri.nii.gz -bvec bvecs.txt
# Segment SC on mean dMRI data
# Note: This segmentation does not need to be accurate-- it is only used to create a mask around the cord
sct_deepseg spinalcord -i dmri_dwi_mean.nii.gz -qc ~/qc_singleSubj
# Create mask (for subsequent cropping)
sct_create_mask -i dmri_dwi_mean.nii.gz -p centerline,dmri_dwi_mean_seg.nii.gz -size 35mm

# Motion correction (moco)
sct_dmri_moco -i dmri.nii.gz -m mask_dmri_dwi_mean.nii.gz -bvec bvecs.txt -qc ~/qc_singleSubj -qc-seg dmri_dwi_mean_seg.nii.gz
# Check results in the QC report

# Segment SC on motion-corrected mean dwi data (check results in the QC report)
sct_deepseg spinalcord -i dmri_moco_dwi_mean.nii.gz -qc ~/qc_singleSubj

# Register template->dwi via t2 to account for cord shape (which is better defined in T2 contrast)
# Tips: Here we use the PAM50 contrast t1, which is closer to the dwi contrast (although we are not using type=im in
#       -param, so it will not make a difference here)
# Note: the flag “-initwarpinv" provides a transformation dmri->template, in case you would like to bring all your DTI
#       metrics in the PAM50 space (e.g. group averaging of FA maps)
sct_register_multimodal -i "${SCT_DIR}/data/PAM50/template/PAM50_t1.nii.gz" -iseg "${SCT_DIR}/data/PAM50/template/PAM50_cord.nii.gz" -d dmri_moco_dwi_mean.nii.gz -dseg dmri_moco_dwi_mean_seg.nii.gz -initwarp ../t2/warp_template2anat.nii.gz -initwarpinv ../t2/warp_anat2template.nii.gz -owarp warp_template2dmri.nii.gz -owarpinv warp_dmri2template.nii.gz -param step=1,type=seg,algo=centermass:step=2,type=seg,algo=bsplinesyn,slicewise=1,iter=3 -qc ~/qc_singleSubj
# Warp template (so 'label/atlas' can be used to extract metrics)
sct_warp_template -d dmri_moco_dwi_mean.nii.gz -w warp_template2dmri.nii.gz -qc ~/qc_singleSubj
# Check results in the QC report

# Compute DTI metrics using dipy [1]
sct_dmri_compute_dti -i dmri_moco.nii.gz -bval bvals.txt -bvec bvecs.txt
# Tips: the flag "-method restore" estimates the tensor with robust fit (RESTORE method [2])

# Compute FA within the white matter from individual level 2 to 5
sct_extract_metric -i dti_FA.nii.gz -f label/atlas -l 51 -method map -vert 2:5 -vertfile label/template/PAM50_levels.nii.gz -perlevel 1 -o fa_in_wm.csv
# Compute FA within the CST, aggregated across z slices, using the weighted average method
sct_extract_metric -i dti_FA.nii.gz -f label/atlas -l 4,5 -method wa -z 2:14 -o fa_in_cst.csv



# Functional MRI (dataset: data_processing-fmri-data)
# ======================================================================================================================

# the T2 segmentation will be reused, but it can also be generated using the commands below:
cd ../t2
sct_deepseg spinalcord -i t2.nii.gz

cd ../fmri
# Preprocessing steps
# Average all fMRI time series to make it a 3D volume (needed by the next command)
sct_maths -i fmri.nii.gz -mean t -o fmri_mean.nii.gz
# Bring t2 segmentation to fmri space (to create a mask)
sct_register_multimodal -i ../t2/t2_seg.nii.gz -d fmri_mean.nii.gz -identity 1
# Create mask at the center of the FOV
sct_create_mask -i fmri.nii.gz -p centerline,t2_seg_reg.nii.gz -size 35mm -f cylinder

# Motion correction (using mask)
sct_fmri_moco -i fmri.nii.gz -m mask_fmri.nii.gz -qc ~/qc_singleSubj -qc-seg t2_seg_reg.nii.gz

# Cord segmentation on motion-corrected averaged time series
sct_deepseg spinalcord -i fmri_moco_mean.nii.gz -qc ~/qc_singleSubj
# TSNR before/after motion correction with QC report
sct_fmri_compute_tsnr -i fmri.nii.gz
sct_fmri_compute_tsnr -i fmri_moco.nii.gz
sct_qc -i fmri_tsnr.nii.gz -d fmri_moco_tsnr.nii.gz -s fmri_moco_mean_seg.nii.gz -p sct_fmri_compute_tsnr -qc ~/qc_singleSubj

# Register the template to the fMRI scan.
# Note: here we don't rely on the segmentation because it is difficult to obtain one automatically. Instead, we rely on
#       ANTs_SyN superpower to find a suitable transformation between the PAM50_t2s and the fMRI scan. We don't want to
#       put too many iterations because this registration is very sensitive to the artifacts (drop out) in the image.
#       Also, we want a 3D transformation (not 2D) because we need the through-z regularization.
sct_register_multimodal -i "${SCT_DIR}/data/PAM50/template/PAM50_t2s.nii.gz" -iseg "${SCT_DIR}/data/PAM50/template/PAM50_cord.nii.gz" -d fmri_moco_mean.nii.gz -dseg fmri_moco_mean_seg.nii.gz -param step=1,type=seg,algo=centermass:step=2,type=seg,algo=bsplinesyn,slicewise=1,iter=3:step=3,type=im,algo=syn,metric=CC,iter=3,slicewise=1 -initwarp ../t2/warp_template2anat.nii.gz -initwarpinv ../t2/warp_anat2template.nii.gz -owarp warp_template2fmri.nii.gz -owarpinv warp_fmri2template.nii.gz -qc ~/qc_singleSubj
# Check results in the QC report

# Warp template with the spinal levels (can be found at $SCT_DIR/data/PAM50/template/)
sct_warp_template -d fmri_moco_mean.nii.gz -w warp_template2fmri.nii.gz -a 0 -qc ~/qc_singleSubj


# FIXME: Find a way to insert these commands into the above tutorial
#        They were in our "New and upcoming features" section.
# Segment the spinal cord on gradient echo EPI data
# Crop extraneous tissue using the t2-based mask generated earlier
sct_crop_image -i fmri_moco_mean.nii.gz -m mask_fmri.nii.gz -b 0
# Segment the cord using the cropped image
sct_deepseg sc_epi -i fmri_moco_mean_crop.nii.gz -qc ~/qc_singleSubj


# Spinal cord smoothing (dataset: data_spinalcord-smoothing)
# ======================================================================================================================

cd ../t1
# Segment T1-weighted image (but, reuse the previous segmentation to save processing time)
## sct_deepseg spinalcord -i t1.nii.gz -qc ~/qc_singleSubj

# Smooth spinal cord along centerline (extracted from the segmentation)
sct_smooth_spinalcord -i t1.nii.gz -s t1_seg.nii.gz
# Tips: use flag "-sigma" to specify smoothing kernel size (in mm)

# Second-pass segmentation using the smoothed anatomical image
sct_deepseg spinalcord -i t1_smooth.nii.gz -qc ~/qc_singleSubj



# Visualizing misaligned cords (dataset: data_visualizing-misaligned-cords)
# ======================================================================================================================

# Align the spinal cord in the right-left direction using slice-wise translations.
sct_flatten_sagittal -i t1.nii.gz -s t1_seg.nii.gz
# Note: Use for visualization purposes only




# Lesion analysis for SCI (dataset: data_lesion-analysis)
# ======================================================================================================================
cd ../t2_lesion
# Segment the spinal cord and intramedullary lesion using the SCIsegV2 model
# Note: t2.nii.gz contains a fake lesion for the purpose of this tutorial
sct_deepseg lesion_sci_t2 -i t2.nii.gz -qc ~/qc_singleSubj
# Note: Two files are output:
# - t2_sc_seg.nii.gz: the spinal cord segmentation
# - t2_lesion_seg.nii.gz: the lesion segmentation
# Check results using FSLeyes
fsleyes t2.nii.gz -cm greyscale t2_sc_seg.nii.gz -cm red -a 70.0 t2_lesion_seg.nii.gz -cm blue-lightblue -a 70.0 &

# Compute various morphometric measures, such as number of lesions, lesion length, lesion volume, etc.
sct_analyze_lesion -m t2_lesion_seg.nii.gz -s t2_sc_seg.nii.gz -qc ~/qc_singleSubj
# Lesion analysis using PAM50 (the -f flag is used to specify the folder containing the atlas/template)
# Note: You must go through the "Register to Template" steps (labeling, registration) first
#       This is because `sct_warp_template` is required to generate the `label` folder used for `-f`
sct_warp_template -d t2.nii.gz -w ../t2/warp_template2anat.nii.gz
sct_analyze_lesion -m t2_lesion_seg.nii.gz -s t2_sc_seg.nii.gz -f label -qc ~/qc_singleSubj

# You can also use the legacy method if the new methods fail for your data (`-c t2s` is also supported)
sct_deepseg_lesion -i t2.nii.gz -c t2


# Rootlets-based registration (dataset: data_rootlets-registration)
# ======================================================================================================================

# Segment the spinal nerve rootlets
cd ../t2/
sct_deepseg rootlets -i t2.nii.gz -o t2_rootlets.nii.gz -qc ~/qc_singleSubj
# Check results using FSLeyes
fsleyes t2.nii.gz -cm greyscale t2_rootlets.nii.gz -cm subcortical -a 70.0 &
# Rootlets-based registration
sct_register_to_template -i t2.nii.gz -s t2_seg.nii.gz -lrootlet t2_rootlets.nii.gz -c t2 -ofolder rootlets-reg -qc ~/qc_singleSubj


# Multiple sclerosis lesion segmentation (dataset: data_ms-lesion-segmentation)
# ======================================================================================================================

# Multiple sclerosis lesion segmentation on T2-weighted images
cd ../t2_ms/
# Segment the spinal cord (to be used as input for lesion_ms QC)
sct_deepseg spinalcord -i t2.nii.gz -qc ~/qc_singleSubj
# Segment using lesion_ms (-single-fold is recommended to speed up inference time)
sct_deepseg lesion_ms -i t2.nii.gz -qc ~/qc_singleSubj -qc-seg t2_seg.nii.gz -single-fold
# Check results using FSLeyes
fsleyes t2.nii.gz -cm greyscale t2_lesion_seg.nii.gz -cm red -a 70.0 &

# Multiple sclerosis lesion segmentation on MP2RAGE images
# Commented out for now due to lack of appropriate public data
# sct_deepseg spinalcord -i "$IMAGE_UNIT1" -o "$IMAGE_SEG"
# sct_crop_image -i "$IMAGE_UNIT1" -m "$IMAGE_SEG" -dilate 30x30x5
# sct_deepseg lesion_ms_mp2rage -i "$IMAGE_UNIT1" -qc ~/qc_singleSubj

# Return to parent directory
cd ..


# Display results (to easily compare integrity across SCT versions)
# ===========================================================================================
set +v
end=$(date +%s)
runtime=$((end-start))
echo "~~~"  # these are used to format as code when copy/pasting in github's markdown
echo "Version:         $(sct_version)"
echo "Ran on:          $(uname -nsr)"
echo "Duration:        $((runtime / 3600))hrs $(((runtime / 60) % 60))min $((runtime % 60))sec"
echo "---"
# The file `test_batch_processing.py` will output tested values when run as a script
"$SCT_DIR"/python/envs/venv_sct/bin/python "$SCT_DIR"/testing/batch_single_subject/test_batch_single_subject.py ||
"$SCT_DIR"/python/envs/venv_sct/python.exe "$SCT_DIR"/testing/batch_single_subject/test_batch_single_subject.py
echo "~~~"

# Display syntax to open QC report on web browser
echo "To open Quality Control (QC) report on a web-browser, run the following:"
echo "$open_command $SCT_BP_QC_FOLDER/index.html"
