#!/bin/bash

# Spinal cord analysis pipeline for the SCT course and webpage tutorials:
# https://spinalcordtoolbox.com/user_section/tutorials/analysis-pipelines-with-sct.html

# First, download the manual correction script into a local folder.
sct_download_data -d manual-correction -o manual-correction

# Apply process_data.sh across each subject directory using config.yml.
# (The tutorial recommends config-file usage for reproducibility.)
sct_run_batch -script process_data.sh -config config.yml

# Re-run analysis with corrections, writing outputs to output_correction/.
sct_run_batch -script process_data.sh -path-data data/ -path-output output_correction -jobs 3
