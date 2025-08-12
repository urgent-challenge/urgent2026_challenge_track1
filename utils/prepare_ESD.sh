#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

###############################################################################
# ESD Dataset Disclaimer and License Verification
#
# WARNING: The Emotional Speech Database (ESD) is protected by a specific
# license agreement from the National University of Singapore (NUS). 
#
# You MUST obtain a signed license agreement before legally using this dataset
# Unauthorized use may result in legal consequences
#
# Official License: 
#   https://drive.google.com/file/d/1Q1Wa45u-ymzpUO3_U-y8yisn4f5PlDIK/view?usp=sharing
# 
# By proceeding, you acknowledge that:
#   - You have read and understood the NUS License Agreement
#   - You accept full responsibility for compliance with the license terms
#   - The script maintainers bear no liability for license violations
#
###############################################################################

# Set this to TRUE only after obtaining signed license from NUS
HAS_VALID_LICENSE=True  # CHANGE TO "True" IF LICENSED

# --- LICENSE VERIFICATION ---
if [ "$HAS_VALID_LICENSE" != "True" ]; then
    echo ""
    echo "ERROR: LICENSE REQUIRED"
    echo "================================================================"
    echo "You MUST obtain a valid license before using the ESD database:"
    echo "1. Review the license terms:"
    echo "   https://hltsingapore.github.io/ESD/index.html"
    echo "2. Request the license by emailing to data publisher"
    echo "3. Set HAS_VALID_LICENSE=True in this script after receiving signed agreement"
    echo "================================================================"
    exit 1
fi


output_dir="downloads/ESD/"
mkdir -p "${output_dir}"

echo "=== Preparing ESD data ==="
#################################
# Download data
#################################
echo "Download ESD data from Google Drive"
if [ ! -e "${output_dir}/download_ESD.done" ]; then
    gdown -O "${output_dir}"/ESD.zip 1scuFwqh8s7KIYAfZW1Eu6088ZAK2SI-v
    unzip ${output_dir}/ESD.zip -d ${output_dir}
    mv "${output_dir}/Emotion Speech Dataset" ${output_dir}/Emotion_Speech_Dataset
else
    echo "Skip downloading ESD as it has already finished"
fi
touch "${output_dir}"/download_ESD.done


#################################
# Data preprocessing
#################################
mkdir -p tmp

BW_EST_FILE=tmp/ESD.json
BW_EST_FILE_JSON_GZ="datafiles/ESD/ESD.json.gz"
if [ -f ${BW_EST_FILE_JSON_GZ} ]; then
    gunzip -c $BW_EST_FILE_JSON_GZ > $BW_EST_FILE
fi
if [ ! -f ${BW_EST_FILE} ]; then
    echo "[ESD] estimating audio bandwidth"
    OMP_NUM_THREADS=1 python utils/estimate_audio_bandwidth.py \
        --audio_dir "${output_dir}/Emotion_Speech_Dataset/" \
        --audio_format wav \
        --chunksize 1000 \
        --nj 8 \
        --outfile "${BW_EST_FILE}"
else
    echo "Estimated bandwidth file already exists. Delete ${BW_EST_FILE} if you want to re-estimate."
fi

RESAMP_SCP_FILE="${output_dir}/ESD_resampled.scp"
if [ ! -f ${RESAMP_SCP_FILE} ]; then
    echo "[ESD] resampling to estimated audio bandwidth"
    OMP_NUM_THREADS=1 python utils/resample_to_estimated_bandwidth.py \
        --bandwidth_data "${BW_EST_FILE}" \
        --out_scpfile "${RESAMP_SCP_FILE}" \
        --outdir "${output_dir}/resampled/" \
        --nj 8 \
        --chunksize 1000
else
    echo "Resampled scp file already exists. Delete ${RESAMP_SCP_FILE} if you want to re-resample."
fi


