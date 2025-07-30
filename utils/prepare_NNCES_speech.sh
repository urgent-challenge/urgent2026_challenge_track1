#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

output_dir="downloads/NNCES/"
mkdir -p "${output_dir}"

echo "=== Preparing NNCES data ==="
#################################
# Download data
#################################
echo "Download NNCES data from https://www.kaggle.com"
if [ ! -e "${output_dir}/download_NNCES.done" ]; then
    curl -L -o ${output_dir}/nonnative-children-english-speech-nnces-corpus.zip \
    https://www.kaggle.com/api/v1/datasets/download/kodaliradha20phd7093/nonnative-children-english-speech-nnces-corpus
    unzip ${output_dir}/nonnative-children-english-speech-nnces-corpus.zip -d ${output_dir}
else
    echo "Skip downloading NNCES as it has already finished"
fi
touch "${output_dir}"/download_NNCES.done

#################################
# Data preprocessing
#################################
mkdir -p tmp

BW_EST_FILE=tmp/NNCES.json
BW_EST_FILE_JSON_GZ="datafiles/NNCES/NNCES.json.gz"
if [ -f ${BW_EST_FILE_JSON_GZ} ]; then
    gunzip -c $BW_EST_FILE_JSON_GZ > $BW_EST_FILE
fi
if [ ! -f ${BW_EST_FILE} ]; then
    echo "[NNCES] estimating audio bandwidth"
    OMP_NUM_THREADS=1 python utils/estimate_audio_bandwidth.py \
        --audio_dir "${output_dir}/Read_Speech_Data" "${output_dir}/Spontaneous_Speech_Data" \
        --audio_format wav \
        --chunksize 1000 \
        --nj 8 \
        --outfile "${BW_EST_FILE}"
else
    echo "Estimated bandwidth file already exists. Delete ${BW_EST_FILE} if you want to re-estimate."
fi

RESAMP_SCP_FILE="${output_dir}/NNCES_resampled.scp"
if [ ! -f ${RESAMP_SCP_FILE} ]; then
    echo "[NNCES] resampling to estimated audio bandwidth"
    OMP_NUM_THREADS=1 python utils/resample_to_estimated_bandwidth.py \
        --bandwidth_data "${BW_EST_FILE}" \
        --out_scpfile "${RESAMP_SCP_FILE}" \
        --outdir "${output_dir}/resampled/" \
        --nj 8 \
        --chunksize 1000
else
    echo "Resampled scp file already exists. Delete ${RESAMP_SCP_FILE} if you want to re-resample."
fi


