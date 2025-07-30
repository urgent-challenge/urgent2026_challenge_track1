#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

output_dir="downloads/VocalSet/"
mkdir -p "${output_dir}"

echo "=== Preparing VocalSet data ==="
#################################
# Download data
#################################
echo "Download VocalSet data from zenodo"
if [ ! -e "${output_dir}/download_VocalSet.done" ]; then
    curl -L -o ${output_dir}/VocalSet.zip \
    https://zenodo.org/records/1193957/files/VocalSet.zip
    unzip ${output_dir}/VocalSet.zip -d ${output_dir}
else
    echo "Skip downloading VocalSet as it has already finished"
fi
touch "${output_dir}"/download_VocalSet.done

#################################
# Data preprocessing
#################################
mkdir -p tmp

BW_EST_FILE=tmp/VocalSet.json
BW_EST_FILE_JSON_GZ="datafiles/VocalSet/VocalSet.json.gz"
if [ -f ${BW_EST_FILE_JSON_GZ} ]; then
    gunzip -c $BW_EST_FILE_JSON_GZ > $BW_EST_FILE
fi
if [ ! -f ${BW_EST_FILE} ]; then
    echo "[VocalSet] estimating audio bandwidth"
    OMP_NUM_THREADS=1 python utils/estimate_audio_bandwidth.py \
        --audio_dir "${output_dir}/FULL/"  \
        --audio_format wav \
        --chunksize 1000 \
        --nj 8 \
        --outfile "${BW_EST_FILE}"
else
    echo "Estimated bandwidth file already exists. Delete ${BW_EST_FILE} if you want to re-estimate."
fi

RESAMP_SCP_FILE="${output_dir}/VocalSet.scp"
if [ ! -f ${RESAMP_SCP_FILE} ]; then
    echo "[VocalSet] resampling to estimated audio bandwidth"
    OMP_NUM_THREADS=1 python utils/resample_to_estimated_bandwidth.py \
        --bandwidth_data "${BW_EST_FILE}" \
        --out_scpfile "${RESAMP_SCP_FILE}" \
        --outdir "${output_dir}/resampled/" \
        --nj 8 \
        --chunksize 1000
else
    echo "Resampled scp file already exists. Delete ${RESAMP_SCP_FILE} if you want to re-resample."
fi


