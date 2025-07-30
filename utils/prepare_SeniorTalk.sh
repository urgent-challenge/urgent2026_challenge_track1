#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

output_dir="downloads/SeniorTalk/"
mkdir -p "${output_dir}"

download_url="https://www.modelscope.cn/datasets/BAAI/SeniorTalk.git"
# You can also replace with the huggingface repo: https://huggingface.co/datasets/BAAI/SeniorTalk

# Check if Git LFS is installed
if git lfs version &> /dev/null; then
    echo "download_url set to ${download_url}"
else
    echo "❌ Git LFS not found"
    echo "======================= INSTALLATION REQUIRED ======================="
    echo "Git Large File Storage (LFS) is required for handling large files."
    echo "Please install using one of these methods:"
    echo ""
    echo "1. macOS (using Homebrew):"
    echo "   brew install git-lfs"
    echo ""
    echo "2. Linux (Debian/Ubuntu):"
    echo "   sudo apt-get install git-lfs"
    echo ""
    echo "3. Linux (RHEL/CentOS):"
    echo "   sudo yum install git-lfs"
    echo ""
    echo "4. Windows:"
    echo "   Download installer from: https://git-lfs.com/"
    echo ""
    echo "5. Other platforms:"
    echo "   See official documentation: https://git-lfs.com/"
    echo "===================================================================="
    # Additional setup instructions
    echo "After installation, run: git lfs install"
    exit 1
fi

echo "=== Preparing SeniorTalk data ==="
#################################
# Download data
#################################
echo "Download SeniorTalk data from ${download_url}"
if [ ! -e "${output_dir}/download_SeniorTalk.done" ]; then
    git clone ${download_url} ${output_dir}
else
    echo "Skip downloading SeniorTalk as it has already finished"
fi
touch "${output_dir}"/download_SeniorTalk.done


echo "Extracting SeniorTalk data"
if [ ! -e "${output_dir}/extract_SeniorTalk.done" ]; then
    cwd=`pwd`
    cd ${output_dir}/sentence_data/wav/
    tar xvf dev/dev.tar; tar xvf test/test.tar
    cd train; for f in *.tar; do tar xvf ${f}; done
    cd ${cwd}
else
    echo "Skip extracting SeniorTalk as it has already finished"
fi
touch "${output_dir}/extract_SeniorTalk.done"

#################################
# Data preprocessing
#################################
mkdir -p tmp

BW_EST_FILE=tmp/SeniorTalk.json
BW_EST_FILE_JSON_GZ="datafiles/SeniorTalk/SeniorTalk.json.gz"
if [ -f ${BW_EST_FILE_JSON_GZ} ]; then
    gunzip -c $BW_EST_FILE_JSON_GZ > $BW_EST_FILE
fi
if [ ! -f ${BW_EST_FILE} ]; then
    echo "[SeniorTalk] estimating audio bandwidth"
    OMP_NUM_THREADS=1 python utils/estimate_audio_bandwidth.py \
        --audio_dir "${output_dir}/sentence_data/wav/train" \
        --audio_format wav \
        --chunksize 1000 \
        --nj 8 \
        --outfile "${BW_EST_FILE}"
else
    echo "Estimated bandwidth file already exists. Delete ${BW_EST_FILE} if you want to re-estimate."
fi

RESAMP_SCP_FILE="${output_dir}/SeniorTalk_resampled.scp"
if [ ! -f ${RESAMP_SCP_FILE} ]; then
    echo "[SeniorTalk] resampling to estimated audio bandwidth"
    OMP_NUM_THREADS=1 python utils/resample_to_estimated_bandwidth.py \
        --bandwidth_data "${BW_EST_FILE}" \
        --out_scpfile "${RESAMP_SCP_FILE}" \
        --outdir "${output_dir}/resampled/" \
        --nj 8 \
        --chunksize 1000
else
    echo "Resampled scp file already exists. Delete ${RESAMP_SCP_FILE} if you want to re-resample."
fi


