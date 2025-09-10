#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

inf_scp=./enhanced/baseline/inf.scp
ref_scp=./data/validation_leaderboard/spk1.scp
output_dir=./enhanced/baseline
nj=8

mkdir -p ${output_dir}

# python evaluation_metrics/calculate_intrusive_se_metrics.py --ref_scp ${ref_scp}  --inf_scp ${inf_scp} --output_dir ${output_dir}/score/se --nj ${nj}
python evaluation_metrics/calculate_nonintrusive_dnsmos.py  --inf_scp ${inf_scp} --output_dir ${output_dir}/score/dnsmos --device cuda
python evaluation_metrics/calculate_nonintrusive_nisqa.py  --inf_scp ${inf_scp} --output_dir ${output_dir}/score/nisqa --device cuda