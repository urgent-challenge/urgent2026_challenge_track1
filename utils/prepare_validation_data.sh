#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


## replace urgent25_path with your urgent2025_challenge path
## This script assume you have alrealy run the 'prepare_espnet_data.sh'
## in urgent2025_challenge project
urgent25_path="/mnt/rexp/urgent2025_challenge"
output_dir=./data/validation

mkdir -p ${output_dir}
mkdir -p data/tmp/validation

declare -A subsets

subsets["dns5"]="data/tmp/dns5_clean_read_speech_resampled_filtered_validation.scp"
subsets["libritts"]="data/tmp/libritts_resampled_validation.scp"
subsets["vctk"]="data/tmp/vctk_validation.scp"
subsets["ears"]="data/tmp/ears_validation.scp"
subsets["common_de"]="data/tmp/commonvoice_19.0_de_resampled_validation.scp"
subsets["common_es"]="data/tmp/commonvoice_19.0_es_resampled_validation.scp"
subsets["common_fr"]="data/tmp/commonvoice_19.0_fr_resampled_validation.scp"
subsets["common_zh"]="data/tmp/commonvoice_19.0_zh-CN_resampled_validation.scp"
subsets["mls_de"]="data/tmp/mls_german_resampled_validation.scp"
subsets["mls_es"]="data/tmp/mls_spanish_resampled_validation.scp"
subsets["mls_fr"]="data/tmp/mls_french_resampled_validation.scp"


cat ${urgent25_path}/data/tmp/*validation.text  >  data/tmp/validation/all_text
cat ${urgent25_path}/data/tmp/*validation.utt2spk >  data/tmp/validation/all_utt2spk
cat ${urgent25_path}/data/tmp/*validation.scp | awk -v pwd="${urgent25_path}" '{ if ($3 !~ /^\//) { sub(/^\.\//, "", $3); $3 = pwd "/" $3 } print }' >  data/tmp/validation/all_scp

# concatenate files and filter them    
./utils/filter_scp.pl meta/validation_selected  data/tmp/validation/all_text > data/tmp/validation/speech_validation_subset.text
./utils/filter_scp.pl meta/validation_selected  data/tmp/validation/all_utt2spk > data/tmp/validation/speech_validation_subset.utt2spk
./utils/filter_scp.pl meta/validation_selected  data/tmp/validation/all_scp > data/tmp/validation/speech_validation_subset.scp


cat ${urgent25_path}/data/tmp/dns5_noise_resampled_validation.scp \
${urgent25_path}/data/tmp/wham_noise_validation.scp \
${urgent25_path}/data/tmp/fma_noise_resampled_validation.scp \
${urgent25_path}/data/tmp/fsd50k_noise_resampled_validation.scp |  awk -v pwd="${urgent25_path}" '{ if ($3 !~ /^\//) { sub(/^\.\//, "", $3); $3 = pwd "/" $3 } print }' >  data/tmp/validation/noise_scoures.scp


awk -v pwd="${urgent25_path}" '{ if ($3 !~ /^\//) { sub(/^\.\//, "", $3); $3 = pwd "/" $3 } print }' ${urgent25_path}/data/tmp/wind_noise_validation.scp >  data/tmp/validation/wind_noise_scoures.scp

awk -v pwd="${urgent25_path}" '{ if ($3 !~ /^\//) { sub(/^\.\//, "", $3); $3 = pwd "/" $3 } print }' ${urgent25_path}/data/tmp/dns5_rirs.scp > data/tmp/validation/rirs.scp


# generate simulation parameters
if [ ! -f "simulation_validation/log/meta.tsv" ]; then
    python simulation/generate_data_param.py --config conf/simulation_validation.yaml
fi


# simulate noisy speech for validation
# It takes ~30 minutes to finish simulation with nj=8
OMP_NUM_THREADS=1 python simulation/simulate_data_from_param.py \
    --config conf/simulation_validation.yaml \
    --meta_tsv simulation_validation/log/meta.tsv \
    --nj 8 \
    --chunksize 100 \
    --highpass True 

mkdir -p "${output_dir}"
awk -F"\t" 'NR==1{for(i=1; i<=NF; i++) {if($i=="noisy_path") {n=i; break}} next} NR>1{print($1" "$n)}' simulation_validation/log/meta.tsv | sort -u -k1,1 > "${output_dir}"/wav.scp
awk -F"\t" 'NR==1{for(i=1; i<=NF; i++) {if($i=="speech_sid") {n=i; break}} next} NR>1{print($1" "$n)}' simulation_validation/log/meta.tsv | sort -u -k1,1 > "${output_dir}"/utt2spk
utils/utt2spk_to_spk2utt.pl "${output_dir}"/utt2spk > "${output_dir}"/spk2utt
awk -F"\t" 'NR==1{for(i=1; i<=NF; i++) {if($i=="text") {n=i; break}} next} NR>1{print($1" "$n)}' simulation_validation/log/meta.tsv | sort -u -k1,1 > "${output_dir}"/text
awk -F"\t" 'NR==1{for(i=1; i<=NF; i++) {if($i=="clean_path") {n=i; break}} next} NR>1{print($1" "$n)}' simulation_validation/log/meta.tsv | sort -u -k1,1 > "${output_dir}"/spk1.scp
awk -F"\t" 'NR==1{for(i=1; i<=NF; i++) {if($i=="fs") {n=i; break}} next} NR>1{print($1" "$n)}' simulation_validation/log/meta.tsv | sort -u -k1,1 > "${output_dir}"/utt2fs
awk '{print($1" 1ch_"$2"Hz")}' "${output_dir}"/utt2fs > "${output_dir}"/utt2category

python utils/get_utt2lang.py \
    --meta_tsv simulation_validation/log/meta.tsv --outfile utt2lang
sort -u -k1,1 utt2lang > "${output_dir}"/utt2lang && rm utt2lang

python utils/utt2numsamples.py --input_scp ${output_dir}/wav.scp --outfile ${output_dir}/speech_length.scp