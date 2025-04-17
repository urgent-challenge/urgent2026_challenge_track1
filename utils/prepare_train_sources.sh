#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail



urgent25_path="./urgent2025_challenge"
output_dir=./data/train_sources

mkdir -p ${output_dir}

false && {
mkdir -p  data/tmp/train_sources

declare -A subsets
subsets["dns5"]="data/tmp/dns5_clean_read_speech_resampled_filtered_train"
subsets["libritts"]="data/tmp/libritts_resampled_train"
subsets["vctk"]="data/tmp/vctk_train"
subsets["ears"]="data/tmp/ears_train"
subsets["common_en"]="data/tmp/commonvoice_19.0_en_resampled_train_track1"
subsets["common_de"]="data/tmp/commonvoice_19.0_de_resampled_train_track1"
subsets["common_es"]="data/tmp/commonvoice_19.0_es_resampled_train_track1"
subsets["common_fr"]="data/tmp/commonvoice_19.0_fr_resampled_train_track1"
subsets["common_zh"]="data/tmp/commonvoice_19.0_zh-CN_resampled_train_track1"
subsets["mls_de"]="data/tmp/mls_german_resampled_train_track1"
subsets["mls_es"]="data/tmp/mls_spanish_resampled_train_track1"
subsets["mls_fr"]="data/tmp/mls_french_resampled_train_track1"

for key in ${!subsets[@]}; do
    cp ${urgent25_path}/${subsets[${key}]}.scp.abspath data/tmp/train_sources/`basename ${urgent25_path}/${subsets[${key}]}.scp.abspath .abspath`
    cp ${urgent25_path}/${subsets[${key}]}.utt2spk data/tmp/train_sources
    cp ${urgent25_path}/${subsets[${key}]}.text data/tmp/train_sources
done

cat data/tmp/train_sources/*.scp > data/tmp/train_sources/all_scp
cat data/tmp/train_sources/*.text > data/tmp/train_sources/all_text
cat data/tmp/train_sources/*.utt2spk > data/tmp/train_sources/all_utt2spk


./utils/filter_scp.pl data/train_clean_id  data/tmp/train_sources/all_scp > ${output_dir}/speech_sources.scp
./utils/filter_scp.pl data/train_clean_id  data/tmp/train_sources/all_text > ${output_dir}/text
./utils/filter_scp.pl data/train_clean_id  data/tmp/train_sources/all_utt2spk > ${output_dir}/utt2spk


cat ${urgent25_path}/data/tmp/dns5_noise_resampled_train.scp.abspath \
${urgent25_path}/data/tmp/wham_noise_train.scp.abspath \
${urgent25_path}/data/tmp/fma_noise_resampled_train.scp.abspath \
${urgent25_path}/data/tmp/fsd50k_noise_resampled_train.scp.abspath  >  ${output_dir}/noise_scoures.scp

cp ${urgent25_path}/data/tmp/wind_noise_train.scp.abspath  ${output_dir}/wind_noise_scoures.scp

cp ${urgent25_path}/data/tmp/dns5_rirs.scp.abspath ${output_dir}/rirs.scp

}

###

false && {


# generate simulation parameters
if [ ! -f "simulation_train/log/meta.tsv" ]; then
    python simulation/generate_data_param.py --config conf/simulation_train.yaml
fi




# simulate noisy speech for train
# It takes ~30 minutes to finish simulation with nj=8
OMP_NUM_THREADS=1 python simulation/simulate_data_from_param.py \
    --config conf/simulation_train.yaml \
    --meta_tsv simulation_train/log/meta.tsv \
    --nj 8 \
    --chunksize 100 \
    --highpass True 

mkdir -p "${output_dir}"
awk -F"\t" 'NR==1{for(i=1; i<=NF; i++) {if($i=="noisy_path") {n=i; break}} next} NR>1{print($1" "$n)}' simulation_train/log/meta.tsv | sort -u -k1,1 > "${output_dir}"/wav.scp
awk -F"\t" 'NR==1{for(i=1; i<=NF; i++) {if($i=="speech_sid") {n=i; break}} next} NR>1{print($1" "$n)}' simulation_train/log/meta.tsv | sort -u -k1,1 > "${output_dir}"/utt2spk
utils/utt2spk_to_spk2utt.pl "${output_dir}"/utt2spk > "${output_dir}"/spk2utt
awk -F"\t" 'NR==1{for(i=1; i<=NF; i++) {if($i=="text") {n=i; break}} next} NR>1{print($1" "$n)}' simulation_train/log/meta.tsv | sort -u -k1,1 > "${output_dir}"/text
awk -F"\t" 'NR==1{for(i=1; i<=NF; i++) {if($i=="clean_path") {n=i; break}} next} NR>1{print($1" "$n)}' simulation_train/log/meta.tsv | sort -u -k1,1 > "${output_dir}"/spk1.scp
awk -F"\t" 'NR==1{for(i=1; i<=NF; i++) {if($i=="fs") {n=i; break}} next} NR>1{print($1" "$n)}' simulation_train/log/meta.tsv | sort -u -k1,1 > "${output_dir}"/utt2fs
awk '{print($1" 1ch_"$2"Hz")}' "${output_dir}"/utt2fs > "${output_dir}"/utt2category

NUMBA_NUM_THREADS=1 python utils/get_utt2lang.py \
    --meta_tsv simulation_train/log/meta.tsv --outfile utt2lang
sort -u -k1,1 utt2lang > "${output_dir}"/utt2lang && rm utt2lang

}
