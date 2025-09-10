#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# run `pip install -r evaluation_metrics/requirements.txt` first

inf_scp=./enhanced/baseline/inf.scp # replace it with your infernece scp file
ref_scp=./data/validation_leaderboard/spk1.scp # replace it with your reference scp file
output_dir=./enhanced/baseline # replace it with your output path
utt2lang=./data/validation_leaderboard/utt2lang # replace it with your utt2lang file
text=./data/validation_leaderboard/text # replace it with your text label
nj=8
device=cuda # or cpu

mkdir -p ${output_dir}

#Intrusive SE metrics	
python evaluation_metrics/calculate_intrusive_se_metrics.py --ref_scp ${ref_scp}  --inf_scp ${inf_scp} --output_dir ${output_dir}/score/se --nj ${nj}

#Non-intrusive SE metrics	
python evaluation_metrics/calculate_nonintrusive_dnsmos.py  --inf_scp ${inf_scp} --output_dir ${output_dir}/score/dnsmos --device ${device}
python evaluation_metrics/calculate_nonintrusive_nisqa.py  --inf_scp ${inf_scp} --output_dir ${output_dir}/score/nisqa --device ${device}
python evaluation_metrics/calculate_nonintrusive_utmos.py  --inf_scp ${inf_scp} --output_dir ${output_dir}/score/utmos --device ${device}
python evaluation_metrics/calculate_nonintrusive_scoreq.py  --inf_scp ${inf_scp} --output_dir ${output_dir}/score/scoreq 

#Downstream-task-independent metrics	
python evaluation_metrics/calculate_speechbert_score.py   --ref_scp ${ref_scp}  --inf_scp ${inf_scp} --output_dir ${output_dir}/score/speechbert_score --device ${device}
python evaluation_metrics/calculate_phoneme_similarity.py --ref_scp ${ref_scp}  --inf_scp ${inf_scp} --output_dir ${output_dir}/score/lps --device ${device}

#Downstream-task-dependent metrics	
python evaluation_metrics/calculate_speaker_similarity.py --ref_scp ${ref_scp}  --inf_scp ${inf_scp} --output_dir ${output_dir}/score/spk_sim --device ${device}
python evaluation_metrics/calculate_emotion_similarity.py --ref_scp ${ref_scp}  --inf_scp ${inf_scp} --output_dir ${output_dir}/score/emo_sim --device ${device}
python evaluation_metrics/calculate_lid_accuracy.py --meta_tsv ${utt2lang}  --inf_scp ${inf_scp} --output_dir ${output_dir}/score/lid_acc --device ${device}
python evaluation_metrics/calculate_wer.py --meta_tsv ${text} --utt2lang ${utt2lang} --inf_scp ${inf_scp} --output_dir ${output_dir}/score/cer --device ${device}
