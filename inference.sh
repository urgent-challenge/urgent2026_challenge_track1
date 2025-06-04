

declare -A models

# models["finetune_full_100h"]="best_epoch=00-step=010000-val_sisnr=11.706.ckpt"
# models["finetune_full_100h_1e-6"]="best_epoch=00-step=004000-val_sisnr=11.498.ckpt"
# models["finetune_full_100h_1e-5"]="best_epoch=00-step=004000-val_sisnr=11.533.ckpt"


# models["finetune_full_700h_5e-4"]="last.ckpt"
# models["finetune_full_700h_1e-4"]="last.ckpt"
# models["finetune_full_700h_1e-5"]="last.ckpt"
# models["finetune_full_700h_1e-6"]="last.ckpt"

# models["run_baseline_bsrnn_dm_picked_700h_noise_50h"]="best_epoch=05-step=130000-val_sisnr=10.345.ckpt"
# models["run_baseline_bsrnn_dm_picked_350h_noise_50h"]="best_epoch=20-step=488000-val_sisnr=10.400.ckpt"
# models["run_baseline_bsrnn_dm_picked_100h_noise_50h"]="best_epoch=102-step=590000-val_loss=280825.469.ckpt"

# models["run_baseline_bsrnn_dm_picked_350h2"]="best_epoch=51-step=600000-val_sisnr=10.724.ckpt"
# models["run_baseline_bsrnn"]="best_epoch=70-step=1675000-val_sisnr=11.151.ckpt"
# models["run_baseline_bsrnn_dm"]="best_epoch=55-step=1315000-val_sisnr=11.275.ckpt"
# models["run_baseline_bsrnn_dm_picked_100h"]="best_epoch=41-step=240000-val_sisnr=9.507.ckpt"
# models["run_baseline_bsrnn_dm_random_100h"]="best_epoch=38-step=250000-val_sisnr=9.224.ckpt"
# models["run_baseline_bsrnn_dm_random_350h"]="best_epoch=25-step=290000-val_sisnr=10.136.ckpt"
# models["run_baseline_bsrnn_dm_random_700h"]="best_epoch=14-step=335000-val_sisnr=10.574.ckpt"
# models["run_baseline_bsrnn_full"]="best_epoch=07-step=340000-val_sisnr=11.396.ckpt"


for model in ${!models[@]};do
    output_dir=enhanced/${model}
    python baseline_code/inference.py --input_scp urgent_submissions_track1/wavs/test.scp --output ${output_dir} --ckpt_path exp/${model}/baseline_bsrnn/version_0/checkpoints/${models[$model]}
    ./utils/filter_scp.pl data/urgent25_blindtest_clean/spk1.scp ${output_dir}/inf.scp | sort > ${output_dir}/sim_inf.scp
done



for model in ${!models[@]};do
    output_dir=enhanced/${model}
    mkdir -p enhanced/${model}/score
    echo  enhanced/${model}/score
    python evaluation_metrics/calculate_intrusive_se_metrics.py --ref_scp data/urgent25_blindtest_clean/spk1.scp  --inf_scp ${output_dir}/sim_inf.scp --output_dir ${output_dir}/score/se
    python evaluation_metrics/calculate_nonintrusive_dnsmos.py  --inf_scp ${output_dir}/inf.scp --output_dir ${output_dir}/score/dnsmos --device cuda
    python evaluation_metrics/calculate_nonintrusive_mos.py  --inf_scp ${output_dir}/inf.scp --output_dir ${output_dir}/score/utmos --device cuda
    python evaluation_metrics/calculate_nonintrusive_nisqa.py --inf_scp ${output_dir}/inf.scp --output_dir ${output_dir}/score/nisqa --device cuda
    python evaluation_metrics/calculate_nonintrusive_sigmos.py --inf_scp ${output_dir}/inf.scp --output_dir ${output_dir}/score/sigmos --device cuda
    python evaluation_metrics/calculate_nonintrusive_dnsmos_pro.py --inf_scp ${output_dir}/inf.scp --output_dir ${output_dir}/score/dnsmos_pro --device cuda
    python evaluation_metrics/calculate_nonintrusive_distillmos.py --inf_scp ${output_dir}/inf.scp --output_dir ${output_dir}/score/distillmos --device cuda
    python evaluation_metrics/calculate_nonintrusive_vqscore.py --inf_scp ${output_dir}/inf.scp --output_dir ${output_dir}/score/vqscore --device cuda

done
#  python baseline_code/inference.py --input_scp urgent_submissions_track1/wavs/test.scp --output tmp/se/ --ckpt_path ./best_epoch=55-step=1315000-val_sisnr=11.275.ckpt