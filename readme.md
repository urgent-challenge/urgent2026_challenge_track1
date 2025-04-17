## How to run:


Setup:
```
pip install espnet
pip install -e ./
```


Train with dynamic mixing:
```bash 
python baseline_code/train_se.py \
--train_tag run_baseline_bsrnn \
--batch_size 2 \
--train_set_path data/train_sources \
--valid_set_path data/validation \
--val_check_interval 10000
```


Train without dynamic mixing:

```bash 
python baseline_code/train_se.py \
--train_tag run_baseline_bsrnn \
--batch_size 2 \
--train_set_path data/train \
--valid_set_path data/validation \
--train_set_dynamic_mixing False \
--val_check_interval 10000
```

If DM is not used, a espnet-like training set should be used:
```
data/train
├── speech_length.scp
├── spk1.scp
├── spk2utt
├── text
├── utt2fs
├── utt2spk
└── wav.scp
```