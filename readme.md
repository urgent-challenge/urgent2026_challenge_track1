## How to run:


### Setup:
```
pip install espnet
pip install -e ./
```

### Training Set Preparation

Check the script `utils/prepare_train_data.sh`  and set `urgent25_path` in it with the path to the `urgent2025_challenge` project in your system. 
This script assumes you have already run `prepare_espnet_data.sh` in the `urgent2025_challenge` project.

```bash 
bash utils/prepare_train_data.sh
```
By default, this script will generate a simulated training set in `./data/train_simulation`, and a dynamic mixing training set in `./data/train_sources`.

### Validation Set Preparation

Check the script `utils/prepare_validation_data.sh`  and set `urgent25_path` in it with the path to the `urgent2025_challenge` project in your system. 
```bash 
bash utils/prepare_validation_data.sh
```
By default, this script will generate a simulated validation set in `./data/validation`.


### Training
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

If dynamic mixing is not used, a ESPnet-like training set should be used:
```
data/train
├── speech_length.scp
├── spk1.scp
├── spk2utt (not used)
├── text (not used)
├── utt2category (not used)
├── utt2fs
├── utt2lang (not used)
├── utt2spk (not used)
└── wav.scp
```