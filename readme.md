## How to run:


### Setup:
```
# Set up the python environment
conda create -n urgent2026_baseline_track1 python=3.10 
conda activate urgent2026_baseline_track1

# Install the baseline code
git clone --recursive git@github.com:urgent-challenge/urgent2026_challenge_track1.git
cd urgent2026_challenge_track1
pip install -e ./
```

### Training Set Preparation

Check the script `utils/prepare_train_data.sh`  and set `urgent25_path` in it with the path to the `urgent2025_challenge` project in your system. 
This script assumes you have already run `prepare_espnet_data.sh` in the [urgent2025_challenge](https://github.com/urgent-challenge/urgent2025_challenge/) project.

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
Train discriminative baseline SE models:
```bash 
python baseline_code/train_se.py  --config_file conf/models/BSRNN_baseline.yaml
```

Train generative FLOW SE models:

```bash 
python baseline_code/train_se.py  --config_file conf/models/BSRNN_flowse.yaml
```


Train with dynamic mixing:

Set `train_set_dynamic_mixing: True` and  `train_set_path: ./data/train_sources` in config files:

```bash 
python baseline_code/train_se.py  --config_file conf/models/BSRNN_baseline_dm.yaml
```
If there is an error message prompting `Failed to intialize FFmpeg extension.` Please make sure FFmpeg has been installed in your machine, and try `conda install ffmpeg`.

### Inference:

```bash
python baseline_code/inference.py --input_scp [path_to_input_scp] --output [output_dir] --ckpt_path [path_to_checkpoint]
```