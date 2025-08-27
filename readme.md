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
By default, this script will generate a pre-simulated training set in `./data/train_simulation`, and a dynamic mixing training set in `./data/train_sources`.

The first is the pre-simulated data, which has the following form of directory structure:
```bash
data/train_simulation/
├── speech_length.scp # Speech duration in number of sample points.
├── spk1.scp # Clean speech file list of id and audio path.
├── utt2fs  # id to sampling rate mapping
├── utt2spk # utterance to speaker mapping 
└── wav.scp # Noisy speech file list of id and audio path.
```
The pre-simulated dataset can be loaded by the `PreSimulatedDataset` in the [baseline code](https://github.com/urgent-challenge/urgent2026_challenge_track1/blob/main/baseline_code/dataset.py) .

We also provided a `DynamicMixingDataset` class in the [baseline code](https://github.com/urgent-challenge/urgent2026_challenge_track1/blob/main/baseline_code/dataset.py) for loading data in a dynamic mixing manner.
The dataset has the following form of directory structure:

```bash
data/train_sources
├── noise_scoures.scp # Noise audio id and audio path.
├── rirs.scp # Room impulse response id and audio path.
├── source_length.scp # Speech duration in number of sample points.
├── speech_sources.scp # Clean speech id and audio path.
└── wind_noise_scoures.scp # # Wind noise audio id and audio path.
```

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

## Pretrained models

We have provided pretrained model checkpoints for both BSRNN and BSRNN-Flow. 
The training data `700h-TBF` is a subset of the ICASSP 2026 URGENT challenge, and the detailed description about it can be found in our recent [paper](https://arxiv.org/abs/2506.23859). 

| Model | Training Data |Download|
|:-----:|:------:|:------:|
|BSRNN | 700h-TBF |[Huggingface 🤗](https://huggingface.co/lichenda/icassp_2026_urgent_baseline/resolve/main/bsrnn.ckpt)|
|BSRNN-Flow | 700h-TBF |[Huggingface 🤗](https://huggingface.co/lichenda/icassp_2026_urgent_baseline/resolve/main/flow_bsrnn.ckpt)|


### Inference:

After downloading the checkpoints we provide or completing your own training, you can run the following script for speech enhancemnet.

```bash
python baseline_code/inference.py --input_scp [path_to_input_scp] --output [output_dir] --ckpt_path [path_to_checkpoint]
```