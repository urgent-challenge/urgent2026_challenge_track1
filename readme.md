## How to run:


### Setup:
```
# Set up the Python environment
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

> Note: The WSJ dataset was included in `urgent2025_challenge` but has been removed in `urgent2026_challenge_track1`. You can skip the [preparation of WSJ](https://github.com/urgent-challenge/urgent2025_challenge/blob/daf1730cc11bf450d05c2d9e1d8bb3afdd63c427/prepare_espnet_data.sh#L91-L107) if you do not have the WSJ license. 


The first is the pre-simulated data, which has the following form of directory structure:
```bash
data/train_simulation/
├── speech_length.scp # Speech duration in number of sample points.
├── spk1.scp # Clean speech file list of ID and audio path.
├── utt2fs  # ID to sampling rate mapping
├── utt2spk # utterance to speaker mapping 
└── wav.scp # Noisy speech file list of ID and audio path.
```
The pre-simulated dataset can be loaded by the `PreSimulatedDataset` in the [baseline code](https://github.com/urgent-challenge/urgent2026_challenge_track1/blob/main/baseline_code/dataset.py).

We also provided a `DynamicMixingDataset` class in the [baseline code](https://github.com/urgent-challenge/urgent2026_challenge_track1/blob/main/baseline_code/dataset.py) for loading data in a dynamic mixing manner.
The dataset has the following form of directory structure:

```bash
data/train_sources
├── noise_scoures.scp # Noise audio ID and audio path.
├── rirs.scp # Room impulse response ID and audio path.
├── source_length.scp # Speech duration in number of sample points.
├── speech_sources.scp # Clean speech ID and audio path.
└── wind_noise_scoures.scp # # Wind noise audio ID and audio path.
```

### Validation Set Preparation

Check the script `utils/prepare_validation_data.sh`  and set `urgent25_path` in it with the path to the `urgent2025_challenge` project in your system. 
```bash 
bash utils/prepare_validation_data.sh
```
By default, this script will generate a simulated validation set in `./data/validation`.

### Download the pre-simulated dataset

A pre-simulated training and validation dataset is available online at [HuggingFace 🤗](https://huggingface.co/datasets/lichenda/urgent26_track1_universal_se). Participants can download and use it directly without running the simulation. However, the simulated speech derived from the ESD subset is excluded due to licensing restrictions. You may apply for the license and run the simulation script yourself to obtain it.


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
If there is an error message prompting `Failed to initialize FFmpeg extension.` Please make sure FFmpeg has been installed on your machine, and try `conda install ffmpeg`.

## Pretrained models

We have provided pretrained model checkpoints for both BSRNN and BSRNN-Flow. 
The training data `700h-TBF` is a subset of the ICASSP 2026 URGENT challenge, and the detailed description about it can be found in our recent [paper](https://arxiv.org/abs/2506.23859). 

| Model | Training Data |Download|
|:-----:|:------:|:------:|
|BSRNN | 700h-TBF |[HuggingFace 🤗](https://huggingface.co/lichenda/icassp_2026_urgent_baseline/resolve/main/bsrnn.ckpt)|
|BSRNN-Flow | 700h-TBF |[HuggingFace 🤗](https://huggingface.co/lichenda/icassp_2026_urgent_baseline/resolve/main/flow_bsrnn.ckpt)|



If you have used the above models, we would appreciate your citation of the following paper:


```
@article{liLessMoreData2025,
	title = {Less is {More}: {Data} {Curation} {Matters} in {Scaling} {Speech} {Enhancement}},
	shorttitle = {Less is {More}},
	url = {http://arxiv.org/abs/2506.23859},
	doi = {10.48550/arXiv.2506.23859},
	urldate = {2025-09-10},
	publisher = {arXiv},
	author = {Li, Chenda and Zhang, Wangyou and Wang, Wei and Scheibler, Robin and Saijo, Kohei and Cornell, Samuele and Fu, Yihui and Sach, Marvin and Ni, Zhaoheng and Kumar, Anurag and Fingscheidt, Tim and Watanabe, Shinji and Qian, Yanmin},
	month = aug,
	year = {2025},
	note = {arXiv:2506.23859 [eess]},
	keywords = {Computer Science - Sound, Electrical Engineering and Systems Science - Audio and Speech Processing},
	annote = {Comment: Accepted by ASRU2025},
}
```


### Inference:

After downloading the checkpoints we provide or completing your own training, you can run the following script for speech enhancement.

```bash
python baseline_code/inference.py --input_scp [path_to_input_scp] --output [output_dir] --ckpt_path [path_to_checkpoint]
```

### Evaluation:

First, install the necessary dependencies for evaluation (Make sure you have finished the [setup](#setup) in previous steps):

```bash
pip install pip==24.0 # Some packages need this version.
pip install -r evaluation_metrics/requirements.txt
```

Check the variables in  `evaluation_metrics/eval_all.sh` and run it:

```bash
bash evaluation_metrics/eval_all.sh
```
