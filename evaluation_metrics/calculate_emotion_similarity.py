from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from tqdm import tqdm

try:
    from emo2vec_versa.emo2vec_class import EMO2VEC
except ImportError:
    print(
        "'emo2vec' is not installed. Please install the package manually:\n"
        "   git clone https://github.com/ftshijt/emotion2vec\n"
        "   cd emotion2vec\n"
        "   pip install -e ./"
    )
    EMO2VEC = None


METRICS = ("EmotionSimilarity",)
TARGET_FS = 16000


################################################################
# Definition of metrics
################################################################
def emo2vec_setup(model_path=None, use_gpu=False):
    """Set up EMO2VEC model for emotion embedding extraction.

    Args:
        model_path (str, optional): Path to model checkpoint.
        use_gpu (bool, optional): Whether to use GPU.

    Returns:
        model (EMO2VEC): The loaded model.

    Raises:
        ImportError: If emo2vec_versa is not installed.
        ValueError: If model_tag is unknown.
        FileNotFoundError: If model file is not found.
    """
    if model_path is not None:
        print(f"Using provided model path: '{model_path}'.")
    elif Path("emotion2vec_base.pt").exists():
        model_path = "emotion2vec_base.pt"
        print(f"Using local model found in {Path.cwd()}: '{model_path}'.")
    else:
        from huggingface_hub import hf_hub_download

        print(
            "Downloading https://huggingface.co/emotion2vec/emotion2vec_base/"
            f"resolve/main/emotion2vec_base.pt to '{Path.cwd()}'"
        )
        model_path = hf_hub_download(
            repo_id="emotion2vec/emotion2vec_base",
            filename="emotion2vec_base.pt",
            local_dir=str(Path.cwd()),
        )
        # check if model exists
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

    return EMO2VEC(str(model_path), use_gpu=use_gpu)


def emotion_similarity_metric(model, ref, inf, fs):
    """Calculate the emotion similarity between two audio samples.

    Args:
        model (EMO2VEC): The loaded EMO2VEC model.
        model (EMO2VEC): loaded EMO2VEC model
            Please use the model in https://huggingface.co/emotion2vec/emotion2vec_base
            to get comparable results.
        ref (np.ndarray): reference signal (time,)
        inf (np.ndarray): enhanced signal (time,)
        fs (int): sampling rate in Hz.

    Returns:
        similarity (float): emotion similarity score.
    """
    if fs != TARGET_FS:
        inf = soxr.resample(inf, fs, TARGET_FS)
        ref = soxr.resample(ref, fs, TARGET_FS)
        fs = TARGET_FS

    emb_inf = model.extract_feature(inf, fs=fs)
    emb_ref = model.extract_feature(ref, fs=fs)
    similarity = np.dot(emb_inf, emb_ref) / (
        np.linalg.norm(emb_inf) * np.linalg.norm(emb_ref)
    )
    return similarity


################################################################
# Main entry
################################################################
def main(args):
    refs = {}
    with open(args.ref_scp, "r") as f:
        for line in f:
            uid, audio_path = line.strip().split()
            refs[uid] = audio_path

    data_pairs = []
    with open(args.inf_scp, "r") as f:
        for line in f:
            uid, audio_path = line.strip().split()
            data_pairs.append((uid, refs[uid], audio_path))

    size = len(data_pairs)
    assert 1 <= args.job <= args.nsplits <= size
    interval = size // args.nsplits
    start = (args.job - 1) * interval
    end = size if args.job == args.nsplits else start + interval
    data_pairs = data_pairs[start:end]
    print(
        f"[Job {args.job}/{args.nsplits}] Processing ({len(data_pairs)}/{size}) samples",
        flush=True,
    )
    suffix = "" if args.nsplits == args.job == 1 else f".{args.job}"

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    writers = {
        metric: (outdir / f"{metric}{suffix}.scp").open("w") for metric in METRICS
    }

    model = emo2vec_setup(use_gpu="cuda" in args.device)
    ret = []
    for uid, ref_audio, inf_audio in tqdm(data_pairs):
        _, score = process_one_pair((uid, ref_audio, inf_audio), model=model)
        ret.append((uid, score))
        for metric, value in score.items():
            writers[metric].write(f"{uid} {value}\n")

    for metric in METRICS:
        writers[metric].close()

    if args.nsplits == args.job == 1:
        with (outdir / "RESULTS.txt").open("w") as f:
            for metric in METRICS:
                mean_score = np.nanmean([score[metric] for uid, score in ret])
                f.write(f"{metric}: {mean_score:.4f}\n")
        print(
            f"Overall results have been written in {outdir / 'RESULTS.txt'}", flush=True
        )


def process_one_pair(data_pair, model=None):
    uid, ref_path, inf_path = data_pair
    ref, fs = sf.read(ref_path, dtype="float32")
    inf, fs2 = sf.read(inf_path, dtype="float32")
    assert fs == fs2, (fs, fs2)
    assert ref.shape == inf.shape, (ref.shape, inf.shape)
    assert ref.ndim == 1, ref.shape

    scores = {}
    for metric in METRICS:
        if metric == "EmotionSimilarity":
            scores[metric] = emotion_similarity_metric(model, ref, inf, fs=fs)
        else:
            raise NotImplementedError(metric)

    return uid, scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref_scp",
        type=str,
        required=True,
        help="Path to the scp file containing reference signals",
    )
    parser.add_argument(
        "--inf_scp",
        type=str,
        required=True,
        help="Path to the scp file containing enhanced signals",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for writing metrics",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for running emotion embedding extraction",
    )
    parser.add_argument(
        "--nsplits",
        type=int,
        default=1,
        help="Total number of computing nodes to speed up evaluation",
    )
    parser.add_argument(
        "--job",
        type=int,
        default=1,
        help="Index of the current node (starting from 1)",
    )
    args = parser.parse_args()

    main(args)
