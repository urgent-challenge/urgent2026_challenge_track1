from distutils.util import strtobool
from pathlib import Path

import librosa
import numpy as np
import torch
from tqdm import tqdm

METRICS = ("UTMOS",)


def str2bool(value: str) -> bool:
    return bool(strtobool(value))


################################################################
# Definition of metrics
################################################################
def utmos_metric(model, audio_path):
    """Calculate the UTMOS metric.

    Args:
        model (torch.nn.Module): UTMOS model
        audio_path: path to the enhanced signal
    Returns:
        dnsmos (float): UTMOS value between [1, 5]
    """
    wave, sr = librosa.load(audio_path, sr=None, mono=True)
    wave = torch.from_numpy(wave).unsqueeze(0).to(device=model.device)
    utmos_score = model(wave, sr)
    return float(utmos_score.cpu().item())


################################################################
# Main entry
################################################################
def main(args):
    data_pairs = []
    with open(args.inf_scp, "r") as f:
        for line in f:
            uid, audio_path = line.strip().split()
            data_pairs.append((uid, audio_path))

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

    utmos_model = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0", args.utmos_tag, trust_repo=True
    ).to(device=args.device)
    utmos_model.device = args.device
    ret = []
    for uid, inf_audio in tqdm(data_pairs):
        _, score = process_one_pair(
            (uid, inf_audio), utmos_model=utmos_model
        )
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


def process_one_pair(data_pair, utmos_model=None):
    uid, inf_path = data_pair

    scores = {}
    for metric in METRICS:
        if metric == "UTMOS":
            scores[metric] = utmos_metric(utmos_model, inf_path)
        else:
            raise NotImplementedError(metric)

    return uid, scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
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
        help="Device for running non-instrusive MOS calculation",
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

    group = parser.add_argument_group("MOS model related")
    group.add_argument(
        "--utmos_tag",
        type=str,
        default="utmos22_strong",
        help="Tag of the UTMOS model to be used",
    )
    args = parser.parse_args()

    main(args)
