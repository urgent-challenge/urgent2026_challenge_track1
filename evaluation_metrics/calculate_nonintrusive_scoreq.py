from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent / "lib/scoreq/src/scoreq/"))
import scoreq




METRICS = ("Scoreq",)
TARGET_FS = 16000


################################################################
# Definition of metrics
################################################################
def scoreq_metric(model, audio_path):
    """Calculate the Scoreq metric.

    Args:
        model (torch.nn.Module): Scoreq model
        audio_path: path to the enhanced signal
    Returns:
        pred_mos (float): predicted MOS value between [1, 5]
    """
    with torch.no_grad():
        pred_mos = model.predict(test_path=audio_path, ref_path=None)

    return pred_mos


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

    # Using the no-reference mode
    # The model checkpoint will be automatically downloaded under ./pt-models
    model = scoreq.Scoreq(data_domain="natural", mode="nr")
    ret = []
    for uid, inf_audio in tqdm(data_pairs):
        _, score = process_one_pair((uid, inf_audio), model=model)
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
    uid, inf_path = data_pair

    scores = {}
    for metric in METRICS:
        if metric == "Scoreq":
            scores[metric] = scoreq_metric(model, inf_path)
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
        help="Device for running Scoreq calculation",
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
