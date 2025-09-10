import json
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from tqdm import tqdm

from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch


METRICS = ("LAcc",)
TARGET_FS = 16000


################################################################
# Definition of metrics
################################################################
def owsm_lid_model_setup(model_tag="espnet/owsm_ctc_v4_1B", device="cpu"):
    s2t = Speech2TextGreedySearch.from_pretrained(
        model_tag,
        device=device,
        generate_interctc_outputs=False,
        lang_sym="<nolang>",
        task_sym="<asr>",
    )
    # To check the full list of supported languages, run
    #     lst = s2t.converter.token_list
    #     print(lst[lst.index("<abk>") : lst.index("<zul>")])
    return s2t


def convert_iso639_to_name(lang_code):
    """Convert ISO 639 language code to language name."""
    # pip install git+https://github.com/noumar/iso639
    from iso639 import languages as iso_languages

    if lang_code.startswith("<") and lang_code.endswith(">"):
        lang_code = lang_code[1:-1]
    if len(lang_code) == 3:
        return iso_languages.get(part3=lang_code).name
    elif len(lang_code) == 2:
        return iso_languages.get(part1=lang_code).name
    else:
        raise ValueError(f"Not a valid ISO 639 code: {lang_code}")


def convert_name_to_iso639(lang_name, part=3):
    """Convert language name to ISO 639 code."""
    from iso639 import languages as iso_languages

    assert part in (1, 3), part
    # Chinese -> <chi>
    lang = iso_languages.get(name=lang_name)
    if lang is None:
        # Note: the first letter must be capitalized
        raise ValueError(f"Not a valid language name: {lang_name}")
    return f"<{lang.part3}>" if part == 3 else f"<{lang.part1}>"


def predict_language_id(model, inf, fs):
    """
    Args:
        model (torch.nn.Module): Language identification (LID) model
            Please use the model in https://huggingface.co/espnet/owsm_ctc_v4_1B
            to get comparable results.
        inf (np.ndarray): enhanced signal (time,)
        fs (int): sampling rate in Hz
    Returns:
        lang_id (str): predicted language (token name), e.g., '<eng>'
    """
    if fs != TARGET_FS:
        inf = soxr.resample(inf, fs, TARGET_FS)
        fs = TARGET_FS

    text, tokens, tokens_int, text_nospecial, hyp = model(inf)[0]
    return tokens[0]


################################################################
# Main entry
################################################################
def main(args):
    language = {}
    if args.meta_tsv.endswith(".tsv"):
        with open(args.meta_tsv, "r") as f:
            headers = next(f).strip().split("\t")
            uid_idx = headers.index("id")
            txt_idx = headers.index("language")
            for line in f:
                tup = line.strip().split("\t")
                uid, txt = tup[uid_idx], tup[txt_idx]
                language[uid] = txt
    else:
        print(f"Assuming '{args.meta_tsv}' as scp file")
        with open(args.meta_tsv, "r") as f:
            for line in f:
                # uid1 <abk>
                # uid2 <eng>
                # uid3 <zul>
                uid, txt = line.strip().split(maxsplit=1)
                language[uid] = txt

    data_pairs = []
    with open(args.inf_scp, "r") as f:
        for line in f:
            uid, audio_path = line.strip().split()
            data_pairs.append((uid, language[uid], audio_path))

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

    model = owsm_lid_model_setup(model_tag="espnet/owsm_ctc_v4_1B", device=args.device)
    ret = []
    for uid, ref_lang, inf_audio in tqdm(data_pairs):
        _, score = process_one_pair((uid, ref_lang, inf_audio), model=model)
        ret.append((uid, score))
        for metric, value in score.items():
            if metric == "LAcc":
                s = json.dumps(value)
                writers[metric].write(f"{uid} {s}\n")

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
    uid, ref_lang, inf_path = data_pair
    inf, fs = sf.read(inf_path, dtype="float32")
    assert inf.ndim == 1, inf.shape

    scores = {}
    for metric in METRICS:
        if metric == "LAcc":
            # e.g., '<eng>', '<zho>'
            lang_id = predict_language_id(model, inf, fs=fs)
            lang_id = lang_id.replace("<", "").replace(">", "")
            scores[metric] = int(lang_id == ref_lang)
        else:
            raise NotImplementedError(metric)

    return uid, scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta_tsv",
        type=str,
        required=True,
        help="Path to the tsv file containing meta information about the data "
        "(including 'language')\n"
        "Alternatively, this can also be a scp file containing language ID per sample",
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
        help="Device for running LID inference",
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
