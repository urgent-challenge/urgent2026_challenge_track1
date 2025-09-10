import json
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch
from espnet2.bin.s2t_inference import Speech2Text
from espnet2.text.cleaner import TextCleaner
from Levenshtein import opcodes
from owsm_utils import owsm_predict
from tqdm import tqdm

METRICS = ("WER", "CER")
TARGET_FS = 16000
BEAMSIZE = 5


################################################################
# Definition of metrics
################################################################
def levenshtein_metric(model, textcleaner, ref_txt, inf, lang_id, fs=16000):
    """Calculate the Levenshtein distance between ref and inf ASR results.

    Args:
        model (torch.nn.Module): ASR model
            Please use the model in https://huggingface.co/espnet/owsm_v3.1_ebf
            to get comparable results.
        textcleaner: Text normalization module
            Please use Whisper's normalizer ("whisper_basic") to get comparable results.
        ref_txt (string): reference transcript
        inf (np.ndarray): enhanced signal (time,)
        fs (int): sampling rate in Hz
    Returns:
        ret (dict): ditionary containing occurrences of edit operations
    """
    if ref_txt == "<not-available>":
        # Skip samples without reference transcripts
        return dict(WER={}, CER={})
    if fs != TARGET_FS:
        inf = soxr.resample(inf, fs, TARGET_FS)
        fs = TARGET_FS

    with torch.no_grad():
        inf_txt = owsm_predict(
            model,
            inf,
            fs,
            src_lang=lang_id,
            beam_size=BEAMSIZE,
            long_form=len(inf) > 30 * fs,
        )
    ref_txt = textcleaner(ref_txt)
    inf_txt = textcleaner(inf_txt)
    ref_words = ref_txt.split()
    inf_words = inf_txt.split()
    ret_wer = {
        "hyp_text": inf_txt,
        "ref_text": ref_txt,
        "delete": 0,
        "insert": 0,
        "replace": 0,
        "equal": 0,
    }
    for op, ref_st, ref_et, inf_st, inf_et in opcodes(ref_words, inf_words):
        if op == "insert":
            ret_wer[op] = ret_wer[op] + inf_et - inf_st
        else:
            ret_wer[op] = ret_wer[op] + ref_et - ref_st
    total = ret_wer["delete"] + ret_wer["replace"] + ret_wer["equal"]
    assert total == len(ref_words), (total, len(ref_words))
    total = ret_wer["insert"] + ret_wer["replace"] + ret_wer["equal"]
    assert total == len(inf_words), (total, len(inf_words))

    # process cer
    ref_words = [c for c in ref_txt]
    inf_words = [c for c in inf_txt]
    ret_cer = {
        "hyp_text": inf_words,
        "ref_text": ref_words,
        "delete": 0,
        "insert": 0,
        "replace": 0,
        "equal": 0,
    }
    for op, ref_st, ref_et, inf_st, inf_et in opcodes(ref_words, inf_words):
        if op == "insert":
            ret_cer[op] = ret_cer[op] + inf_et - inf_st
        else:
            ret_cer[op] = ret_cer[op] + ref_et - ref_st
    total = ret_cer["delete"] + ret_cer["replace"] + ret_cer["equal"]
    assert total == len(ref_words), (total, len(ref_words))
    total = ret_cer["insert"] + ret_cer["replace"] + ret_cer["equal"]
    assert total == len(inf_words), (total, len(inf_words))

    ret = dict(WER=ret_wer, CER=ret_cer)
    return ret


################################################################
# Main entry
################################################################
def main(args):
    transcripts = {}
    language_id = {}
    if args.meta_tsv.endswith(".tsv"):
        assert False, "tsv is not supported now"
        with open(args.meta_tsv, "r") as f:
            headers = next(f).strip().split("\t")
            uid_idx = headers.index("id")
            txt_idx = headers.index("text")
            for line in f:
                tup = line.strip().split("\t")
                uid, txt = tup[uid_idx], tup[txt_idx]
                transcripts[uid] = txt
    else:
        print(f"Assuming '{args.meta_tsv}' as scp file")
        with open(args.meta_tsv, "r") as f:
            for line in f:
                uid, txt = line.strip().split(maxsplit=1)
                transcripts[uid] = txt

        with open(args.utt2lang, "r") as f:
            for line in f:
                uid, lang_id = line.strip().split(maxsplit=1)
                assert uid in transcripts, uid
                language_id[uid] = lang_id

    data_pairs = []
    with open(args.inf_scp, "r") as f:
        for line in f:
            uid, audio_path = line.strip().split()
            data_pairs.append((uid, transcripts[uid], audio_path, language_id[uid]))

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

    model = Speech2Text.from_pretrained(
        model_tag="espnet/owsm_v3.1_ebf",
        device=args.device,
        # lang_sym="<eng>",
        task_sym="<asr>",
        beam_size=BEAMSIZE,
        predict_time=False,
    )
    # print([a for a in model.s2t_model.token_list if "<" in a and ">" in a], flush=True)
    textcleaner = TextCleaner("whisper_basic")
    ret = []
    for uid, ref_text, inf_audio, lang_id in tqdm(data_pairs):
        _, score = process_one_pair(
            (uid, ref_text, inf_audio, lang_id), model=model, textcleaner=textcleaner
        )
        ret.append((uid, score))
        for metric, value in score.items():
            # if metric == "WER":
            if metric in ["WER", "CER"]:
                s = json.dumps(value)
                writers[metric].write(f"{uid} {s}\n")

    for metric in METRICS:
        writers[metric].close()

    if args.nsplits == args.job == 1:
        with (outdir / "RESULTS.txt").open("w") as f:
            for metric in METRICS:
                # if metric == "WER":
                if metric in ["WER", "CER"]:
                    dic = {"delete": [], "insert": [], "replace": [], "equal": []}
                    for uid, score in ret:
                        if len(score[metric]) == 0:
                            continue
                        for op in dic.keys():
                            dic[op].append(score[metric][op])
                    dic = {op: sum(count) for op, count in dic.items()}
                    numerator = dic["replace"] + dic["delete"] + dic["insert"]
                    denominator = dic["replace"] + dic["delete"] + dic["equal"]
                    wer = numerator / denominator
                    f.write(f"{metric}: {wer:.4f}\n")
                    for op, count in dic.items():
                        f.write(f"    {op}: {count}\n")
                else:
                    mean_score = np.nanmean([score[metric] for uid, score in ret])
                    f.write(f"{metric}: {mean_score:.4f}\n")
        print(
            f"Overall results have been written in {outdir / 'RESULTS.txt'}", flush=True
        )


def process_one_pair(data_pair, model=None, textcleaner=None):
    uid, ref_txt, inf_path, lang_id = data_pair
    inf, fs = sf.read(inf_path, dtype="float32")
    assert inf.ndim == 1, inf.shape

    scores = {}
    ret = levenshtein_metric(model, textcleaner, ref_txt, inf, lang_id, fs=fs)
    for metric in METRICS:
        if metric in ["WER", "CER"]:
            # scores[metric] = levenshtein_metric(model, textcleaner, ref_txt, inf, lang_id, fs=fs)
            scores[metric] = ret[metric]
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
        "(including transcripts)\n"
        "Alternatively, this can also be a scp file containing transcript per sample",
    )
    parser.add_argument(
        "--inf_scp",
        type=str,
        required=True,
        help="Path to the scp file containing enhanced signals",
    )
    parser.add_argument(
        "--utt2lang",
        type=str,
        required=True,
        help="Path to the scp file containing language id",
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
        help="Device for running ASR inference",
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
