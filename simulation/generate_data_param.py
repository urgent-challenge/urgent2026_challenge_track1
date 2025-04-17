import argparse
import random
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool
from tqdm import tqdm

# Avaiable sampling rates for bandwidth limitation
SAMPLE_RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)

RESAMPLE_METHODS = (
    "kaiser_best",
    "kaiser_fast",
    "scipy",
    "polyphase",
    #    "linear",
    #    "zero_order_hold",
    #    "sinc_best",
    #    "sinc_fastest",
    #    "sinc_medium",
)

AUGMENTATIONS = ("bandwidth_limitation", "clipping")


#############################
# Augmentations per sample
#############################
def bandwidth_limitation(fs: int = 16000, res_type="random"):
    """Apply the bandwidth limitation distortion to the input signal.

    Args:
        fs (int): sampling rate in Hz
        res_type (str): resampling method

    Returns:
        res_type (str): adopted resampling method
        fs_new (int): effective sampling rate in Hz
    """
    # resample to a random sampling rate
    fs_opts = [fs_new for fs_new in SAMPLE_RATES if fs_new < fs]
    if fs_opts:
        if res_type == "random":
            res_type = np.random.choice(RESAMPLE_METHODS)
        fs_new = np.random.choice(fs_opts)
        opts = {"res_type": res_type}
    else:
        res_type = "none"
        fs_new = fs
    return res_type, fs_new


def packet_loss(
    speech_length, fs, packet_duration_ms, packet_loss_rate, max_continuous_packet_loss
):
    """Returns a list of indices (of packets) that are zeroed out."""

    # speech duration in ms and the number of packets
    speech_duration_ms = speech_length / fs * 1000
    num_packets = int(speech_duration_ms // packet_duration_ms)

    # randomly select the packet loss rate and calculate the packet loss duration
    packet_loss_rate = np.random.uniform(*packet_loss_rate)
    packet_loss_duration_ms = packet_loss_rate * speech_duration_ms

    # calculate the number of packets to be zeroed out
    num_packet_loss = int(round(packet_loss_duration_ms / packet_duration_ms, 0))

    # list of length of each packet loss
    packet_loss_lengths = []
    for _ in range(num_packet_loss):
        num_continuous_packet_loss = np.random.randint(1, max_continuous_packet_loss)
        packet_loss_lengths.append(num_continuous_packet_loss)

        if num_packet_loss - sum(packet_loss_lengths) <= max_continuous_packet_loss:
            packet_loss_lengths.append(num_packet_loss - sum(packet_loss_lengths))
            break

    packet_loss_start_indices = np.random.choice(
        range(num_packets), len(packet_loss_lengths), replace=False
    )
    packet_loss_indices = []
    for idx, length in zip(packet_loss_start_indices, packet_loss_lengths):
        packet_loss_indices += list(range(idx, idx + length))

    return list(set(packet_loss_indices))


def weighted_sample(population, weights, k, replace=True, rng=np.random):
    weights = np.array(weights)
    weights = weights / weights.sum()
    idx = rng.choice(range(len(population)), size=k, replace=replace, p=weights)
    return [population[i] for i in idx]


#############################
# Audio utilities
#############################
def read_audio(filename, force_1ch=False, fs=None):
    audio, fs_ = sf.read(filename, always_2d=True)
    audio = audio[:, :1].T if force_1ch else audio.T
    if fs is not None and fs != fs_:
        audio = librosa.resample(audio, orig_sr=fs_, target_sr=fs, res_type="soxr_hq")
        return audio, fs
    return audio, fs_


def save_audio(audio, filename, fs):
    if audio.ndim != 1:
        audio = audio[0] if audio.shape[0] == 1 else audio.T
    sf.write(filename, audio, samplerate=fs)


#############################
# Main entry
#############################
def main(args):
    speech_dic = defaultdict(dict)
    # scp file of clean speech samples (three columns per line: uid, fs, audio_path)
    for scp in args.speech_scps:
        with open(scp, "r") as f:
            for line in f:
                uid, fs, audio_path = line.strip().split()
                assert uid not in speech_dic[int(fs)], (uid, fs)
                speech_dic[int(fs)][uid] = audio_path

    # speaker ID of each sample (two columns per line: uid, speaker_id)
    utt2spk = {}
    for scp in args.speech_utt2spk:
        with open(scp, "r") as f:
            for line in f:
                uid, sid = line.strip().split()
                assert uid not in utt2spk, (uid, sid)
                utt2spk[uid] = sid

    # transcript of each sample (two columns per line: uid, text)
    text = {}
    for scp in args.speech_text:
        with open(scp, "r") as f:
            for line in f:
                uid, txt = line.strip().split(maxsplit=1)
                assert uid not in text, (uid, txt)
                text[uid] = txt

    # scp file of noise samples (three columns per line: uid, fs, audio_path)
    noise_dic = defaultdict(dict)
    for scp in args.noise_scps:
        with open(scp, "r") as f:
            for line in f:
                uid, fs, audio_path = line.strip().split()
                assert uid not in noise_dic[int(fs)], (uid, fs)
                noise_dic[int(fs)][uid] = audio_path
    used_noise_dic = {fs: {} for fs in noise_dic.keys()}

    # scp file of noise samples (three columns per line: uid, fs, audio_path)
    wind_noise_dic = defaultdict(dict)
    for scp in args.wind_noise_scps:
        with open(scp, "r") as f:
            for line in f:
                uid, fs, audio_path = line.strip().split()
                assert uid not in wind_noise_dic[int(fs)], (uid, fs)
                wind_noise_dic[int(fs)][uid] = audio_path
    used_wind_noise_dic = {fs: {} for fs in wind_noise_dic.keys()}

    # [optional] scp file of RIR samples (three columns per line: uid, fs, audio_path)
    rir_dic = None
    if args.rir_scps is not None and args.prob_reverberation > 0.0:
        rir_dic = defaultdict(dict)
        for scp in args.rir_scps:
            with open(scp, "r") as f:
                for line in f:
                    uid, fs, audio_path = line.strip().split()
                    assert uid not in rir_dic[int(fs)], (uid, fs)
                    rir_dic[int(fs)][uid] = audio_path
    used_rir_dic = {fs: {} for fs in rir_dic.keys()} if rir_dic is not None else None

    f = open(Path(args.log_dir) / "meta.tsv", "w")
    headers = [
        "id",
        "noisy_path",
        "speech_uid",
        "speech_sid",
        "clean_path",
        "noise_uid",
    ]
    if args.store_noise:
        headers.append("noise_path")
    headers += ["snr_dB", "rir_uid", "augmentation", "fs", "length", "text"]
    f.write("\t".join(headers) + "\n")

    outdir = Path(args.output_dir)
    snr_range = (args.snr_low_bound, args.snr_high_bound)
    wind_noise_snr_range = (
        args.wind_noise_snr_low_bound,
        args.wind_noise_snr_high_bound,
    )

    augmentations = list(args.augmentations.keys())
    weight_augmentations = [v["weight"] for v in args.augmentations.values()]
    weight_augmentations = weight_augmentations / np.sum(weight_augmentations)

    count = 0
    for fs in sorted(speech_dic.keys(), reverse=True):
        for uid, audio_path in tqdm(speech_dic[fs].items()):
            sid = utt2spk[uid]
            transcript = text.get(uid, "<not-available>")  # placeholder of missing text
            # Load speech sample (Channel, Time)
            if audio_path.endswith(".wav"):
                with sf.SoundFile(audio_path) as af:
                    speech_length = af.frames
            else:
                # Sometimes the acutal loaded audio's length differs from af.frames
                speech_length = sf.read(audio_path)[0].shape[0]

            for n in range(args.repeat_per_utt):
                use_wind_noise = np.random.random() < args.prob_wind_noise

                num_aug = np.random.choice(
                    list(args.num_augmentations.keys()),
                    p=list(args.num_augmentations.values()),
                )
                if num_aug == 0:
                    aug = "none"
                else:
                    aug = np.random.choice(
                        augmentations,
                        p=weight_augmentations,
                        size=num_aug,
                        replace=False,
                    )
                    # As wind-noise simulation include clipping,
                    # we exclude clipping from augmentation list
                    while use_wind_noise and "clipping" in aug:
                        aug = np.random.choice(
                            augmentations,
                            p=weight_augmentations,
                            size=num_aug,
                            replace=False,
                        )

                info = process_one_sample(
                    args,
                    speech_length,
                    fs,
                    noise_dic=noise_dic,
                    used_noise_dic=used_noise_dic,
                    wind_noise_dic=wind_noise_dic,
                    used_wind_noise_dic=used_wind_noise_dic,
                    use_wind_noise=use_wind_noise,
                    snr_range=snr_range,
                    wind_noise_snr_range=wind_noise_snr_range,
                    store_noise=args.store_noise,
                    rir_dic=rir_dic,
                    used_rir_dic=used_rir_dic,
                    augmentations=aug,
                    force_1ch=True,
                )
                count += 1

                # limit the number of files in each directory to 5000
                filedir = str(count // 5000)
                (outdir / "noisy" / filedir).mkdir(parents=True, exist_ok=True)
                (outdir / "clean" / filedir).mkdir(parents=True, exist_ok=True)

                filename = f"fileid_{count}.{args.out_format}"
                lst = [
                    f"fileid_{count}",
                    str(outdir / "noisy" / filedir / filename),
                    uid,
                    sid,
                    str(outdir / "clean" / filedir / filename),
                    info["noise_uid"],
                ]
                if args.store_noise:
                    (outdir / "noise" / filedir).mkdir(parents=True, exist_ok=True)
                    lst.append(str(outdir / "noise" / filedir / filename))
                lst += [
                    str(info["snr"]),
                    info["rir_uid"],
                    info["augmentation"],
                    str(info["fs"]),
                    str(info["length"]),
                    transcript,
                ]
                f.write("\t".join(lst) + "\n")
    f.close()


def process_one_sample(
    args,
    speech_length,
    fs,
    noise_dic,
    used_noise_dic,
    wind_noise_dic,
    used_wind_noise_dic,
    snr_range,
    wind_noise_snr_range,
    use_wind_noise,
    store_noise=False,
    rir_dic=None,
    used_rir_dic=None,
    augmentations="none",
    force_1ch=True,
):
    # select a noise sample
    if use_wind_noise:
        noise_uid, _ = select_sample(
            fs, wind_noise_dic, used_sample_dic=used_wind_noise_dic, reuse_sample=True
        )

        # wind-noise simulation config
        wn_conf = args.wind_noise_config
        threshold = np.random.uniform(*wn_conf["threshold"])
        ratio = np.random.uniform(*wn_conf["ratio"])
        attack = np.random.uniform(*wn_conf["attack"])
        release = np.random.uniform(*wn_conf["release"])
        sc_gain = np.random.uniform(*wn_conf["sc_gain"])
        clipping_threshold = np.random.uniform(*wn_conf["clipping_threshold"])
        clipping = np.random.random() < wn_conf["clipping_chance"]
        augmentation_config = (
            "wind_noise("
            f"threshold={threshold},ratio={ratio},"
            f"attack={attack},release={release},"
            f"sc_gain={sc_gain},clipping={clipping},"
            f"clipping_threshold={clipping_threshold})/"
        )
        snr = np.random.uniform(*wind_noise_snr_range)
    else:
        noise_uid, noise = select_sample(
            fs, noise_dic, used_sample_dic=used_noise_dic, reuse_sample=args.reuse_noise
        )
        augmentation_config = ""
        snr = np.random.uniform(*snr_range)
    if noise_uid is None:
        raise ValueError(f"Noise sample not found for fs={fs}+ Hz")

    # select a room impulse response (RIR)
    if (
        rir_dic is None
        or args.prob_reverberation <= 0.0
        or np.random.rand() <= args.prob_reverberation
    ):
        rir_uid, rir = None, None
    else:
        rir_uid, rir = select_sample(
            fs, rir_dic, used_sample_dic=used_rir_dic, reuse_sample=args.reuse_rir
        )

    # apply an additional augmentation
    if isinstance(augmentations, str) and augmentations == "none":
        if not use_wind_noise:
            augmentation_config = "none"
    else:
        for i, augmentation in enumerate(augmentations):
            this_aug = args.augmentations[augmentation]
            if augmentation == "bandwidth_limitation":
                res_type, fs_new = bandwidth_limitation(fs=fs, res_type="random")
                augmentation_config += f"{augmentation}-{res_type}->{fs_new}"
            elif augmentation == "clipping":
                min_quantile = np.random.uniform(*this_aug["clipping_min_quantile"])
                max_quantile = np.random.uniform(*this_aug["clipping_max_quantile"])
                augmentation_config += (
                    f"{augmentation}(min={min_quantile},max={max_quantile})"
                )
            elif augmentation == "codec":
                # vbr_quality = np.random.uniform(*this_aug["vbr_quality"])
                # augmentation_config += f"{augmentation}(vbr_quality={vbr_quality})"
                codec_config = np.random.choice(this_aug["config"], 1)[0]
                format, encoder, qscale = (
                    codec_config["format"],
                    codec_config["encoder"],
                    codec_config["qscale"],
                )
                if encoder is not None and isinstance(encoder, list):
                    encoder = np.random.choice(encoder, 1)[0]
                if qscale is not None and isinstance(qscale, list):
                    qscale = np.random.randint(*qscale)
                augmentation_config += (
                    f"{augmentation}"
                    f"(format={format},encoder={encoder},qscale={qscale})"
                )

            elif augmentation == "packet_loss":
                packet_duration_ms = this_aug["packet_duration_ms"]
                packet_loss_indices = packet_loss(
                    speech_length,
                    fs,
                    packet_duration_ms,
                    this_aug["packet_loss_rate"],
                    this_aug["max_continuous_packet_loss"],
                )
                augmentation_config += (
                    f"{augmentation}"
                    f"(packet_loss_indices={packet_loss_indices},"
                    f"packet_duration_ms={packet_duration_ms})"
                )
            else:
                raise NotImplementedError(augmentation)

            # / is used for splitting multiple augmentation configuration
            if i < len(augmentations) - 1:
                augmentation_config += "/"

    meta = {
        "noise_uid": "none" if noise_uid is None else noise_uid,
        "rir_uid": "none" if rir_uid is None else rir_uid,
        "snr": snr,
        "augmentation": augmentation_config,
        "fs": fs,
        "length": speech_length,
    }
    return meta


def select_sample(fs, sample_dic, used_sample_dic=None, reuse_sample=False):
    """Randomly select a sample from the given dictionary.

    First try to select an unused sample with the same sampling rate (= fs).
    Then try to select an unused sample with a higher sampling rate (> fs).
    If no unused sample is found and reuse_sample=True,
        try to select a used sample with the same strategy.
    """
    if fs not in sample_dic.keys() or len(sample_dic[fs]) == 0:
        fs_opts = list(sample_dic.keys())
        np.random.shuffle(fs_opts)
        for fs2 in fs_opts:
            if fs2 > fs and len(sample_dic[fs2]) > 0:
                uid = np.random.choice(list(sample_dic[fs2].keys()))
                if used_sample_dic is not None:
                    sample = sample_dic[fs2].pop(uid)
                    used_sample_dic[fs2][uid] = sample
                else:
                    sample = sample_dic[fs2][uid]
                break
        else:
            if reuse_sample:
                return select_sample(fs, used_sample_dic, reuse_sample=False)
            return None, None
    else:
        uid = np.random.choice(list(sample_dic[fs].keys()))
        if used_sample_dic is not None:
            sample = sample_dic[fs].pop(uid)
            used_sample_dic[fs][uid] = sample
        else:
            sample = sample_dic[fs][uid]
    return uid, sample


#############################
# Commandline related
#############################
def get_parser(parser=None):
    if parser is None:

        class ArgumentDefaultsRawTextHelpFormatter(
            argparse.RawTextHelpFormatter,
            argparse.ArgumentDefaultsHelpFormatter,
        ):
            pass

        # support --config to specify all arguments in a yaml file
        parser = config_argparse.ArgumentParser(
            description="base parser",
            formatter_class=ArgumentDefaultsRawTextHelpFormatter,
        )

    group = parser.add_argument_group(description="General arguments")
    group.add_argument(
        "--speech_scps",
        type=str,
        nargs="+",
        help="Path to the scp file containing speech samples",
    )
    group.add_argument(
        "--speech_utt2spk",
        type=str,
        nargs="+",
        help="Path to the utt2spk file containing speaker mappings",
    )
    group.add_argument(
        "--speech_text",
        type=str,
        nargs="+",
        help="Path to the text file containing transcripts",
    )
    group.add_argument(
        "--log_dir",
        type=str,
        help="Log directory for storing log and scp files",
    )
    group.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for storing processed audio files",
    )
    group.add_argument(
        "--out_format", type=str, default="flac", help="Output audio format"
    )
    group.add_argument(
        "--repeat_per_utt",
        type=int,
        default=1,
        help="Number of times to use each utterance\n"
        "(The final amount of simulated samples will be "
        "`repeat_per_utt` * size(speech_scp))",
    )
    group.add_argument("--seed", type=int, default=0, help="Random seed")

    group = parser.add_argument_group(description="Additive noise related")
    group.add_argument(
        "--noise_scps",
        type=str,
        nargs="+",
        help="Path to the scp file containing noise samples",
    )
    group.add_argument(
        "--snr_low_bound",
        type=float,
        default=-5.0,
        help="Lower bound of signal-to-noise ratio (SNR) in dB",
    )
    group.add_argument(
        "--snr_high_bound",
        type=float,
        default=20.0,
        help="Higher bound of signal-to-noise ratio (SNR) in dB",
    )
    group.add_argument(
        "--reuse_noise",
        type=str2bool,
        default=False,
        help="Whether or not to allow reusing noise samples",
    )
    group.add_argument(
        "--store_noise",
        type=str2bool,
        default=False,
        help="Whether or not to store parallel noise samples",
    )

    group = parser.add_argument_group(description="Wind-noise related")
    group.add_argument(
        "--wind_noise_scps",
        type=str,
        nargs="+",
        help="Path to the scp file containing wind noise samples\n"
        "(If not provided, wind noise will not be applied)",
    )
    group.add_argument(
        "--prob_wind_noise",
        type=float,
        default=0.05,
        help="Probability of using wind noise instead of other environmental noise"
        "to input speech samples",
    )
    group.add_argument(
        "--wind_noise_config",
        type=dict,
        default={},
        help="Whether or not to allow reusing wind noise samples",
    )
    group.add_argument(
        "--reuse_wind_noise",
        type=str2bool,
        default=False,
        help="Whether or not to allow reusing wind noise samples",
    )
    group.add_argument(
        "--wind_noise_snr_low_bound",
        type=float,
        default=-5.0,
        help="Lower bound of signal-to-noise ratio (SNR) in dB",
    )
    group.add_argument(
        "--wind_noise_snr_high_bound",
        type=float,
        default=20.0,
        help="Higher bound of signal-to-noise ratio (SNR) in dB",
    )

    group = parser.add_argument_group(description="Reverberation related")
    group.add_argument(
        "--rir_scps",
        type=str,
        nargs="+",
        help="Path to the scp file containing RIR samples\n"
        "(If not provided, reverberation will not be applied)",
    )
    group.add_argument(
        "--prob_reverberation",
        type=float,
        default=0.5,
        help="Probability of randomly adding reverberation to input speech samples",
    )
    group.add_argument(
        "--reuse_rir",
        type=str2bool,
        default=False,
        help="Whether or not to allow reusing RIR samples",
    )

    group = parser.add_argument_group(description="Additional augmentation related")
    group.add_argument(
        "--augmentations",
        default=dict(none=dict(weight=1.0)),
        help="Dict of mutually-exclusive augmentations to apply to input speech "
        "samples",
    )
    group.add_argument(
        "--num_augmentations",
        default=dict(),
        help="Dict of mutually-exclusive augmentations to apply to input speech "
        "samples",
    )
    parser.set_defaults(required=["speech_scps", "log_dir", "output_dir", "noise_scps"])
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    assert len(args.speech_utt2spk) == len(args.speech_scps)
    if args.speech_text:
        assert len(args.speech_text) == len(args.speech_scps)
    if args.prob_reverberation > 0:
        assert args.rir_scps

    outdir = Path(args.output_dir)
    (outdir / "clean").mkdir(parents=True, exist_ok=True)
    (outdir / "noisy").mkdir(parents=True, exist_ok=True)
    if args.store_noise:
        (outdir / "noise").mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    main(args)
