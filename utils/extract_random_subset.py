import argparse
import random

import soundfile as sf
from espnet2.utils import config_argparse
from tqdm import tqdm


def select_audio(scp_file, num_data):
    # read scp file
    with open(scp_file, "r") as f:
        lines = f.readlines()
    random.shuffle(lines)

    selected_lines = []
    for i, line in enumerate(tqdm(lines)):
        utt_id, fs, audio_path = line.strip().split()

        # get length of audio_file
        with sf.SoundFile(audio_path) as audio_file:
            assert int(fs) == audio_file.samplerate, (fs, audio_file.samplerate)
            audio_duration = len(audio_file) / audio_file.samplerate

        if audio_duration < 2.0 or audio_duration > 15.0:
            continue

        # add the audio file
        selected_lines.append(line)

        if len(selected_lines) == num_data:
            break

    return selected_lines


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
        "--num_data_per_dataset",
        type=int,
        nargs="+",
        help="Number of data to be extracted from each dataset",
    )
    parser.add_argument(
        "--outfile", type=str, required=True, help="Path to the output json file"
    )
    group.add_argument("--seed", type=int, default=0, help="Random seed")

    parser.set_defaults(required=["speech_scps", "num_data_per_dataset"])
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    assert len(args.speech_scps) == len(args.num_data_per_dataset)

    # ensure reproducibility
    random.seed(args.seed)

    selected_audios = []
    for speech_scp, num_data in zip(args.speech_scps, args.num_data_per_dataset):
        selected_audios += select_audio(speech_scp, num_data)

    with open(args.outfile, "w") as f_out:
        f_out.writelines(selected_audios)
