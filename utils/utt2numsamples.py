import argparse
import random

import soundfile as sf
from espnet2.utils import config_argparse
from tqdm import tqdm
from collections import defaultdict





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
        "--input_scp",
        type=str,
        required=True,
        help="Path to the scp file containing speech samples",
    )

    parser.add_argument(
        "--outfile", type=str, required=True, help="Path to the output json file"
    )
    return parser



def read_source_scp(scp):
    source_dict = defaultdict(dict)
    source_dict_flatten = {}
    with open(scp, "r") as f:
        for line in f:
            line = line.strip().split()
            if len(line) == 3:
                uid, fs, audio_path = line
            else:
                uid, audio_path = line
                fs = 0

            assert uid not in source_dict[int(fs)], (uid, fs)
            source_dict[int(fs)][uid] = audio_path
            source_dict_flatten[uid] = audio_path

    source_uids =  {k: list(source_dict[k].keys()) for k in source_dict}

    return source_dict, source_uids, source_dict_flatten

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    
    _, _, scp_dict = read_source_scp(args.input_scp)


    with open(args.outfile, "w") as f_out:
        for uid, speech_path in tqdm(scp_dict.items()):
            if speech_path.endswith(".wav"):
                with sf.SoundFile(speech_path) as af:
                    speech_length = af.frames
            else:
                # Sometimes the acutal loaded audio's length differs from af.frames
                speech_length = sf.read(speech_path)[0].shape[0]  
            
            print(f"{uid} {speech_length}", file=f_out)
