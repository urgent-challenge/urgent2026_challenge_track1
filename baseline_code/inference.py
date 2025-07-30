from distutils.util import strtobool
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch
from tqdm import tqdm
from baseline_code.d_model import SEModel
from baseline_code.flow_model import FlowSEModel
import os
import tqdm



def str2bool(value: str) -> bool:
    return bool(strtobool(value))





################################################################
# Main entry
################################################################
def main(args):

    device = args.device

    try:
        model = SEModel.load_from_checkpoint(args.ckpt_path, map_location=args.device)
    except:
        model = FlowSEModel.load_from_checkpoint(args.ckpt_path, map_location=args.device)
    model.eval()

    input_audios = {}
    with open(args.input_scp) as f:
        for line in f:
            utt, wav = line.strip().split()
            input_audios[utt] = wav


    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir + "/wav", exist_ok=True)

    with open(args.output_dir + "/inf.scp", "w") as f:

        for uid in tqdm.tqdm(input_audios):
            wav_path = input_audios[uid]
            wav, sr = sf.read(wav_path)
            wav = torch.tensor(wav).float().to(device).view(1, -1)
            length = torch.tensor(wav.shape[-1]).to(device).view(1)

            with torch.no_grad():
                if isinstance(model, SEModel):
                    enhanced, _ = model.se_model(wav, length, sr)
                elif isinstance(model, FlowSEModel):
                    enhanced = model.enhance(wav, sr, length)

                enhanced = enhanced / enhanced.abs().max() * 0.9

                sf.write(args.output_dir + f"/wav/{uid}.wav", enhanced.cpu().numpy().flatten(), sr)

            print(f"{uid} {args.output_dir}/wav/{uid}.wav", file=f)



    print("done")










if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_scp",
        type=str,
        required=False,
        help="Path to the tsv file containing audio samples",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="./tmp/se",
        help="Path to the output directory for writting enhanced speeches",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=False,
        default="./tmp/se",
        help="Path to the checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for running TorchAudio-SQUIM calculation",
    )

    args = parser.parse_args()

    main(args)
