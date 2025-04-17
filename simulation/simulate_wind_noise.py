"""Example to generate a wind noise signal."""

import argparse
import os
from pathlib import Path

import numpy as np
import tqdm
import yaml

# Need access to the WindNoiseGenerator library (file: sc_wind_noise_generator.py) presented in D. Mirabilii et al. "Simulating wind noise with airflow speed-dependent characteristics,” in Int. Workshop on Acoustic Signal Enhancement, Sept. 2022"
# Please ask the authors as we are not responsible for the distribution of their code
from sc_wind_noise_generator import WindNoiseGenerator as wng

parser = argparse.ArgumentParser()

parser.add_argument("--output_dir", type=Path)
parser.add_argument("--config", type=Path)

args = parser.parse_args()

with open(args.config, "r") as yml:
    config = yaml.safe_load(yml)
print(config)

# params = vars(args)
# params = {**wind_params, **params}

if os.path.exists(args.output_dir):
    raise RuntimeError(
        f"{args.output_dir} already exists."
        "Please delete it if you want to run again."
    )
args.output_dir.mkdir(parents=True)

scp = open(args.output_dir / "wind_noise.scp", "w")
for seed, sample_rate in zip(config["seeds"], config["sample_rates"]):
    output_dir = args.output_dir / f"wind_noise_{sample_rate}hz"
    output_dir.mkdir(parents=True)

    for i in tqdm.tqdm(range(config["num_data"])):
        # Generate wind profile
        gustiness = np.random.uniform(
            config["gustiness_range"][0], config["gustiness_range"][1]
        )  # Number of speed points. One yields constant wind. High values yields gusty wind.
        number_points_wind_profile = int(1.5 * gustiness)

        # wind_profile is ignored when generate=True is given to wng
        # so wind_profile defined here is just a placeholder and you can ignore it.
        wind_profile = [
            np.random.uniform(
                config["wind_profile_magnitude_range"][0],
                config["wind_profile_magnitude_range"][1],
            )
        ]

        while len(wind_profile) < number_points_wind_profile:
            is_valid = False
            while not is_valid:
                new_point = np.random.uniform(
                    config["wind_profile_magnitude_range"][0],
                    config["wind_profile_magnitude_range"][1],
                )
                is_valid = (
                    new_point
                    < wind_profile[-1]
                    + config["wind_profile_acceptable_transition_threshold"]
                    and new_point
                    > wind_profile[-1]
                    - config["wind_profile_acceptable_transition_threshold"]
                )
            wind_profile.append(new_point)

        seed_sample = seed + i
        # Generate wind noise
        wn = wng(
            fs=sample_rate,
            duration=config["duration"],
            generate=True,
            wind_profile=wind_profile,
            gustiness=gustiness,
            start_seed=seed_sample,
        )
        wn_signal, wind_profile = wn.generate_wind_noise()

        # Save signal in .wav file
        output_path = output_dir / f"wind_noise_{i}.wav"
        wn.save_signal(
            wn_signal,
            filename=output_path,
            num_ch=1,
            fs=sample_rate,
        )

        scp.write(
            f"wind_noise_{sample_rate}hz_{i} {sample_rate} {str(output_path.resolve())}\n"
        )

scp.close()
