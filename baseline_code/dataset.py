import torch
import glob
import itertools
import soundfile
import torchaudio
import lightning
import os
from collections import defaultdict
import numpy as np
import torchaudio
from simulation.generate_data_param import process_one_sample as get_simu_meta
from simulation.simulate_data_from_param import process_one_sample, save_audio, read_audio
import copy
import random
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from baseline_code.config import Config


class SimulationConfigs:
    snr_low_bound = -5.0
    snr_high_bound = 20.0
    reuse_noise = True
    prob_wind_noise = 0.05
    wind_noise_config = dict(
        threshold=[0.1, 0.3],
        ratio=[1, 20],
        attack=[5, 100],
        release=[5, 100],
        sc_gain=[0.8, 1.2],
        clipping_threshold=[0.85, 1.0],
        clipping_chance=0.75,
        wind_noise_snr_low_bound=-10.0,
        wind_noise_snr_high_bound=15.0,
    )
    prob_reverberation = 0.5
    reuse_rir = True

    augmentations_name = [
        'bandwidth_limitation',
        'clipping',
        'codec',
        'packet_loss'
    ]

    num_augmentations = {
        0: 0.25,
        1: 0.40,
        2: 0.20,
        3: 0.15,
    }

    augmentations = dict(
        bandwidth_limitation=dict(
            weight=1.0,
            resample_methods='random',
        ),
        clipping=dict(
            weight=1.0,
            clipping_min_quantile=[0.0, 0.1],
            clipping_max_quantile=[0.9, 1.0],
        ),
        codec=dict(
            weight=1.0,
            config=[
                dict(format='mp3', encoder=None, qscale=[1, 10]),
                dict(format='ogg', encoder=['vorbis'], qscale=[-1, 10]),
            ]
        ),
        packet_loss=dict(
            weight=1.0,
            packet_duration_ms=20,
            max_continuous_packet_loss=10,
            packet_loss_rate=[0.05, 0.25],
        )
    )


def read_kv_scp(scp):
    rtv = {}
    with open(scp, "r") as f:
        for line in f:
            uid, value = line.strip().split()
            assert uid not in rtv, (uid)
            rtv[uid] = value
    return rtv


def read_source_scp(scp):
    source_dict = defaultdict(dict)
    source_dict_flatten = {}
    with open(scp, "r") as f:
        for line in f:
            uid, fs, audio_path = line.strip().split()
            assert uid not in source_dict[int(fs)], (uid, fs)
            source_dict[int(fs)][uid] = audio_path
            source_dict_flatten[uid] = audio_path

    source_uids = {k: list(source_dict[k].keys()) for k in source_dict}

    return source_dict, source_uids, source_dict_flatten


class PreSimulatedDataset(torch.utils.data.Dataset):
    def __init__(self, clean_speech, noisy_speech, utt2fs, speech_length, max_duration=-1):

        self.clean_speech = read_kv_scp(clean_speech)
        self.noisy_speech = read_kv_scp(noisy_speech)
        self.utt2fs = {k: int(v) for k, v in read_kv_scp(utt2fs).items()}
        self.speech_length = {k: int(v)
                              for k, v in read_kv_scp(speech_length).items()}

        self.uid = list(self.clean_speech.keys())
        self.max_duration = max_duration

        assert len(self.clean_speech) == len(self.noisy_speech)
        assert len(self.clean_speech) == len(self.utt2fs)
        assert len(self.clean_speech) == len(self.speech_length)

    def get_source_length(self):
        if self.max_duration > 0:
            return [min(self.speech_length[k], self.max_duration) for k in self.uid]   
        else:
            return [self.speech_length[k] for k in self.uid]

    def get_srs(self):
        return [self.utt2fs[k] for k in self.uid]

    def __len__(self):

        return len(self.clean_speech)

    def __getitem__(self, index):

        uid = self.uid[index]
        audio, fs = read_audio(self.clean_speech[uid], max_duration=self.max_duration)

        assert fs == self.utt2fs[uid]

        noisy, fs = read_audio(self.noisy_speech[uid], max_duration=self.max_duration)
        assert fs == self.utt2fs[uid]

        speech_length = audio.shape[1]

        return audio, noisy, fs, speech_length


class DynamicMixingDataset(torch.utils.data.Dataset):
    def __init__(self, speech_source_scp, noise_source_scp, rir_scp, windnoise_scp, speech_length_file, use_high_pass=True, retry_when_fails=False, max_duration=240000):
        super().__init__()

        self.speech_source, self.speech_uids, self.speech_source_flt = read_source_scp(
            speech_source_scp)
        self.noise_source, self.noise_uids, self.noise_source_flt = read_source_scp(
            noise_source_scp)
        self.rirs, self.rir_uids, self.rirs_flt = read_source_scp(rir_scp)
        self.wind_noises, self.wind_noises_uids, self.wind_noises_flt = read_source_scp(
            windnoise_scp)

        self.all_noise_flt = copy.deepcopy(self.noise_source_flt)
        self.all_noise_flt.update(self.wind_noises_flt)

        self.source_length = {k: min(int(v), max_duration)
                              for k, v in read_kv_scp(speech_length_file).items()}

        self.max_duration = max_duration

        self.length = sum([len(self.speech_source[k])
                          for k in self.speech_source])

        self.samplerates = list(self.speech_source.keys())
        self.fs_sub_lengths = [len(self.speech_source[k])
                               for k in self.samplerates]
        self.accum_lengths = [sum(self.fs_sub_lengths[0:i+1])
                              for i in range(len(self.fs_sub_lengths))]

        self.augmentations = list(SimulationConfigs.augmentations.keys())
        weight_augmentations = np.array(
            [v["weight"] for v in SimulationConfigs.augmentations.values()])
        self.weight_augmentations = weight_augmentations / \
            np.sum(weight_augmentations)
        self.use_high_pass = use_high_pass
        self.retry_when_fails = retry_when_fails

    def get_srs(self,):

        srs = []
        for i in range(len(self)):
            srs.append(self._get_from_index(i)[0])
        return srs

    def get_source_length(self,):

        length = []
        for i in range(len(self)):
            fs, real_idx = self._get_from_index(i)
            uid = self.speech_uids[fs][real_idx]
            length.append(self.source_length[uid])

        return length

    def __len__(self):

        return self.length

    def simulation(self, ):
        pass

    def _get_from_index(self, index):

        real_idx = -1
        speech_fs = -1
        previous = 0

        for i, fs in enumerate(self.samplerates):
            if index >= previous and index < self.accum_lengths[i]:
                speech_fs = fs
                real_idx = index - previous
                break
            previous = self.accum_lengths[i]

        assert real_idx >= 0 and speech_fs > 0

        return speech_fs, real_idx

    def run_simulation(self, speech_uid, speech_length, sr):

        use_wind_noise = np.random.random() < SimulationConfigs.prob_wind_noise
        num_aug = np.random.choice(
            list(SimulationConfigs.num_augmentations.keys()),
            p=list(SimulationConfigs.num_augmentations.values()),
        )

        if num_aug == 0:
            aug = "none"
        else:
            aug = np.random.choice(
                self.augmentations,
                p=self.weight_augmentations,
                size=num_aug,
                replace=False,
            )
            # As wind-noise simulation include clipping,
            # we exclude clipping from augmentation list
            while use_wind_noise and "clipping" in aug:
                aug = np.random.choice(
                    self.augmentations,
                    p=self.weight_augmentations,
                    size=num_aug,
                    replace=False,
                )

        info = get_simu_meta(
            SimulationConfigs,
            speech_length,
            sr,
            noise_dic=self.noise_source,
            used_noise_dic=None,
            wind_noise_dic=self.wind_noises,
            used_wind_noise_dic=None,
            use_wind_noise=use_wind_noise,
            snr_range=(SimulationConfigs.snr_low_bound,
                       SimulationConfigs.snr_high_bound),
            wind_noise_snr_range=(SimulationConfigs.wind_noise_config['wind_noise_snr_low_bound'],
                                  SimulationConfigs.wind_noise_config['wind_noise_snr_high_bound']
                                  ),
            store_noise=False,
            rir_dic=self.rirs,
            used_rir_dic=None,
            augmentations=aug,
            force_1ch=True,
        )

        info['speech_uid'] = speech_uid
        info['id'] = speech_uid
        info['snr_dB'] = info['snr']

        speech, noisy_speech, fs = process_one_sample(
            info,
            store_noise=False,
            speech_dic=self.speech_source_flt,
            noise_dic=self.all_noise_flt,
            rir_dic=self.rirs_flt,
            highpass=self.use_high_pass,
            on_the_fly=True,
            max_duration=self.max_duration,

        )

        return speech, noisy_speech, fs

    def __getitem__(self, index):

        speech_fs, real_idx = self._get_from_index(index)

        speech_uid = self.speech_uids[speech_fs][real_idx]
        speech_path = self.speech_source[speech_fs][speech_uid]

        # Load speech sample (Channel, Time)
        if speech_path.endswith(".wav"):
            with soundfile.SoundFile(speech_path) as af:
                speech_length = af.frames
        else:
            # Sometimes the acutal loaded audio's length differs from af.frames
            speech_length = soundfile.read(speech_path)[0].shape[0]

        speech_length = min(self.max_duration, speech_length)

        if self.retry_when_fails:
            attempts = 0

            while attempts < 3:
                try:
                    speech, noisy_speech, fs = self.run_simulation(
                        speech_uid, speech_length, speech_fs)
                    return speech, noisy_speech, fs, speech_length
                except:
                    attempts += 1

            # if simulation failed, return clean speech
            speech, fs = soundfile.read(speech_path)
            noisy_speech = speech
            print('Simulation Failed after 3 times try, return clean speech')
            return speech, noisy_speech, fs, speech_length

        else:
            speech, noisy_speech, fs = self.run_simulation(
                speech_uid, speech_length, speech_fs)
            return speech, noisy_speech, fs, speech_length


class GroupedBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, rank, world_size, seed=0, drop_last=False, bucket_size_mult=100, sampler=None):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.bucket_size = batch_size * bucket_size_mult  # 桶大小（可调整）
        self.epoch = 0
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed + rank + self.epoch)

        # 按采样率分组索引
        sr_groups = defaultdict(list)
        for idx, sr in enumerate(dataset.get_srs()):
            sr_groups[sr].append(idx)

        # 对每个采样率组内的样本按长度排序并分桶
        self.buckets = []

        source_length = dataset.get_source_length()
        for sr, indices in sr_groups.items():
            # 按音频长度排序
            sorted_indices = sorted(indices, key=lambda x: source_length[x])
            sorted_indices = sorted_indices[self.rank::self.world_size]

            # 分桶
            for i in range(0, len(sorted_indices), self.bucket_size):
                bucket = sorted_indices[i:i + self.bucket_size]
                self.buckets.append(bucket)

    def set_epoch(self, epoch):
        self.epoch = epoch  # 每次epoch更新时会调用
        self.generator.manual_seed(self.seed + self.rank + self.epoch)

    def __iter__(self):
        # 打乱桶的顺序以增加随机性
        random.seed(self.epoch + self.rank)
        random.shuffle(self.buckets)
        all_batches = []
        for bucket in self.buckets:
            # 打乱桶内样本顺序
            random.shuffle(bucket)
            # 生成批次
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch)
        # 打乱所有批次的顺序
        random.shuffle(all_batches)
        return iter(all_batches)
    
    def state_dict(self):
        return {'seed': self.seed, 'epoch': self.epoch}

    def __len__(self):
        total = 0
        for bucket in self.buckets:
            num_samples = len(bucket)
            if self.drop_last:
                total += num_samples // self.batch_size
            else:
                total += (num_samples + self.batch_size - 1) // self.batch_size
        return total


def collate_fn(batch):
    """
    处理不同长度的音频，动态进行右侧补零padding
    输入格式：batch = [(audio_1, sr_1), (audio_2, sr_2), ...]
    其中 audio_i 形状为 (1, T_i)
    """
    # 分离音频和采样率
    speechs = [torch.tensor(item[0]) for item in batch]
    noisy_speechs = [torch.tensor(item[1]) for item in batch]
    srs = [item[2] for item in batch]
    lengths = [item[3] for item in batch]

    # 检查批次内采样率一致性（由采样器保证）
    assert all(sr == srs[0] for sr in srs), "同一批次内采样率不一致"
    sr = srs[0]

    # 获取最长音频长度
    max_length = max(audio.shape[1] for audio in speechs)

    # 动态padding并堆叠
    padded_audios = torch.stack([
        torch.nn.functional.pad(
            audio,
            (0, max_length - audio.shape[1]),  # 右侧补零
            value=0.0
        ) for audio in speechs
    ], dim=0)  # 输出形状：(B, 1, T_max)

    padded_noisy_speech = torch.stack([
        torch.nn.functional.pad(
            audio,
            (0, max_length - audio.size(1)),  # 右侧补零
            value=0.0
        ) for audio in noisy_speechs
    ], dim=0)  # 输出形状：(B, 1, T_max)

    # 返回格式：padded音频张量 + 标量采样率
    return padded_audios, padded_noisy_speech, torch.tensor(sr, dtype=torch.int32), torch.tensor(lengths, dtype=torch.int32)


class AudioDataModule(lightning.LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.train_dir = config.train_set_path
        self.valid_dir = config.valid_set_path
        self.num_worker = config.num_worker
        self.batch_size = config.batch_size


        if self.config.train_set_dynamic_mixing:
            self.train_dataset = DynamicMixingDataset(
                speech_source_scp=f'{self.train_dir}/speech_sources_relative.scp',
                noise_source_scp=f'{self.train_dir}/noise_scoures_relative.scp',
                speech_length_file=f'{self.train_dir}/source_length.scp',
                rir_scp=f'{self.train_dir}/rirs_relative.scp',
                windnoise_scp=f'{self.train_dir}/wind_noise_scoures_relative.scp',
                retry_when_fails=True,
                max_duration=config.max_duration,
            )
        else:
            self.train_dataset = PreSimulatedDataset(
                clean_speech=f'{self.train_dir}/spk1.scp',
                noisy_speech=f'{self.train_dir}/wav.scp',
                utt2fs=f'{self.train_dir}/utt2fs',
                speech_length=f'{self.train_dir}/speech_length.scp',
                max_duration=config.max_duration,
            )

        self.val_dataset = PreSimulatedDataset(
            clean_speech=f'{self.valid_dir}/spk1.scp',
            noisy_speech=f'{self.valid_dir}/wav.scp',
            utt2fs=f'{self.valid_dir}/utt2fs',
            speech_length=f'{self.valid_dir}/speech_length.scp'
        )


    def on_train_epoch_start(self):
        """每个epoch开始时更新batch_sampler状态"""
        if self.train_batch_sampler is not None:
            self.train_batch_sampler.set_epoch(self.current_epoch)

    
    def train_dataloader(self):
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        self.train_batch_sampler = GroupedBatchSampler(
            self.train_dataset,
            batch_size=self.batch_size,
            rank=rank,
            world_size=world_size,
            drop_last=True,
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=self.num_worker,
            pin_memory=False,
            persistent_workers=True,
            collate_fn=collate_fn,
        )
    
    def val_dataloader(self):
        # rank = torch.distributed.get_rank()
        # world_size = torch.distributed.get_world_size()
        self.val_batch_sampler = GroupedBatchSampler(
            self.val_dataset,
            batch_size=self.batch_size,
            rank=0,
            world_size=1,
            drop_last=True,
        )
        return DataLoader(
            self.val_dataset,
            batch_sampler=self.val_batch_sampler,
            num_workers=self.num_worker,
            pin_memory=False,
            persistent_workers=True,
            collate_fn=collate_fn,
        )


if __name__ == "__main__":
    import tqdm

    train_set = DynamicMixingDataset(speech_source_scp='data/train_sources/speech_sources_relative.scp',
                                     noise_source_scp='data/train_sources/noise_scoures_relative.scp',
                                     speech_length_file='data/train_sources/source_length.scp',
                                     rir_scp='data/train_sources/rirs_relative.scp', windnoise_scp='data/train_sources/wind_noise_scoures_relative.scp', retry_when_fails=False)

    dev_set = PreSimulatedDataset(
        clean_speech='data/validation/spk1.scp',
        noisy_speech='data/validation/wav.scp',
        utt2fs='data/validation/utt2fs',
        speech_length='data/validation/speech_length.scp'
    )

    dl = AudioDataModule(train_set, batch_size=32).train_dataloader()

    for bs in tqdm.tqdm(dl):

        bs[2]

    # for i in range(100):
    # # for i, bs in tqdm.tqdm(enumerate(train_set)):
    #     speech, noisyspeech, fs, _ = train_set[i]

    #     save_audio(speech, f'/tmp/debug_{i}.wav', fs)
    #     save_audio(noisyspeech, f'/tmp/debug_{i}.noisy.wav', fs)

    #     if i == 10:
    #         break
