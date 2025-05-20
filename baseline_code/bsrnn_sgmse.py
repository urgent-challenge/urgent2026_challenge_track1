from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.decoder.stft_decoder import STFTDecoder

from espnet2.enh.separator.bsrnn_separator import BSRNNSeparator
import torch
from espnet2.enh.diffusion.sdes import OUVESDE, OUVPSDE, SDE
from espnet2.enh.diffusion.score_based_diffusion import ScoreModel
from espnet2.enh.diffusion.abs_diffusion import AbsDiffusion
from espnet2.enh.layers.bsrnn import choose_norm, BandSplit, MaskDecoder, choose_norm1d

import torch.nn as nn



class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * torch.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)



class GradDecoder(nn.Module):
    def __init__(self, freq_dim, subbands, channels=128, num_spk=1, norm_type="GN"):
        super().__init__()
        assert freq_dim == sum(subbands), (freq_dim, subbands)
        self.subbands = subbands
        self.freq_dim = freq_dim
        self.num_spk = num_spk
        assert num_spk == 1
        self.mlp_mask = nn.ModuleList()
        self.mlp_residual = nn.ModuleList()
        self.conv_after_mask = torch.nn.Sequential(nn.Conv2d(channels, 4, 5, 1, 2), 
                                                   nn.GLU(dim=1),)
        self.conv_after_residual = torch.nn.Sequential(nn.Conv2d(channels, 4, 5, 1, 2), 
                                                   nn.GLU(dim=1),)
        for subband in self.subbands:
            self.mlp_mask.append(
                nn.Sequential(
                    choose_norm1d(norm_type, channels),
                    nn.Conv1d(channels, subband * channels, 1),
                    nn.Tanh(),
                )
            )
            self.mlp_residual.append(
                nn.Sequential(
                    choose_norm1d(norm_type, channels),
                    nn.Conv1d(channels, subband * channels, 1),
                    nn.Tanh(),
                    # nn.Conv1d(4 * channels, int(subband * 4 channels), 1),
                    # nn.GLU(dim=1),
                )
            )

    def forward(self, x):
        """MaskDecoder forward.

        Args:
            x (torch.Tensor): input tensor of shape (B, N, T, K)
        Returns:
            m (torch.Tensor): output mask of shape (B, num_spk, T, F, 2)
            r (torch.Tensor): output residual of shape (B, num_spk, T, F, 2)
        """
        B, N, T, K = x.shape
        for i in range(len(self.subbands)):
            sub = self.subbands[i]
            if i >= x.size(-1):
                break
            x_band = x[:, :, :, i]
            out = self.mlp_mask[i](x_band).view(B,N, sub, T)
            if i == 0:
                m = out
            else:
                m = torch.cat((m, out), dim=2)

            res = self.mlp_residual[i](x_band).view(B,N, sub, T)
            if i == 0:
                r = res
            else:
                r = torch.cat((r, res), dim=2)

        m = self.conv_after_mask(m)
        r = self.conv_after_residual(r)
        # Pad zeros in addition to effective subbands to cover the full frequency range
        m = nn.functional.pad(m, (0, 0, 0, int(self.freq_dim - m.size(-2))))
        r = nn.functional.pad(r, (0, 0, 0, int(self.freq_dim - r.size(-2))))
        return m.moveaxis(1, 3).contiguous(), r.moveaxis(1, 3).contiguous()


class BSRNN(nn.Module):
    # ported from https://github.com/sungwon23/BSRNN
    def __init__(
        self,
        input_dim=481,
        num_channel=16,
        num_layer=6,
        target_fs=48000,
        causal=True,
        num_spk=1,
        norm_type="GN",
    ):
        """Band-Split RNN (BSRNN).

        References:
            [1] J. Yu, H. Chen, Y. Luo, R. Gu, and C. Weng, “High fidelity speech
            enhancement with band-split RNN,” in Proc. ISCA Interspeech, 2023.
            https://isca-speech.org/archive/interspeech_2023/yu23b_interspeech.html
            [2] J. Yu, and Y. Luo, “Efficient monaural speech enhancement with
            universal sample rate band-split RNN,” in Proc. ICASSP, 2023.
            https://ieeexplore.ieee.org/document/10096020

        Args:
            input_dim (int): maximum number of frequency bins corresponding to
                `target_fs`
            num_channel (int): embedding dimension of each time-frequency bin
            num_layer (int): number of time and frequency RNN layers
            target_fs (int): maximum sampling frequency supported by the model
            causal (bool): Whether or not to adopt causal processing
                if True, LSTM will be used instead of BLSTM for time modeling
            num_spk (int): number of outputs to be generated
            norm_type (str): type of normalization layer (cfLN / cLN / BN / GN)
        """
        super().__init__()
        norm1d_type = norm_type if norm_type != "cfLN" else "cLN"
        self.num_layer = num_layer
        self.band_split_y = BandSplit(
            input_dim, target_fs=target_fs, channels=num_channel, norm_type=norm1d_type
        )
        self.band_split_x = BandSplit(
            input_dim, target_fs=target_fs, channels=num_channel, norm_type=norm1d_type
        )
        self.condition_fc = nn.Linear(in_features=2*num_channel, out_features=num_channel)

        self.target_fs = target_fs
        self.causal = causal
        self.num_spk = num_spk

        self.norm_time = nn.ModuleList()
        self.rnn_time = nn.ModuleList()
        self.fc_time = nn.ModuleList()
        self.norm_freq = nn.ModuleList()
        self.rnn_freq = nn.ModuleList()
        self.fc_freq = nn.ModuleList()
        self.t_cond = nn.ModuleList()
        hdim = 2 * num_channel
        for i in range(self.num_layer):
            self.t_cond.append(GaussianFourierProjection(num_channel//2, scale=1))
            self.norm_time.append(choose_norm(norm_type, num_channel))
            self.rnn_time.append(
                nn.LSTM(
                    num_channel,
                    hdim,
                    batch_first=True,
                    bidirectional=not causal,
                )
            )
            self.fc_time.append(nn.Linear(hdim if causal else hdim * 2, num_channel))
            self.norm_freq.append(choose_norm(norm_type, num_channel))
            self.rnn_freq.append(
                nn.LSTM(num_channel, hdim, batch_first=True, bidirectional=True)
            )
            self.fc_freq.append(nn.Linear(4 * num_channel, num_channel))

        self.grad_decoder = GradDecoder(
            input_dim,
            self.band_split_x.subbands,
            channels=num_channel,
            num_spk=1,
            norm_type=norm1d_type,
        )

        self.current_fs = None

    def forward(self, dnn_input, t=None, fs=None):
       
        """BSRNN forward.
        dnn_input: B, 2, F, T (complex)
        Args:
            x (torch.Tensor): input tensor of shape (B, T, F, 2)
            fs (int, optional): sampling rate of the input signal.
                if not None, the input signal will be truncated to only process the
                effective frequency subbands.
                if None, the input signal is assumed to be already truncated to only
                contain effective frequency subbands.
        Returns:
            out (torch.Tensor): output tensor of shape (B, num_spk, T, F, 2)
        """

        if fs is None:
            fs = self.current_fs

        x = dnn_input[:, 0, :, : ].permute(0, 2, 1) #, B, T, F
        y = dnn_input[:, 1, :, : ].permute(0, 2, 1) # B, T, F

        x = torch.stack([x.real, x.imag], dim=-1)
        y = torch.stack([y.real, y.imag], dim=-1)

        # B, T, F, 2
        assert t is not None
        xx = self.band_split_x(x, fs=fs)
        yy = self.band_split_y(y, fs=fs)

        zz = torch.cat([xx, yy], dim=1).permute(0, 2, 3, 1) # B, T, K, 2N
        z = self.condition_fc(zz).permute(0, 3, 1, 2)
        B, N, T, K = z.shape
        skip = z
        for i in range(self.num_layer):
            

            out = self.norm_time[i](skip)

            t_emb = self.t_cond[i](t)
            out = out + t_emb[..., None, None]

            out = out.transpose(1, 3).reshape(B * K, T, N)
            out, _ = self.rnn_time[i](out)
            out = self.fc_time[i](out)
            out = out.reshape(B, K, T, N).transpose(1, 3)
            skip = skip + out

            out = self.norm_freq[i](skip)
            out = out.permute(0, 2, 3, 1).contiguous().reshape(B * T, K, N)
            out, _ = self.rnn_freq[i](out)
            out = self.fc_freq[i](out)
            out = out.reshape(B, T, K, N).permute(0, 3, 1, 2).contiguous()
            skip = skip + out

        m, r  = self.grad_decoder(skip)

        x_t = dnn_input[:, 0, :, :]
        B, F, T = x_t.shape
        m = torch.view_as_complex(m)[:, 0:F, :]
        r = torch.view_as_complex(r)[:, 0:F, :]
        g = m*x_t + r
        g = g.unsqueeze(1)

        return g




class BSRNNScoreModel(ScoreModel):

    def __init__(self, input_dim, cfg):
        super(AbsDiffusion, self).__init__()

        self.dnn = BSRNN(input_dim=input_dim, 
                         num_spk=1,
                         num_layer=6,
                         target_fs=48000,
                         causal=False,
                         num_channel=cfg.bsrnn_hidden if hasattr(cfg, 'bsrnn_hidden') else 196)
        self.sde = OUVESDE(
            sigma_min=0.05,
            sigma_max=0.5,
            N=1000,
        )
        self.loss_type = 'mse'
        self.t_eps = 3e-2


        

class SGMSE_BSRNN(torch.nn.Module):
    def __init__(self, cfg):

        super().__init__()

        self.encoder = STFTEncoder(
                    n_fft=960,
                    hop_length=480,
                    use_builtin_complex=True,
                    default_fs=48000,
                    spec_transform_type='exponent',
                )
        self.decoder = STFTDecoder(
            n_fft=960,
            hop_length=480,
            default_fs=48000,
            spec_transform_type='exponent',
        )

        self.diffusion = BSRNNScoreModel(
            input_dim=self.encoder.output_dim, cfg=cfg,
        )



    def forward(self, noisy_speech, clean_speech, speech_length, fs):
        
        feature_mix, flens = self.encoder(noisy_speech, speech_length, fs=fs)
        feature_ref, flens = self.encoder(clean_speech, speech_length, fs=fs)

        self.diffusion.dnn.current_fs = fs
        loss = self.diffusion(
            feature_ref=feature_ref, feature_mix=feature_mix
        )

        return loss


    def enhance(self, noisy_speech, length, sr):

        feats, flens = self.encoder(noisy_speech, length, sr)

        self.diffusion.dnn.current_fs = sr

        enhanced_spec = self.diffusion.enhance(feats,
                                               snr=0.3,
                                               N=50)

        enhanced_speech, ilens = self.decoder(enhanced_spec, length, sr)

        return enhanced_speech
        pass




        


if __name__ == "__main__":


    input = torch.rand((3, 48000))
    speech_length = torch.tensor([48000]*3)
    fs = 48000

    model = BSRNN()
    out = model.forward(input, speech_length, fs)
    print(out)


