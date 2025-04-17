from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.decoder.stft_decoder import STFTDecoder

from espnet2.enh.separator.bsrnn_separator import BSRNNSeparator
import torch



class BSRNN(torch.nn.Module):

    def __init__(self, ):
        super().__init__()

        self.encoder = STFTEncoder(
            n_fft=960,
            hop_length=480,
            use_builtin_complex=True,
            default_fs=48000,
        )
    
        self.decoder = STFTDecoder(
            n_fft=960,
            hop_length=480,
            default_fs=48000,
        )

        self.bsrnn = BSRNNSeparator(
            input_dim=self.encoder.output_dim,
            num_spk=1,
            num_channels=196,
            num_layers=6,
            target_fs=48000,
            causal=False,
        )

    def forward(self, speech_mix, speech_lengths, fs):
        feature_mix, flens = self.encoder(speech_mix, speech_lengths, fs=fs)
        feature_pre, flens, others = self.bsrnn(feature_mix, flens, None)
        enhanced_feature = feature_pre[0]
        enhanced_wav, _ = self.decoder(enhanced_feature, speech_lengths, fs=fs)
        return enhanced_wav, enhanced_feature
        
        


if __name__ == "__main__":


    input = torch.rand((3, 48000))
    speech_length = torch.tensor([48000]*3)
    fs = 48000

    model = BSRNN()
    out = model.forward(input, speech_length, fs)
    print(out)


