from mxnet import nd, gluon
from dataset.preprocessing.audio_preprocessing import dynamic_range_decompression, dynamic_range_compression
from librosa.filters import mel
from model.layers.stft import STFT

import torch


class TacotronSTFT:
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024, n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0, mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()

        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = mel(sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        self.mel_basis = nd.array(mel_basis)
        #self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]
        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert y.min() >= -1
        assert y.max() <= 1

        magnitudes, phases = self.stft_fn.transform(y)

        #mel_output = torch.matmul(torch.from_numpy(self.mel_basis.asnumpy()), torch.from_numpy(magnitudes.asnumpy()))
        mel_output = nd.dot(self.mel_basis, magnitudes.transpose((1, 2, 0))).transpose((2, 0, 1))
        mel_output = self.spectral_normalize(mel_output)

        return mel_output
