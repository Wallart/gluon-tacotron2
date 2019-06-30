"""
BSD 3-Clause License

Copyright (c) 2017, Prem Seetharaman
All rights reserved.

* Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from mxnet import nd, gluon
from scipy.signal import get_window
from librosa.util import pad_center, tiny
from torch.autograd import Variable
from dataset.preprocessing.audio_preprocessing import window_sumsquare

import torch
import numpy as np
import torch.nn.functional as F


# TODO Refactor
class STFT(gluon.HybridBlock):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
    def __init__(self, filter_length=800, hop_length=200, win_length=800, window='hann'):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])])

        self.forward_basis = nd.array(fourier_basis[:, np.newaxis, :])
        self.inverse_basis = nd.array(np.linalg.pinv(scale * fourier_basis).T[:, np.newaxis, :])

        if window is not None:
            assert(win_length >= filter_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = nd.array(fft_window)

            # window the bases
            self.forward_basis *= fft_window
            self.inverse_basis *= fft_window

        #self.register_buffer('forward_basis', forward_basis.float())
        #self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.shape[0]
        num_samples = input_data.shape[1]

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.reshape(num_batches, 1, num_samples).expand_dims(1)
        input_data = nd.pad(input_data, 'reflect', (0, 0, 0, 0, 0, 0, self.filter_length // 2, self.filter_length // 2))
        input_data = input_data.squeeze(axis=1)

        forward_transform = nd.Convolution(input_data, self.forward_basis, no_bias=True, kernel=self.forward_basis.shape[2], num_filter=self.forward_basis.shape[0], stride=self.hop_length, pad=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = nd.sqrt(real_part**2 + imag_part**2)
        phase = nd.array(np.arctan2(imag_part.asnumpy(), real_part.asnumpy()))
        #phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat([magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window, magnitude.size(-1), hop_length=self.hop_length,
                win_length=self.win_length, n_fft=self.filter_length,
                dtype=np.float32)
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False)
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
        inverse_transform = inverse_transform[:, :, :-int(self.filter_length/2):]

        return inverse_transform

    def hybrid_forward(self, F, x, *args, **kwargs):
        self.magnitude, self.phase = self.transform(x)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction
