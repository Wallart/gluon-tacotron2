from mxnet import nd
from mxnet.gluon import data
from dataset.wav_dataset import WavDataset
from dataset.preprocessing.text import text_to_sequence

import numpy as np


def pad(seq, max_len):
    pad_size = max_len - seq.shape[-1]
    new_shape = [1] * (4 - len(seq.shape)) + list(seq.shape)
    seq = seq.reshape(new_shape).astype(np.float32)
    seq = seq.pad(mode='constant', constant_value=0, pad_width=(0, 0, 0, 0, 0, 0, 0, pad_size))
    return seq.squeeze()

# def pad1d(seq, max_len):
#     seq = seq.asnumpy()
#     return nd.array(np.pad(seq, (0, max_len - len(seq)), mode='constant'))
#
#
# def pad2d(seq, max_len, dim=80, pad_value=0.0):
#     seq = seq.asnumpy()
#     padded = np.zeros((max_len, dim)) + pad_value
#     padded[:len(seq), :] = seq
#     return padded


def collate_fn(batch):
    mel_spec = [item[0] for item in batch]
    mel_spec_lengths = [x.shape[1] for x in mel_spec]
    text = [item[1] for item in batch]
    text_lengths = [x.shape[0] for x in text]

    max_text = max(text_lengths)
    max_mel_spec = max([x.shape[1] for x in mel_spec])

    mel_batch = nd.stack(*[pad(x, max_mel_spec) for x in mel_spec])
    text_batch = nd.stack(*[pad(x, max_text) for x in text])

    gates = []
    for m_len in mel_spec_lengths:
        gate = np.zeros(max_mel_spec)
        gate[:m_len-1] = 1
        gates.append(nd.array(gate))
    gates = nd.stack(*gates)

    return (mel_batch, nd.array(mel_spec_lengths)), gates, (text_batch, nd.array(text_lengths))


def get_dataset(path, batch_size):
    params = {
        'max_wav_value': 32768.0,  # for 16 bits files
        'sampling_rate': 22050,
        'filter_length': 1024,
        'hop_length': 256,
        'win_length': 1024,
        'n_mel_channels': 80,
        'mel_fmin': 0.0,
        'mel_fmax': 8000.0
    }
    dataset = WavDataset(path, text_to_sequence, **params)
    dataloader = data.DataLoader(dataset, batchify_fn=collate_fn, batch_size=batch_size, shuffle=True, last_batch='discard')
    return dataloader

