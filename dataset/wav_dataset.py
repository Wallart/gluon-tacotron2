from mxnet.gluon.data import dataset
from mxnet import nd
from scipy.io.wavfile import read
from dataset.preprocessing.text import text_to_sequence
from model.layers.tacotron_stft import TacotronSTFT
from params import tacotron_params

import os
import glob
import logging
import numpy as np
import matplotlib.pyplot as plt


class WavDataset(dataset.Dataset):

    def __init__(self, root, text_transforms, text_cleaners=['english_cleaners'], *args, **kwargs):
        super(WavDataset, self).__init__()
        self._root = os.path.expanduser(root)
        self._exts = ['.wav', '.txt']

        self._max_wav_value = kwargs['max_wav_value']
        del kwargs['max_wav_value']

        self._stft = TacotronSTFT(**kwargs)
        self._text_cleaners = text_cleaners
        self._text_transforms = text_transforms

        self._items = []
        self._list_records()

    def _list_records(self):
        pattern = os.path.join(self._root, '*{}'.format(self._exts[0]))
        wav_files = glob.glob(pattern)
        logging.info('{} sample(s) found in dataset'.format(len(wav_files)))

        for wav_file in wav_files:
            text_file = '{}{}'.format(os.path.splitext(wav_file)[0], self._exts[1])
            if os.path.isfile(text_file):
                self._items.append((wav_file, self._load_text(text_file)))

    def _load_wav(self, file_path):
        sampling_rate, data = read(file_path)
        return nd.array(data.astype(np.float32))

    def _load_text(self, file_path):
        with open(file_path, encoding='utf-8') as f:
            text = [line.strip() for line in f]
            text = ' '.join(text)

        if self._text_transforms is not None:
            text = self._text_transforms(text, self._text_cleaners)

        return nd.array(text, dtype=np.int)

    def __getitem__(self, idx):
        wav_file, encoded_text = self._items[idx]
        wav = self._load_wav(wav_file)

        audio_normalized = wav / self._max_wav_value
        audio_normalized = audio_normalized.expand_dims(0)
        #audio_normalized = torch.autograd.Variable(audio_norm, requires_grad=False)

        melspec = self._stft.mel_spectrogram(audio_normalized).squeeze()
        return melspec, encoded_text

    def __len__(self):
        return len(self._items)


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    french = WavDataset('~/datasets/tacotron', text_to_sequence, **tacotron_params)
    assert type(french[0]) == tuple

    for i, (data, label) in enumerate(french):
        print('Plotting sample {}'.format(i))

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 4))
        ax.set_title('Mel spectrogram')
        ax.set_xlabel('Frames')
        ax.set_ylabel('Mel channels')
        cax = ax.matshow(data.asnumpy(), interpolation='nearest', aspect='auto', cmap='viridis', origin='lower')
        fig.colorbar(cax)

        plt.show()

