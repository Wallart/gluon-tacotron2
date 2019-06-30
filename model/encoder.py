from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon import rnn
from model.layers.conv_norm import ConvNorm
from params import tacotron_params


class Encoder(gluon.HybridBlock):

    def __init__(self, opts, *args, **kwargs):
        super(Encoder, self).__init__(*args, **kwargs)

        self._num_symbols = opts.n_symbols
        self._symbols_embedding_dim = 512

        params = tacotron_params
        self._embedding_dim = params.encoder_embedding_dim
        self._hidden_size = self._embedding_dim // 2
        #self._lstm_units = 512

        with self.name_scope():
            self._embed = nn.Embedding(self._num_symbols, self._symbols_embedding_dim)

            self._stages = nn.HybridSequential()
            for _ in range(0, 3):
                self._stages.add(ConvNorm(self._embedding_dim, self._embedding_dim, kernel_size=5, init_gain='relu'))
                self._stages.add(nn.BatchNorm())
                self._stages.add(nn.Activation('relu'))
                self._stages.add(nn.Dropout(.5))

            self._lstm = rnn.LSTM(self._hidden_size, input_size=self._embedding_dim, bidirectional=True, layout='NTC')

    def hybrid_forward(self, F, x, *args, **kwargs):
        # input (batch_size, max_seq_length)
        x = self._embed(x)
        # (batch_size, max_seq_length, symbols_embedding_dim)
        x = x.transpose((0, 2, 1))
        # (batch_size, symbols_embedding_dim, max_seq_length)
        x = self._stages(x)
        # (batch_size, max_seq_length, symbols_embedding_dim)
        x = x.transpose((0, 2, 1))
        # (batch_size, symbols_embedding_dim, max_seq_length)
        #x = x.transpose((2, 0, 1))
        # Because LSTM layout is TNC
        # T, N and C stand for sequence length, batch size, and feature dimensions respectively.
        # (max_seq_length, batch_size, symbols_embedding_dim)
        x = self._lstm(x)
        # bidirectional is True, output shape will be (max_seq_length, batch_size, 2*hidden_size)
        # TODO Find out why we have to sum the bidirectionnal inputs
        #x = F.broadcast_add(F.slice_axis(x, axis=2, begin=0, end=256), F.slice_axis(x, axis=2, begin=256, end=None))
        # (max_seq_length, batch_size, 2 * hidden_size)
        return x
