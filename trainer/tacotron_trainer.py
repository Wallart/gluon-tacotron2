from mxnet import nd, gluon
from mxboard import SummaryWriter
from model import get_model
from model.decoder import Decoder
from model.encoder import Encoder
from trainer.trainer import Trainer

import os


class TacotronTrainer(Trainer):

    def __init__(self, opts, ctx):
        super(TacotronTrainer, self).__init__(opts, ctx)
        self._decoder = self._build_decoder(opts)
        self._encoder = self._build_encoder(opts)

    def _build_encoder(self, opts):
        encoder = get_model(opts, self._ctx, Encoder)
        logs = os.path.join(self._outlogs, 'encoder')
        self._networks.append((encoder, logs))
        return encoder

    def _build_decoder(self, opts):
        decoder = get_model(opts, self._ctx, Decoder)
        logs = os.path.join(self._outlogs, 'decoder')
        self._networks.append((decoder, logs))
        return decoder

    def train(self, train_data):
        with SummaryWriter(logdir=self._outdir, flush_secs=5, verbose=False) as writer:
            num_iter = len(train_data._dataset) // self._batch_size

            for epoch in range(self._epochs):
                self.e_tick()

                for i, batch in enumerate(train_data):
                    self.b_tick()

                    self._visualize_graphs(epoch, i)
                    self._visualize_weights(writer, epoch, i)

                    text_batch, text_lengths = batch[1]

                    # split data across gpus
                    data = gluon.utils.split_and_load(batch[0], ctx_list=self._ctx, batch_axis=0)
                    label = gluon.utils.split_and_load(text_batch, ctx_list=self._ctx, batch_axis=0)
                    label_lengths = gluon.utils.split_and_load(text_lengths, ctx_list=self._ctx, batch_axis=0)

                    encoder_outputs = [self._encoder(l) for l in label]
                    decoder_outputs = [self._decoder(mem, mel_spec, lbl_lens) for mem, mel_spec, lbl_lens in zip(encoder_outputs, data, label_lengths)]

                    # per x iter logging
                    if (i + 1) % self._log_interval == 0:
                        b_time = self.b_ellapsed()
                        speed = self._batch_size / b_time
                        iter_stats = 'exec time: {:.2f} second(s) speed: {:.2f} samples/s'.format(b_time, speed)
                        self._log('[Epoch {}] --[{}/{}]-- {}'.format(epoch + 1, i + 1, num_iter, iter_stats))

                global_step = epoch + 1
                self._log('[Epoch {}] exec time: {:.2f}'.format(global_step, self.e_ellapsed()))

        nd.waitall()
        self._save_profile()
        self._export_model(self._epochs)

    def model_name(self):
        return 'SeqToSeq'

    def _export_model(self, num_epoch):
        pass

    def _do_checkpoint(self, cur_epoch):
        pass
