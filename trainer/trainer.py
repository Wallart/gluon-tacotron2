from abc import ABC, abstractmethod
from datetime import datetime
from mxnet import nd, gluon, profiler
from mxboard import SummaryWriter

import os
import time
import json
import shutil
import logging
import mxnet as mx


class TrainerException(Exception):
    pass


class Trainer(ABC):

    def __init__(self, opts, ctx):
        self._opts = opts
        self._epochs = opts.epochs
        self._batch_size = opts.batch_size
        self._ctx = ctx

        self._chkpt_interval = opts.chkpt_interval
        self._log_interval = opts.log_interval
        self._weight_interval = opts.weight_interval
        self._profile = opts.profile

        self._epoch_tick = 0
        self._batch_tick = 0

        self._networks = []

        self._overwrite = opts.overwrite
        self._outdir = opts.outdir or os.path.join(os.getcwd(), '{}-{}e-{}'.format(self.model_name(), self._epochs, datetime.now().strftime('%y_%m_%d-%H_%M')))
        self._outdir = os.path.expanduser(self._outdir)
        self._outlogs = os.path.join(self._outdir, 'logs')
        self._outchkpts = os.path.join(self._outdir, 'checkpoints')
        self._outsounds = os.path.join(self._outdir, 'sounds')
        self._prepare_outdir()

        if self._profile:
            self._outprofile = os.path.join(self._outdir, 'profile.json')
            profiler.set_config(profile_all=True, aggregate_stats=True, filename=self._outprofile)

        logging.basicConfig()
        self._logger = logging.getLogger()
        self._logger.setLevel(logging.INFO)

    def _prepare_outdir(self):
        outdir_exists = os.path.isdir(self._outdir)
        if outdir_exists and not self._overwrite:
            raise TrainerException('Output directory already exists.')
        elif os.path.isdir(self._outdir) and self._overwrite:
            shutil.rmtree(self._outdir)

        os.makedirs(self._outlogs)
        os.makedirs(self._outchkpts)
        os.makedirs(self._outsounds)

        params_dump = os.path.join(self._outdir, 'parameters_dump.json')
        with open(params_dump, 'w') as f:
            json.dump(vars(self._opts), f, indent=4,  skipkeys=True)

    def b_tick(self):
        self._batch_tick = time.time()

    def e_tick(self):
        self._epoch_tick = time.time()

    def b_ellapsed(self):
        return time.time() - self._batch_tick

    def e_ellapsed(self):
        return time.time() - self._epoch_tick

    def _log(self, message, level=logging.INFO):
        self._logger.log(level, message)

    def _save_profile(self):
        if self._profile:
            print(profiler.dumps())
            profiler.dump()

    @abstractmethod
    def train(self, train_data):
        pass

    @abstractmethod
    def model_name(self):
        pass

    @abstractmethod
    def _export_model(self, num_epoch):
        pass

    @abstractmethod
    def _do_checkpoint(self, cur_epoch):
        pass

    def _visualize_graphs(self, cur_epoch, cur_iter):
        if not self._opts.no_hybridize and cur_epoch == 0 and cur_iter == 1:
            for net, out_path in self._networks:
                with SummaryWriter(logdir=out_path, flush_secs=5, verbose=False) as writer:
                    writer.add_graph(net)

    def _visualize_weights(self, writer, cur_epoch, cur_iter):
        if not self._opts.no_hybridize and self._weight_interval > 0 and cur_iter == 0 and cur_epoch % self._weight_interval == 0:
            for net, _ in self._networks:
                # to visualize gradients each x epochs
                params = [p for p in net.collect_params().values() if type(p) == gluon.Parameter and p._grad]
                for p in params:
                    name = '{}/{}/{}'.format(net._name, '_'.join(p.name.split('_')[:-1]), p.name.split('_')[-1])
                    aggregated_grads = nd.concat(*[grad.as_in_context(mx.cpu()) for grad in p._grad], dim=0)
                    writer.add_histogram(tag=name, values=aggregated_grads, global_step=cur_epoch + 1, bins=1000)
