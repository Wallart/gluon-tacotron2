import logging
import argparse
import mxnet as mx

from dataset import get_dataset
from trainer.tacotron_trainer import TacotronTrainer


def get_ctx(args):
    try:
        devices_id = [int(i) for i in args.gpus.split(',') if i.strip()]
        if len(devices_id) == 0:
            devices_id = mx.test_utils.list_gpus()

        ctx = [mx.gpu(i) for i in devices_id]
        ctx = ctx if len(ctx) > 0 else [mx.cpu()]
    except Exception as err:
        logging.error('Cannot access GPU, fallback to CPU')
        ctx = [mx.cpu()]

    return ctx


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='Tacotron 2')
    sub_parsers = parser.add_subparsers(dest='action')

    train_parser = sub_parsers.add_parser('train')
    train_parser.add_argument('dataset', type=str, help='training dataset path')
    train_parser.add_argument('-b', '--batch', dest='batch_size', type=int, default=32, help='batch size')
    train_parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=1000, help='learning epochs')
    train_parser.add_argument('-o', '--output', dest='outdir', type=str, help='model output directory')
    train_parser.add_argument('-s', '--symbols', dest='n_symbols', type=int, default=23, help='???')
    train_parser.add_argument('--gpus', dest='gpus', type=str, default='', help='gpus id to use, for example 0,1')
    train_parser.add_argument('--log-interval', dest='log_interval', type=int, default=5, help='iterations log interval')
    train_parser.add_argument('--chkpt-interval', dest='chkpt_interval', type=int,  default=5, help='model checkpointing interval (epochs)')
    train_parser.add_argument('--weight-interval', dest='weight_interval', type=int, default=2, help='model weights visualization interval (epochs)')
    train_parser.add_argument('--profile', action='store_true', help='enable profiling')
    train_parser.add_argument('--overwrite', action='store_true', help='overwrite model if output directory already exists')
    train_parser.add_argument('--no-hybridize', dest='no_hybridize', action='store_true', help='disable mxnet hybridize network (debug purpose)')

    args = parser.parse_args()
    if args.action == 'train':
        ctx = get_ctx(args)

        dataset = get_dataset(args.dataset, args.batch_size)

        trainer = TacotronTrainer(args, ctx)
        trainer.train(dataset)
