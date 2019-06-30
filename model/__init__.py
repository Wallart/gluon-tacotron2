from mxnet import gluon, MXNetError

import os
import logging


def get_model(args, ctx, model_class, model_path=None, symbol_path=None):
    nn = None
    if model_path is not None:
        model_path = os.path.expanduser(model_path)

        try:
            symbol_path = symbol_path or get_symbol(model_path)
            nn = gluon.nn.SymbolBlock.imports(symbol_path, ['data'], model_path)
            nn.collect_params().reset_ctx(ctx=ctx)
        except Exception as e:
            logging.info('{}. Trying to load model as a checkpoint'.format(e))

        if nn is None:
            try:
                nn = model_class(args, ctx)
                nn.load_parameters(model_path, ctx=ctx)
            except MXNetError as _:
                logging.error('Cannot load model. Invalid file.')
                exit(1)
    else:
        nn = model_class(args)
        nn.initialize(ctx=ctx)

    if not args.no_hybridize:
        nn.hybridize()

    return nn


def get_symbol(weights_path):
    weights_file = os.path.basename(weights_path)
    symbol_file = '{}-symbol.json'.format(weights_file.split('-')[0])
    symbol_path = os.path.join(os.path.dirname(weights_path), symbol_file)
    if not os.path.isfile(symbol_path):
        raise Exception('Cannot find symbol file')

    return symbol_path

