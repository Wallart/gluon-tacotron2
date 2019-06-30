from mxnet import gluon
from mxnet.gluon import nn


class ConvBlock(gluon.HybridBlock):

    def __init__(self, filters=512, kernel_size=5, strides=1, padding=None, *args, **kwargs):
        super(ConvBlock, self).__init__(*args, **kwargs)

        if padding is None:
            padding = (kernel_size - 1) // 2

        with self.name_scope():
            self._stages = nn.HybridSequential()
            self._stages.add(nn.Conv1D(channels=filters, kernel_size=kernel_size, strides=strides, padding=padding))
            self._stages.add(nn.BatchNorm())
            self._stages.add(nn.Activation('relu'))

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self._stages(x)
