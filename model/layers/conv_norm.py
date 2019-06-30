from mxnet import gluon
from mxnet.gluon import nn

from initializer.custom_xavier import CustomXavier


class ConvNorm(gluon.HybridBlock):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, dilation=1, bias=True, init_gain='linear'):
        super(ConvNorm, self).__init__()
        self._init = {
            'weight_initializer': CustomXavier(init_gain)
        }

        if padding is None:
            assert kernel_size % 2 == 1
            padding = dilation * (kernel_size - 1) // 2

        self._conv = nn.Conv1D(out_channels, in_channels=in_channels, kernel_size=kernel_size, strides=stride, padding=padding, dilation=dilation, use_bias=bias, **self._init)

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self._conv(x)
