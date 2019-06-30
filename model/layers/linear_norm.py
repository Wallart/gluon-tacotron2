from mxnet import gluon
from mxnet.gluon import nn
from initializer.custom_xavier import CustomXavier


class LinearNorm(gluon.HybridBlock):
    def __init__(self, in_dim, out_dim, weight_init_gain, bias=True):
        super(LinearNorm, self).__init__()
        self._init = {
            'weight_initializer': CustomXavier(weight_init_gain),
        }
        self._in_dim = in_dim
        self._linear_layer = nn.Dense(in_units=in_dim, units=out_dim, use_bias=bias, **self._init)

    def hybrid_forward(self, F, x, *args, **kwargs):
        # N.B. nn.Dense only supports tensors of rank 2
        return self._linear_layer(x)
