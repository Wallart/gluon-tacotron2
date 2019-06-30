from mxnet import gluon
from mxnet.gluon import nn
from model.layers.linear_norm import LinearNorm


class PreNet(gluon.HybridBlock):

    def __init__(self, in_dim, sizes):
        super(PreNet, self).__init__()
        self._in_sizes = [in_dim] + sizes[:-1]

        with self.name_scope():
            self._stages = nn.HybridSequential()
            for in_size, out_size in zip(self._in_sizes, sizes):
                self._stages.add(LinearNorm(in_size, out_size, 'linear', bias=False))
                self._stages.add(nn.Activation('relu'))
                self._stages.add(nn.Dropout(.5))

    def hybrid_forward(self, F, x, *args, **kwargs):
        # TODO Make it hybridize ready
        batch = x.shape[1]
        x = x.reshape(-1, self._in_sizes[0])
        x = self._stages(x)
        return x.reshape((-1, batch, self._in_sizes[1]))

