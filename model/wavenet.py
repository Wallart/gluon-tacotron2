from mxnet import gluon


class WaveNet(gluon.HybridBlock):

    def __init__(self, *args, **kwargs):
        super(WaveNet, self).__init__(*args, **kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        pass