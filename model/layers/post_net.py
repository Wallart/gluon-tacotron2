from mxnet import gluon


class PostNet(gluon.HybridBlock):

    def __init__(self, *args, **kwargs):
        super(PostNet, self).__init__(*args, **kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        pass
