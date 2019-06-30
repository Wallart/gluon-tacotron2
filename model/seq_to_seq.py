from mxnet import gluon


class SeqToSeq(gluon.HybridBlock):

    def __init__(self, *args, **kwargs):
        super(SeqToSeq, self).__init__(*args, **kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        pass
