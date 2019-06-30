from mxnet import gluon
from model.layers.conv_norm import ConvNorm
from model.layers.linear_norm import LinearNorm


class Location(gluon.HybridBlock):

    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        super(Location, self).__init__()
        padding = (attention_kernel_size - 1) // 2

        with self.name_scope():
            self.location_conv = ConvNorm(2, attention_n_filters, kernel_size=attention_kernel_size, padding=padding, bias=False, stride=1, dilation=1)
            self.location_dense = LinearNorm(attention_n_filters, attention_dim, 'tanh', bias=False)

    def hybrid_forward(self, F, attention_weights_cat, *args, **kwargs):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention
