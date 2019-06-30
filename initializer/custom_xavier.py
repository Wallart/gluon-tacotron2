from mxnet import nd
from mxnet.initializer import Xavier, register

import math


def calculate_gain(nonlinearity, param=None):
    r"""Return the recommended gain value for the given nonlinearity function.
    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    ================= ====================================================

    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function

    Examples:
        >>> gain = nn.init.calculate_gain('leaky_relu')
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError('negative_slope {} not a valid number'.format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError('Unsupported nonlinearity {}'.format(nonlinearity))


@register
class CustomXavier(Xavier):
    def __init__(self, gain):
        super(CustomXavier, self).__init__(rnd_type='uniform', factor_type='out', magnitude=3)
        self._gain = nd.array(calculate_gain(gain))

    def _init_weight(self, _, arr):
        super()._init_weight(_, arr)
        nd.elemwise_mul(arr, self._gain, out=arr)

