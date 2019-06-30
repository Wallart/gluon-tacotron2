import numpy as np


def get_mask_from_lengths(F, lengths):
    max_len = lengths.max().asscalar()
    ids = F.arange(0, max_len)
    return (ids < lengths.expand_dims(1)).astype(np.uint8)
