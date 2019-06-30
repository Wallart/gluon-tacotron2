from mxnet import gluon
from model.layers.linear_norm import LinearNorm
from model.layers.location import Location


class Attention(gluon.HybridBlock):

    def __init__(self, attn_rnn_dim, embedding_dim, attn_dim, attn_location_n_filters, attn_location_kernel_size):
        super(Attention, self).__init__()

        self.memory_layer = LinearNorm(embedding_dim, attn_dim, 'tanh', bias=False)

        self._query_layer = LinearNorm(attn_rnn_dim, attn_dim, 'tanh', bias=False)
        self._v = LinearNorm(attn_dim, 1, 'linear', bias=False)
        self._location_layer = Location(attn_location_n_filters, attn_location_kernel_size, attn_dim)
        self._score_mask_value = -float('inf')

    def hybrid_forward(self, F, attn_hidden_state, memory, processed_memory, attn_weights_cat, mask, *args, **kwargs):
        """
        PARAMS
        ------
        attn_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attn_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(F, attn_hidden_state, processed_memory, attn_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self._score_mask_value)

        attn_weights = F.softmax(alignment, dim=1)
        attn_context = torch.bmm(attn_weights.unsqueeze(1), memory)
        attn_context = attn_context.squeeze(1)

        return attn_context, attn_weights

    def get_alignment_energies(self, F, query, processed_memory, attn_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attn_dim)
        attn_weights_cat: cumulative and prev. att weights (B, 2, max_time)
        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self._query_layer(query.unsqueeze(1))
        processed_attn_weights = self._location_layer(attn_weights_cat)
        energies = self._v(F.tanh(processed_query + processed_attn_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies
