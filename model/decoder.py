from mxnet import gluon
from model.layers.pre_net import PreNet
from model.layers.linear_norm import LinearNorm
from model.layers.attention import Attention
from params import tacotron_params
from utils import get_mask_from_lengths


class Decoder(gluon.HybridBlock):

    def __init__(self, opts):
        super(Decoder, self).__init__()
        params = tacotron_params

        # preparing PreNet
        self._mel_channels = params.n_mel_channels
        self._frames_per_step = params.n_frames_per_step
        self._prenet_dim = params.prenet_dim
        self._prenet = PreNet(self._mel_channels * self._frames_per_step, [self._prenet_dim, self._prenet_dim])

        # self.encoder_embedding_dim = kwargs['encoder_embedding_dim']
        # self.attention_rnn_dim = kwargs['attention_rnn_dim']
        # self.decoder_rnn_dim = kwargs['decoder_rnn_dim']
        # self.max_decoder_steps = kwargs['max_decoder_steps']
        # self.gate_threshold = kwargs['gate_threshold']
        # self.p_attention_dropout = kwargs['p_attention_dropout']
        # self.p_decoder_dropout = kwargs['p_decoder_dropout']
        # self.attention_dim = kwargs['attention_dim']
        # self.attention_location_n_filters = kwargs['attention_location_n_filters']
        # self.attention_location_kernel_size = kwargs['attention_location_kernel_size']


        #self.attention_rnn = nn.LSTMCell(hparams.prenet_dim + hparams.encoder_embedding_dim, hparams.attention_rnn_dim)

        # preparing Attention network
        self._attn_dim = params.attn_dim
        self._attn_rnn_dim = params.attn_rnn_dim
        self._enc_embed_dim = params.encoder_embedding_dim
        self._attn_loc_kernel = params.attn_loc_kernel
        self._attn_loc_filters = params.attn_loc_n_filters
        self._attn_layer = Attention(self._attn_rnn_dim, self._enc_embed_dim, self._attn_dim, self._attn_loc_filters, self._attn_loc_kernel)

        #self.decoder_rnn = nn.LSTMCell(hparams.attention_rnn_dim + hparams.encoder_embedding_dim, hparams.decoder_rnn_dim, 1)

        self._dec_rnn_dim = params.decoder_rnn_dim
        lin_input_dim = self._dec_rnn_dim + self._enc_embed_dim
        self._linear_proj = LinearNorm(lin_input_dim, self._mel_channels * self._frames_per_step, 'linear')
        self._gate_layer = LinearNorm(lin_input_dim, 1, 'sigmoid', bias=True)

    def hybrid_forward(self, F, memory, decoder_inputs, memory_lengths, *args, **kwargs):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.
        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """

        #memory = memory.transpose((1, 0, 2))
        decoder_input = self.get_go_frame(F, memory).expand_dims(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = F.concat(decoder_input, decoder_inputs, dim=0)
        decoder_inputs = self._prenet(decoder_inputs)

        self.initialize_decoder_states(F, memory, mask=get_mask_from_lengths(F, memory_lengths))
        #self.initialize_decoder_states(F, memory, mask=~get_mask_from_lengths(F, memory_lengths))

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.shape[0] - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(F, decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze()]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)
        return mel_outputs, gate_outputs, alignments

    def get_go_frame(self, F, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs
        RETURNS
        -------
        decoder_input: all zeros frames
        """
        batch = memory.shape[0]
        return F.zeros((batch, self._mel_channels * self._frames_per_step))

    def initialize_decoder_states(self, F, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        batch = memory.shape[0]
        max_time = memory.shape[1]

        self.attention_hidden = F.zeros((batch, self._attn_rnn_dim))
        self.attention_cell = F.zeros((batch, self._attn_rnn_dim))
        self.decoder_hidden = F.zeros((batch, self._dec_rnn_dim))
        self.decoder_cell = F.zeros((batch, self._dec_rnn_dim))

        #self.attention_hidden = Variable(memory.data.new(B, self.attention_rnn_dim).zero_())
        #self.attention_cell = Variable(memory.data.new(B, self.attention_rnn_dim).zero_())
        #self.decoder_hidden = Variable(memory.data.new(B, self.decoder_rnn_dim).zero_())
        #self.decoder_cell = Variable(memory.data.new(B, self.decoder_rnn_dim).zero_())

        #self.attention_weights = Variable(memory.data.new(B, MAX_TIME).zero_())
        #self.attention_weights_cum = Variable(memory.data.new(B, MAX_TIME).zero_())
        #self.attention_context = Variable(memory.data.new(B, self.encoder_embedding_dim).zero_())

        self.attention_weights = F.zeros((batch, max_time))
        self.attention_weights_cum = F.zeros((batch, max_time))
        self.attention_context = F.zeros((batch, self._enc_embed_dim))

        self.memory = memory
        self.processed_memory = self._attn_layer.memory_layer(memory.reshape(-1, memory.shape[2]))
        self.processed_memory = self.processed_memory.reshape((batch, -1, self.processed_memory.shape[1]))
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs
        RETURNS
        -------
        inputs: processed decoder inputs
        """
        # (batch, mel_channels, T_out) -> (batch, T_out, mel_channels)
        decoder_inputs = decoder_inputs.transpose((0, 2, 1))
        decoder_inputs = decoder_inputs.reshape(decoder_inputs.shape[0], decoder_inputs.shape[1] // self._frames_per_step, -1)

        # (batch, T_out, mel_channels) -> (T_out, batch, mel_channels)
        decoder_inputs = decoder_inputs.transpose((1, 0, 2))
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:
        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self._mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, F, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output
        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(self.attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat((self.attention_weights.unsqueeze(1), self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(self.attention_hidden, self.memory, self.processed_memory, attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat((self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat((self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def inference(self, memory):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs
        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self._prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments
