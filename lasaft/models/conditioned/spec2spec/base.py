from warnings import warn

from lasaft.models.conditioned.spec2spec.embedding import LinearEmbedding
from lasaft.models.sub_modules.building_blocks import mk_norm_2d
from lasaft.utils.functions import get_activation_by_name
import torch
import torch.nn as nn


class BaseNet(nn.Module):

    def __init__(self,
                 n_fft,
                 input_channels, internal_channels,
                 n_blocks, n_internal_layers,
                 mk_block_f, mk_ds_f, mk_us_f,
                 first_conv_activation, last_activation,

                 # Conditional Mechanism #
                 control_vector_type, control_input_dim, embedding_dim, condition_to,
                 norm='bn'
                 ):

        first_conv_activation = get_activation_by_name(first_conv_activation)
        last_activation = get_activation_by_name(last_activation)

        super(BaseNet, self).__init__()

        '''num_block should be an odd integer'''
        assert n_blocks % 2 == 1

        dim_f, t_down_layers, f_down_layers = self.mk_overall_structure(n_fft, internal_channels, input_channels,
                                                                        n_blocks,
                                                                        n_internal_layers, last_activation,
                                                                        first_conv_activation, norm)

        #########################
        # Conditional Mechanism #
        #########################
        assert control_vector_type in ['one_hot_mode', 'embedding', 'linear']
        if control_vector_type == 'one_hot_mode':
            if control_input_dim != embedding_dim:
                warn('in one_hot_mode, embedding_dim should be the same as num_targets. auto correction')
                embedding_dim = control_input_dim

                with torch.no_grad():
                    one_hot_weight = torch.zeros((control_input_dim, embedding_dim))
                    for i in range(control_input_dim):
                        one_hot_weight[i, i] = 1.

                    self.embedding = nn.Embedding(control_input_dim, embedding_dim, _weight=one_hot_weight)
                    self.embedding.weight.requires_grad = True
        elif control_vector_type == 'embedding':
            self.embedding = nn.Embedding(control_input_dim, embedding_dim)

        elif control_vector_type == 'linear':
            self.embedding = LinearEmbedding(control_input_dim, embedding_dim)

        self.control_input_dim = control_input_dim
        self.embedding_dim = embedding_dim

        # Where to condition
        assert condition_to in ['encoder', 'decoder', 'full']
        self.is_encoder_conditioned = self.is_middle_conditioned = self.is_decoder_conditioned = False
        if condition_to == 'encoder':
            self.is_encoder_conditioned = True
        elif condition_to == 'decoder':
            self.is_decoder_conditioned = True
        elif condition_to == 'full':
            self.is_encoder_conditioned = self.is_middle_conditioned = self.is_decoder_conditioned = True
        else:
            raise NotImplementedError

        f = dim_f
        for i in range(self.n):
            self.encoders.append(mk_block_f(internal_channels, f, self.is_encoder_conditioned))
            ds_layer, f = mk_ds_f(internal_channels, f)
            self.downsamplings.append(ds_layer)
        self.mid_block = mk_block_f(internal_channels, f, self.is_middle_conditioned)
        for i in range(self.n):
            us_layer, f = mk_us_f(internal_channels, f)
            self.upsamplings.append(us_layer)
            self.decoders.append(mk_block_f(2 * internal_channels, f, self.is_decoder_conditioned))

        self.activation = self.last_conv[-1]

    def mk_overall_structure(self, n_fft, internal_channels, input_channels, n_blocks, n_internal_layers,
                             last_activation, first_conv_activation, norm='bn'):
        dim_f = n_fft // 2
        input_channels = input_channels

        mk_norm = mk_norm_2d(internal_channels, norm)
        self.first_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=internal_channels,
                kernel_size=(1, 2),
                stride=1
            ),
            mk_norm(),
            first_conv_activation(),
        )
        self.encoders = nn.ModuleList()
        self.downsamplings = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upsamplings = nn.ModuleList()
        self.last_conv = nn.Sequential(

            nn.Conv2d(
                in_channels=internal_channels,
                out_channels=input_channels,
                kernel_size=(1, 2),
                stride=1,
                padding=(0, 1)
            ),
            last_activation()
        )
        self.n = n_blocks // 2

        t_down_layers = list(range(self.n))
        f_down_layers = list(range(self.n))

        return dim_f, t_down_layers, f_down_layers

    def forward(self, input_spec, input_condition):

        condition_embedding = self.embedding(input_condition)

        x = self.first_conv(input_spec)
        encoding_outputs = []

        for encoder, downsampling in zip(self.encoders, self.downsamplings):
            if self.is_encoder_conditioned:
                x = encoder(x, condition_embedding)
            else:
                x = encoder(x)
            encoding_outputs.append(x)
            x = downsampling(x)

        if self.is_middle_conditioned:
            x = self.mid_block(x, condition_embedding)
        else:
            x = self.mid_block(x)

        for i in range(self.n):
            x = self.upsamplings[i](x)
            x = torch.cat((x, encoding_outputs[-i - 1]), 1)

            if self.is_decoder_conditioned:
                x = self.decoders[i](x, condition_embedding)
            else:
                x = self.decoders[i](x)

        return self.last_conv(x)

    def forward_multi_source(self, input_spec, conditions, num_targets):

        conditions = conditions.view(-1)
        condition_embedding = self.embedding(conditions)

        x = self.first_conv(input_spec)
        encoding_outputs = []

        for encoder, downsampling in zip(self.encoders, self.downsamplings):
            x = encoder(x)
            encoding_outputs.append(repeat_conditions(num_targets, x))
            x = downsampling(x)

        x = self.mid_block(x)

        x = repeat_conditions(num_targets, x)

        for i in range(self.n):
            x = self.upsamplings[i](x)
            x = torch.cat((x, encoding_outputs[-i - 1]), 1)
            x = self.decoders[i](x, condition_embedding)

        output = self.last_conv(x)
        output_shape = list(output.shape)
        output_shape[0] = num_targets
        return output.view([-1] + output_shape)


class BaseNetWithMultiSource(BaseNet):

    def __init__(self, n_fft, input_channels, internal_channels, n_blocks, n_internal_layers, mk_block_f, mk_ds_f,
                 mk_us_f, first_conv_activation, last_activation, control_vector_type, control_input_dim, embedding_dim,
                 condition_to, norm='bn'):
        super().__init__(n_fft, input_channels, internal_channels, n_blocks, n_internal_layers, mk_block_f, mk_ds_f,
                         mk_us_f, first_conv_activation, last_activation, control_vector_type, control_input_dim,
                         embedding_dim, condition_to, norm)

        assert control_vector_type == 'linear'
        assert not self.is_encoder_conditioned
        assert not self.is_middle_conditioned


def repeat_conditions(num_conditions, a_tensor):
    x_shape = list(a_tensor.shape)
    a_tensor = a_tensor.unsqueeze(1).repeat(1, num_conditions, 1, 1, 1)
    x_shape[0] = -1
    a_tensor = a_tensor.view(x_shape)
    return a_tensor
