import math

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from lasaft.utils.PoCM_utils import Pocm_Matmul


def mk_norm_2d(num_channels, norm):
    if norm == 'bn':
        def norm():
            return nn.BatchNorm2d(num_channels)

        mk_norm = norm

    elif norm == 'gn':
        num_groups = 1
        gr_sqrt = math.ceil(math.sqrt(num_channels))
        for i in range(gr_sqrt, 2, -1):
            if num_channels % i == 0:
                num_groups = i
                break

        def norm():
            return nn.GroupNorm(num_groups, num_channels)

        mk_norm = norm

    else:
        raise ModuleNotFoundError

    return mk_norm


class TFC(nn.Module):
    """ [B, in_channels, T, F] => [B, gr, T, F] """

    def __init__(self, in_channels, num_layers, gr, kt, kf, activation, norm='bn'):
        """
        in_channels: number of input channels
        num_layers: number of densely connected conv layers
        gr: growth rate
        kt: kernel size of the temporal axis.
        kf: kernel size of the freq. axis
        activation: activation function
        """
        super(TFC, self).__init__()

        c = in_channels
        self.H = nn.ModuleList()

        mk_norm_func = mk_norm_2d(gr, norm)

        for i in range(num_layers):
            self.H.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=gr, kernel_size=(kf, kt), stride=1,
                              padding=(kt // 2, kf // 2)),
                    mk_norm_func(),
                    activation(),
                )
            )
            c += gr

        self.activation = self.H[-1][-1]

    def forward(self, x):
        """ [B, in_channels, T, F] => [B, gr, T, F] """
        x_ = self.H[0](x)
        for h in self.H[1:]:
            x = torch.cat((x_, x), 1)
            x_ = h(x)

        return x_


class TDF(nn.Module):
    """ [B, in_channels, T, F] => [B, gr, T, F] """

    def __init__(self, channels, f, bn_factor=16, bias=False, min_bn_units=16, activation=nn.ReLU, norm='bn'):

        """
        channels: # channels
        f: num of frequency bins
        bn_factor: bottleneck factor. if None: single layer. else: MLP that maps f => f//bn_factor => f
        bias: bias setting of linear layers
        activation: activation function
        """

        mk_norm = mk_norm_2d(channels, norm)

        super(TDF, self).__init__()
        if bn_factor is None:
            self.tdf = nn.Sequential(
                nn.Linear(f, f, bias),
                mk_norm(),
                activation()
            )

        else:
            bn_units = max(f // bn_factor, min_bn_units)
            self.bn_units = bn_units
            self.tdf = nn.Sequential(
                nn.Linear(f, bn_units, bias),
                mk_norm(),
                activation(),
                nn.Linear(bn_units, f, bias),
                mk_norm(),
                activation()
            )

    def forward(self, x):
        return self.tdf(x)


class TFC_TDF(nn.Module):
    def __init__(self, in_channels, num_layers, gr, kt, kf, f, bn_factor=16, min_bn_units=16, bias=False,
                 activation=nn.ReLU, norm='bn'):
        """
        in_channels: number of input channels
        num_layers: number of densely connected conv layers
        gr: growth rate
        kt: kernel size of the temporal axis.
        kf: kernel size of the freq. axis
        f: num of frequency bins

        below are params for TDF
        bn_factor: bottleneck factor. if None: single layer. else: MLP that maps f => f//bn_factor => f
        bias: bias setting of linear layers

        activation: activation function
        """

        super(TFC_TDF, self).__init__()
        self.tfc = TFC(in_channels, num_layers, gr, kt, kf, activation, norm)
        self.tdf = TDF(gr, f, bn_factor, bias, min_bn_units, activation, norm)
        self.activation = self.tdf.tdf[-1]

    def forward(self, x):
        x = self.tfc(x)
        return x + self.tdf(x)


class TDF_f1_to_f2(nn.Module):
    """ [B, in_channels, T, F] => [B, gr, T, F] """

    def __init__(self, channels, f1, f2, bn_factor=16, bias=False, min_bn_units=16, activation=nn.ReLU, norm='bn'):

        """
        channels:  # channels
        f1: num of frequency bins (input)
        f2: num of frequency bins (output)
        bn_factor: bottleneck factor. if None: single layer. else: MLP that maps f => f//bn_factor => f
        bias: bias setting of linear layers
        activation: activation function
        """

        super(TDF_f1_to_f2, self).__init__()

        self.num_target_f = f2

        mk_norm = mk_norm_2d(channels, norm)
        if bn_factor is None:
            self.tdf = nn.Sequential(
                nn.Linear(f1, f2, bias),
                mk_norm(),
                activation()
            )

        else:
            bn_units = max(f2 // bn_factor, min_bn_units)
            self.tdf = nn.Sequential(
                nn.Linear(f1, bn_units, bias),
                mk_norm(),
                activation(),
                nn.Linear(bn_units, f2, bias),
                mk_norm(),
                activation()
            )

    def forward(self, x):
        return self.tdf(x)



class TFC_LaSAFT(nn.Module):
    def __init__(self, in_channels, num_layers, gr, kt, kf, f, bn_factor, min_bn_units, bias,
                 activation, condition_dim, num_tdfs, dk, norm='bn',
                 query_listen=False, key_listen=False):
        super(TFC_LaSAFT, self).__init__()
        import math
        self.dk_sqrt = math.sqrt(dk)
        self.num_tdfs = num_tdfs
        self.tfc = TFC(in_channels, num_layers, gr, kt, kf, activation, norm)
        self.tdfs = TDF_f1_to_f2(gr, f, f * num_tdfs, bn_factor, bias, min_bn_units, activation, norm)

        self.query_listen = query_listen
        self.key_listen = key_listen

        if self.query_listen and self. key_listen:
            raise NotImplementedError

        elif self.query_listen:
            self.query_pocm_generator = nn.Sequential(
                weight_norm(nn.Linear(condition_dim, condition_dim)),
                nn.ReLU(),
                weight_norm(nn.Linear(condition_dim, (gr + 1)))
            )
            self.linear_query = nn.Linear(f, dk)

            self.keys = nn.Parameter(torch.randn(1, 1, dk, num_tdfs), requires_grad=True)

        elif self.key_listen:
            self.linear_query = nn.Linear(condition_dim, dk)
            self.keys = nn.Sequential(
                nn.Conv2d(gr, num_tdfs, (1, 1), (1, 1)),
                nn.BatchNorm2d(num_tdfs),
                nn.Linear(f, dk)
            )

        else:
            self.linear_query = nn.Linear(condition_dim, dk)
            self.keys = nn.Parameter(torch.randn(dk, num_tdfs), requires_grad=True)

        self.activation = self.tdfs.tdf[-1]

    def forward(self, x, c):
        x = self.tfc(x)
        return x + self.lasaft(x, c)

    def lasaft(self, x, c):

        value = (self.tdfs(x)).view(list(x.shape)[:-1] + [-1, self.num_tdfs])

        if self.query_listen and self.key_listen:
            raise NotImplementedError

        elif self.query_listen:

            # Query [B, T, 1, Dk]
            # Key [1, 1, Dk, L]
            # QK [B, T, 1, L]
            # Value [B, ch, T, F, L]

            query_pocm_weight = self.query_pocm_generator(c)
            gammas = query_pocm_weight[..., : -1].view(-1, self.internal_channels, 1)
            betas = query_pocm_weight[..., -1:]
            query = Pocm_Matmul(x, gammas, betas).relu()
            query_listened = self.linear_query(query).squeeze(-3).unsqueeze(-2)
            qk = torch.matmul(query_listened, self.keys) / self.dk_sqrt
            att = qk.softmax(-1).transpose(-1, -2).unsqueeze(-4)
            return torch.matmul(value, att).squeeze(-1)

        elif self.key_listen:

            # Query [B, 1, Dk, 1]
            # Key [Dk, L]:  [B, ch, T, F] => [B, L, T, Dk] => [B, T, L, Dk]
            # KQ [B, 1, T, L, 1]: [B, T, L ,-, 1] => [B, 1, T, L, 1]
            # Value [B, ch, T, F, L]

            query = self.linear_query(c).unsqueeze(-2).unsqueeze(-1)
            key = self.keys(x).transpose(-2, -3)
            kq = torch.matmul(key, query) / self.dk_sqrt
            att = kq.softmax(-2).unsqueeze(-4)
        else:

            # Query [B, Dk]
            # Key [Dk, L]
            # QK [B, L]
            # Value [B, ch, T, F, L]

            query = self.linear_query(c)
            key = self.keys
            qk = torch.matmul(query, key) / self.dk_sqrt
            att = qk.softmax(-1).unsqueeze(-2).unsqueeze(-3).unsqueeze(-1)
        return torch.matmul(value, att).squeeze(-1)


class TFC_LightSAFT(nn.Module):
    def __init__(self, in_channels, num_layers, gr, kt, kf, f, bn_factor, min_bn_units, bias,
                 activation, condition_dim, num_tdfs, dk, norm='bn',
                 query_listen=False, key_listen=False):
        super(TFC_LightSAFT, self).__init__()
        import math
        self.dk_sqrt = math.sqrt(dk)
        self.internal_channels = gr
        self.num_tdfs = num_tdfs
        self.tfc = TFC(in_channels, num_layers, gr, kt, kf, activation, norm)
        num_bn_features = f // bn_factor
        if num_bn_features < min_bn_units:
            num_bn_features = min_bn_units

        mk_norm = mk_norm_2d(gr, norm)

        self.LaSAFT_bn_layer = nn.Sequential(
            nn.Linear(f, num_bn_features * num_tdfs, bias),
            mk_norm(),
            activation()
        )
        self.LaSIFT = nn.Sequential(
            mk_norm(),
            nn.Linear(num_bn_features, f, bias),
            activation()
        )

        # Conditioning

        self.query_listen = query_listen
        self.key_listen = key_listen

        if self.query_listen:  # from branch query_listen
            self.query_pocm_generator = nn.Sequential(
                weight_norm(nn.Linear(condition_dim, condition_dim)),
                nn.ReLU(),
                weight_norm(nn.Linear(condition_dim, (gr + 1)))
            )

            self.linear_query = nn.Sequential(
                nn.LayerNorm(f),
                nn.ReLU(),
                nn.Linear(f, dk)
            )

        else:
            self.linear_query = nn.Linear(condition_dim, dk)

        if self.key_listen:  # from branch light_kl3
            self.keys = nn.Sequential(
                nn.Conv2d(gr, num_tdfs, (1, 1), (1, 1)),
                nn.BatchNorm2d(num_tdfs),
                nn.ReLU(),
                nn.Linear(f, dk)
            )
        else:
            self.keys = nn.Parameter(torch.randn(1, 1, dk, num_tdfs), requires_grad=True)

        self.activation = self.LaSIFT[-1]

    def forward(self, x, c):
        x = self.tfc(x)
        return x + self.lasaft(x, c)

    def lasaft(self, x, c):

        value = (self.LaSAFT_bn_layer(x)).view(list(x.shape)[:-1] + [-1, self.num_tdfs])

        if self.query_listen and self.key_listen:  # from branch light_qkl4

            # Query [B, T, 1, Dk]
            # Key [B, T, L, Dk]
            # QK [B, T, L, 1]
            # Value [B, ch, T, F, L]

            query_listened = self.listen_mixture_with_condition(c, x)
            key_listened = self.keys(x).transpose(-2, -3)
            qk = torch.matmul(key_listened, query_listened.transpose(-1, -2)) / self.dk_sqrt
            att = qk.softmax(-2).unsqueeze(-4)

        elif self.query_listen:  # from branch query_listen

            # Query [B, T, 1, Dk]
            # Key [1, 1, Dk, L]
            # QK [B, T, 1, L]
            # Value [B, ch, T, F, L]

            query_listened = self.listen_mixture_with_condition(c, x)
            qk = torch.matmul(query_listened, self.keys) / self.dk_sqrt
            att = qk.softmax(-1).transpose(-1, -2).unsqueeze(-4)

        elif self.key_listen:  # from branch light_kl3
            query = self.linear_query(c)
            query = query.unsqueeze(-2).unsqueeze(-1)
            key = self.keys(x).transpose(-2, -3)
            kq = torch.matmul(key, query) / self.dk_sqrt
            att = kq.softmax(-2).unsqueeze(-4)

        else:  # from original lasaft v1
            query = self.linear_query(c).unsqueeze(-2).unsqueeze(-3)
            key = self.keys
            qk = torch.matmul(query, key) / self.dk_sqrt
            att = qk.softmax(-1).unsqueeze(-1)

        lasaft_features = torch.matmul(value, att).squeeze(-1)

        return self.LaSIFT(lasaft_features)

    def listen_mixture_with_condition(self, c, x):
        query_pocm_weight = self.query_pocm_generator(c)
        gammas = query_pocm_weight[..., : -1].view(-1, self.internal_channels, 1)
        betas = query_pocm_weight[..., -1:]
        query = Pocm_Matmul(x, gammas, betas)
        query_listened = self.linear_query(query).squeeze(-3).unsqueeze(-2)
        return query_listened


class TFC_LaSAFT_GPoCM(nn.Module):
    def __init__(self, in_channels, num_layers, gr, kt, kf, f, bn_factor, min_bn_units, bias,
                 activation, condition_dim, num_tdfs, dk, norm='bn', key_listen=False):
        super(TFC_LaSAFT_GPoCM, self).__init__()

        self.tfc_lasaft = TFC_LaSAFT(in_channels, num_layers, gr, kt, kf, f,
                                     bn_factor, min_bn_units, bias, activation, condition_dim, num_tdfs, dk, norm,
                                     key_listen)

        self.pocm_generator = nn.Sequential(
            weight_norm(nn.Linear(condition_dim, condition_dim)),
            nn.ReLU(),
            weight_norm(nn.Linear(condition_dim, (gr ** 2 + gr)))
        )

        self.out_channels = gr

    def forward(self, x, c):
        x = self.tfc_lasaft.tfc(x)
        pocm_weight = self.pocm_generator(c)
        gamma = pocm_weight[..., :-self.out_channels]
        gamma = gamma.view(list(pocm_weight.shape[:-1]) + [self.out_channels, self.out_channels])
        beta = pocm_weight[..., -self.out_channels:]
        x = x * Pocm_Matmul(x, gamma, beta).sigmoid()

        return x + self.tfc_lasaft.lasaft(x, c)


class TFC_LightSAFT_GPoCM(nn.Module):
    def __init__(self, in_channels, num_layers, gr, kt, kf, f, bn_factor, min_bn_units, bias,
                 activation, condition_dim, num_tdfs, dk, norm,
                 query_listen=False, key_listen=False):
        super(TFC_LightSAFT_GPoCM, self).__init__()

        self.tfc_lightsaft = TFC_LightSAFT(in_channels, num_layers, gr, kt, kf, f,
                                           bn_factor, min_bn_units, bias, activation,
                                           condition_dim, num_tdfs, dk,
                                           norm, query_listen, key_listen)

        self.pocm_generator = nn.Sequential(
            weight_norm(nn.Linear(condition_dim, condition_dim)),
            nn.ReLU(),
            weight_norm(nn.Linear(condition_dim, (gr ** 2 + gr)))
        )

        self.out_channels = gr

    def forward(self, x, c):
        x = self.tfc_lightsaft.tfc(x)
        pocm_weight = self.pocm_generator(c)
        gamma = pocm_weight[..., :-self.out_channels]
        gamma = gamma.view(list(pocm_weight.shape[:-1]) + [self.out_channels, self.out_channels])
        beta = pocm_weight[..., -self.out_channels:]
        x = x * Pocm_Matmul(x, gamma, beta).sigmoid()

        return x + self.tfc_lightsaft.lasaft(x, c)
