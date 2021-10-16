from abc import abstractmethod

from lasaft.models.sub_modules.building_blocks import *
from lasaft.utils.functions import get_activation_by_name


class MkBlock(object):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, **kwargs):
        pass


class MkTFCLaSAFT(MkBlock):

    def __init__(self, gr, num_layers, kt, kf, bn_factor, min_bn_units, bias, activation,
                 condition_dim, num_tdfs, dk, norm='bn',
                 query_listen=False, key_listen=False) -> None:
        super().__init__()
        self.gr = gr
        self.num_layers = num_layers
        self.kt = kt
        self.kf = kf
        self.bn_factor = bn_factor
        self.min_bn_units = min_bn_units
        self.bias = bias
        self.activation = get_activation_by_name(activation)
        self.condition_dim = condition_dim
        self.num_tdfs = num_tdfs
        self.dk = dk
        self.norm = norm

        # QKV Listening
        self.query_listen = query_listen
        self.key_listen = key_listen

    def __call__(self, input_channels, f, is_conditioned):

        if is_conditioned:
            return TFC_LaSAFT(
                in_channels=input_channels,
                num_layers=self.num_layers,
                gr=self.gr,
                kt=self.kt,
                kf=self.kf,
                f=f,
                bn_factor=self.bn_factor,
                min_bn_units=self.min_bn_units,
                bias=self.bias,
                activation=self.activation,
                condition_dim=self.condition_dim,
                num_tdfs=self.num_tdfs,
                dk=self.dk,
                norm=self.norm,
                query_listen = self.query_listen,
                key_listen=self.key_listen
            )

        else:
            return TFC_TDF(
                in_channels=input_channels,
                num_layers=self.num_layers,
                gr=self.gr,
                kt=self.kt,
                kf=self.kf,
                f=f,
                bn_factor=self.bn_factor,
                min_bn_units=self.min_bn_units,
                bias=self.bias,
                activation=self.activation,
                norm=self.norm
            )


class MkTFCLaSAFTGPoCM(MkTFCLaSAFT):

    def __init__(self, gr, num_layers, kt, kf, bn_factor, min_bn_units, bias, activation,
                 condition_dim, num_tdfs, dk, norm='bn',
                 key_listen=False) -> None:
        super().__init__(gr, num_layers, kt, kf, bn_factor, min_bn_units, bias, activation,
                         condition_dim, num_tdfs, dk, norm, key_listen)

    def __call__(self, input_channels, f, is_conditioned):
        if is_conditioned:
            return TFC_LaSAFT_GPoCM(
                in_channels=input_channels,
                num_layers=self.num_layers,
                gr=self.gr,
                kt=self.kt,
                kf=self.kf,
                f=f,
                bn_factor=self.bn_factor,
                min_bn_units=self.min_bn_units,
                bias=self.bias,
                activation=self.activation,
                condition_dim=self.condition_dim,
                num_tdfs=self.num_tdfs,
                dk=self.dk,
                norm=self.norm,
                key_listen=self.key_listen
            )
        else:
            return TFC_TDF(
                in_channels=input_channels,
                num_layers=self.num_layers,
                gr=self.gr,
                kt=self.kt,
                kf=self.kf,
                f=f,
                bn_factor=self.bn_factor,
                min_bn_units=self.min_bn_units,
                bias=self.bias,
                activation=self.activation,
                norm=self.norm
            )


class MkTFCLightSAFTGPoCM(MkTFCLaSAFT):

    def __init__(self, gr, num_layers, kt, kf, bn_factor, min_bn_units, bias, activation,
                 condition_dim, num_tdfs, dk, norm='bn',
                 query_listen=False, key_listen=False) -> None:
        super().__init__(gr, num_layers, kt, kf, bn_factor, min_bn_units, bias, activation,
                         condition_dim, num_tdfs, dk, norm, query_listen, key_listen)

    def __call__(self, input_channels, f, is_conditioned):
        if is_conditioned:
            return TFC_LightSAFT_GPoCM(
                in_channels=input_channels,
                num_layers=self.num_layers,
                gr=self.gr,
                kt=self.kt,
                kf=self.kf,
                f=f,
                bn_factor=self.bn_factor,
                min_bn_units=self.min_bn_units,
                bias=self.bias,
                activation=self.activation,
                condition_dim=self.condition_dim,
                num_tdfs=self.num_tdfs,
                dk=self.dk,
                norm=self.norm,
                query_listen=self.query_listen,
                key_listen=self.key_listen
            )
        else:
            return TFC_TDF(
                in_channels=input_channels,
                num_layers=self.num_layers,
                gr=self.gr,
                kt=self.kt,
                kf=self.kf,
                f=f,
                bn_factor=self.bn_factor,
                min_bn_units=self.min_bn_units,
                bias=self.bias,
                activation=self.activation,
                norm=self.norm
            )


class MKConvDS(MkBlock):

    def __init__(self, activation, norm='bn') -> None:
        super().__init__()
        self.activation = get_activation_by_name(activation)
        self.norm = norm

    def __call__(self, input_channels, f):
        mk_norm = mk_norm_2d(input_channels, self.norm)
        ds = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=input_channels,
                      kernel_size=(2, 2), stride=(2, 2)),
            mk_norm(),
            self.activation()
        )
        return ds, f // 2


class MkConvTransposeUS(MkBlock):

    def __init__(self, activation, norm='bn') -> None:
        super().__init__()
        self.activation = get_activation_by_name(activation)
        self.norm = norm

    def __call__(self, input_channels, f):
        mk_norm = mk_norm_2d(input_channels, self.norm)
        us = nn.Sequential(
            nn.ConvTranspose2d(in_channels=input_channels, out_channels=input_channels,
                               kernel_size=(2, 2), stride=(2, 2)),
            mk_norm(),
            self.activation()
        )
        return us, f * 2
