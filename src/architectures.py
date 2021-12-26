#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================================ #
# Part of:
# "The Stock Transformer: whole-system multidimensional financial time series
# forecasting from timestamped prices via stacked self-attention"
#
# Davide Roznowicz, Emanuele Ballarin <emanuele@ballarin.cc>
#
# (https://github.com/emaballarin/financial-wholenamycs)
# ============================================================================ #

# Imports
import torch as th
from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from ebtorch.nn import FCBlock, CausalConv1d
from ebtorch.nn.utils import argser_f

# Typing
from typing import Union, Optional
from torch import Tensor

# Utility
import copy


# Classes
class ConvFeaturizer1d(nn.Module):
    def __init__(
        self,
        # Fixed-length lists (lists of inputs to Conv1d)
        in_channels: Union[list, tuple],
        out_channels: Union[list, tuple],
        kernel_size: Union[list, tuple],
        stride: Optional[Union[list, tuple]] = None,
        padding: Optional[Union[list, tuple]] = None,
        dilation: Optional[Union[list, tuple]] = None,
        groups: Optional[Union[list, tuple]] = None,
        # Optionals (lists of inputs to Conv1d)
        bias: Optional[Union[list, tuple, bool]] = None,
        padding_mode: Optional[Union[list, tuple, str]] = None,
        # Optionals (layer descriptions)
        batchnorm: Optional[Union[list, tuple, bool]] = None,
        causal: Optional[Union[list, tuple, bool]] = None,
        activation_fx: Optional[Union[nn.ModuleList, nn.Module]] = None,
    ) -> None:

        # Call to super()
        super().__init__()

        self.activation_fx = nn.ModuleList()

        error_uneven_size: str = (
            "The length of lists of arguments must be the same across them."
        )

        # Sanitize
        if not len(in_channels) == len(out_channels) == len(kernel_size):
            raise ValueError(error_uneven_size)

        # Handle default values for convolutions
        if stride is None:
            stride = [1] * len(in_channels)
        if padding is None:
            padding = [0] * len(in_channels)
        if dilation is None:
            dilation = [1] * len(in_channels)
        if groups is None:
            groups = [1] * len(in_channels)
        if bias is None:
            bias = True
        if padding_mode is None:
            padding_mode = "zeros"
        if batchnorm is None:
            batchnorm = [True for _ in range(len(in_channels) - 1)] + [False]
        if causal is None:
            causal = True
        if activation_fx is None:
            for _ in range(len(in_channels) - 1):
                self.activation_fx.append(nn.ReLU())
            self.activation_fx.append(nn.Identity())

        if (
            not len(in_channels)
            == len(stride)
            == len(dilation)
            == len(groups)
            == len(padding)
        ):
            raise ValueError(error_uneven_size)

        if isinstance(bias, bool):
            bias = [bias] * len(in_channels)
        if isinstance(padding_mode, str):
            padding_mode = [padding_mode] * len(in_channels)

        if isinstance(batchnorm, bool):
            batchnorm = [batchnorm] * len(in_channels)
        if isinstance(causal, bool):
            causal = [causal] * len(in_channels)

        if isinstance(activation_fx, nn.Module) and not isinstance(
            activation_fx, nn.ModuleList
        ):
            for _ in range(len(in_channels)):
                self.activation_fx.append(copy.deepcopy(activation_fx))
        elif isinstance(activation_fx, nn.ModuleList):
            self.activation_fx = copy.deepcopy(activation_fx)

        if (
            not len(in_channels)
            == len(bias)
            == len(padding_mode)
            == len(batchnorm)
            == len(causal)
            == len(self.activation_fx)
        ):
            print(
                len(in_channels),
                len(bias),
                len(padding_mode),
                len(batchnorm),
                len(causal),
                len(self.activation_fx),
            )
            raise ValueError(error_uneven_size)

        # Start with an empty module list
        self.conv_battery = nn.ModuleList(modules=None)

        # Build it iteratively
        for conv_idx in range(len(in_channels)):

            if causal[conv_idx]:
                self.conv_battery.append(
                    CausalConv1d(
                        in_channels=in_channels[conv_idx],
                        out_channels=out_channels[conv_idx],
                        kernel_size=kernel_size[conv_idx],
                        stride=stride[conv_idx],  # legit
                        dilation=dilation[conv_idx],  # legit
                        groups=groups[conv_idx],
                        bias=bias[conv_idx],
                        padding_mode=padding_mode[conv_idx],
                    )
                )
            else:
                self.conv_battery.append(
                    nn.Conv1d(
                        in_channels=in_channels[conv_idx],
                        out_channels=out_channels[conv_idx],
                        kernel_size=kernel_size[conv_idx],
                        stride=stride[conv_idx],  # legit
                        padding=padding[conv_idx],  # legit
                        dilation=dilation[conv_idx],  # legit
                        groups=groups[conv_idx],
                        bias=bias[conv_idx],
                        padding_mode=padding_mode[conv_idx],
                    )
                )

            self.conv_battery.append(copy.deepcopy(self.activation_fx[conv_idx]))

            if batchnorm[conv_idx]:
                self.conv_battery.append(
                    nn.BatchNorm1d(num_features=out_channels[conv_idx])
                )

    def forward(self, x: Tensor) -> Tensor:
        for module_idx in range(len(self.conv_battery)):
            x = self.conv_battery[module_idx](x)
        return x


class ConvFeaturizer(nn.Module):
    def __init__(
        self,
        conv_nr,
        in_channels: Union[list, tuple],
        out_channels: Union[list, tuple],
        kernel_size: Union[list, tuple],
        stride: Optional[Union[list, tuple]] = None,
        padding: Optional[Union[list, tuple]] = None,
        dilation: Optional[Union[list, tuple]] = None,
        groups: Optional[Union[list, tuple]] = None,
        bias: Optional[Union[list, tuple, bool]] = None,
        padding_mode: Optional[Union[list, tuple, str]] = None,
        batchnorm: Optional[Union[list, tuple, bool]] = None,
        causal: Optional[Union[list, tuple, bool]] = None,
        activation_fx: Optional[Union[nn.ModuleList, nn.Module]] = None,
    ) -> None:

        super().__init__()

        self.conv_nr = conv_nr

        self.convfeat_battery = nn.ModuleList(
            ConvFeaturizer1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
                padding_mode,
                batchnorm,
                causal,
                activation_fx,
            )
            for _ in range(self.conv_nr)
        )

    def forward(self, x: Tensor) -> Tensor:

        temp_list = []

        for series_idx in range(self.conv_nr):

            temp: Tensor = self.convfeat_battery[series_idx](
                x[:, series_idx, :, :]
            ).transpose(1, 2)

            temp_list.append(temp)

        return th.stack(temp_list, dim=2)


class StockTransformerModel(nn.Module):
    def __init__(
        self,
        conv_featurizer_args: list,
        mlp_correlator_args: list,
        encoder_layer_args: list,
        encoder_args: list,
        decoder_args: list,
        ctx_size: int,
        fin_size: int,
    ) -> None:
        super().__init__()

        # Useful parameters
        self.ctx_size = ctx_size
        self.fin_size = fin_size

        # Blocks
        self.conv_featurizer = argser_f(ConvFeaturizer, conv_featurizer_args)()
        self.mlp_correlator = argser_f(FCBlock, mlp_correlator_args)()
        encoder_layer = argser_f(TransformerEncoderLayer, encoder_layer_args)(
            batch_first=True
        )
        self.encoder = argser_f(TransformerEncoder, encoder_args)(
            encoder_layer=encoder_layer
        )
        self.decoder = argser_f(FCBlock, decoder_args)()

    def forward(self, x: Tensor) -> Tensor:

        x_ctx, x_fin = th.split(
            x,
            [self.ctx_size, self.fin_size],
            dim=1,
        )

        featurized_fin = self.conv_featurizer(x_fin)

        # TODO:  Check dimensions from outside!
        # mlp_correlator_input_size = featurized_fin.shape[2] * featurized_fin.shape[3]

        temp_list = []

        for time_point in range(featurized_fin.shape[1]):
            temp = th.flatten(featurized_fin, start_dim=2, end_dim=-1)[:, time_point, :]
            temp_2 = self.mlp_correlator(temp)
            temp_list.append(temp_2)

        corrinfo_fin = th.stack(temp_list, dim=1)

        x_ctx_new = th.flatten(x_ctx, start_dim=2, end_dim=-1).transpose(-2, -1)

        initial = x_ctx_new[:, 0, :]
        increment = (x_ctx_new[:, -1, :] - initial) / (corrinfo_fin.shape[1] - 1)

        temp_list = []
        for token_nr in range(corrinfo_fin.shape[1]):
            temp_list.append(initial + token_nr * increment)
        ready_ctx = th.stack(temp_list, dim=1)

        transformer_input = th.cat(
            (
                th.flatten(featurized_fin, start_dim=2, end_dim=-1),
                corrinfo_fin,
                ready_ctx,
            ),
            dim=-1,
        )

        tr_enc_output = self.encoder(transformer_input)
        post_dec = self.decoder(th.flatten(tr_enc_output, start_dim=1, end_dim=-1))
        return post_dec.unflatten(1, (-1, self.fin_size))
