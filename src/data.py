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
import numpy as np

import torch as th
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from attrdict import AttrDict

# Typing
from typing import Union, List, Tuple
from pathlib import Path


# Build dataset
class StockDataset(Dataset):
    def __init__(
        self,
        data_path: Union[str, Path] = "./",
        which_financial: Union[List[int], Tuple] = (k for k in range(92)),
        which_contextual: Union[List[int], Tuple] = (k for k in range(8)),
        time_lookback: int = 120,
        time_predict: int = 5,
        window_stride: int = 1,
        train: bool = True,
        ttsr: float = 0.8,
    ) -> None:

        # Save arguments useful in the training loop
        self.ctx_size = len(which_contextual)
        self.fin_size = len(which_financial)

        # Save useful arguments
        self.time_lookback = time_lookback
        self.time_predict = time_predict
        self.window_stride = window_stride
        self.train = train
        self.ttsr = ttsr

        # Compute useful quantities
        self.window_size = time_lookback + time_predict

        # Load binary dataset
        financial_np = np.load(data_path + "/financial.npy")
        contextual_np = np.load(data_path + "/contextual.npy")

        # Convert contextual: np -> torch; cast types to commonsense default
        contextual: Tensor = th.from_numpy(contextual_np).to(dtype=th.float32)
        financial: Tensor = th.from_numpy(financial_np).to(dtype=th.float32)

        # Same length for time ticks for both sub-datasets
        if contextual.shape[0] != financial.shape[0]:
            raise RuntimeError(
                "Both financial and contextual information datasets must have the same temporal dimension!"
            )

        # Get # of points in each time series
        self.timespan_size = contextual.shape[0]

        # Data subsetting
        financial = financial.index_select(1, th.tensor(which_financial))
        contextual = contextual.index_select(1, th.tensor(which_contextual))

        # Precompute dataset lengths
        self.trainlen = int(
            (self.timespan_size * ttsr - self.window_size) // window_stride
        )
        self.testlen = int(
            (self.timespan_size * (1 - ttsr) - self.window_size) // window_stride
        )

        self.timespan_offset = (
            self.timespan_size
            - (self.trainlen + self.testlen) * window_stride
            - self.window_size
        )

        # Finalize dataset
        self.financial = financial
        self.contextual = contextual

    def __getitem__(self, item):

        # Get item indices
        if self.train:
            indices = th.tensor(
                [
                    self.timespan_offset + item * self.window_stride + i
                    for i in range(0, self.window_size)
                ]
            )
        else:
            indices = th.tensor(
                [
                    self.timespan_offset
                    + self.trainlen * self.window_stride
                    + item * self.window_stride
                    + i
                    for i in range(0, self.window_size)
                ]
            )

        # Itemize
        financial_item = self.financial.index_select(0, indices)
        ctx_item = self.contextual.index_select(0, indices)

        # Compute "true" prediction output (y)
        ret_y = financial_item.index_select(
            0,
            th.tensor(
                [
                    i
                    for i in range(
                        self.time_lookback,
                        self.window_size,
                    )
                ]
            ),
        )

        # Compute inference input (x)

        lookback_indices = th.tensor([i for i in range(0, self.time_lookback)])

        ret_x_fin = financial_item.index_select(
            0,
            lookback_indices,
        )

        ret_x_ctx = ctx_item.index_select(
            0,
            lookback_indices,
        )

        ret_x = th.cat((ret_x_ctx, ret_x_fin), dim=1)

        # Retline
        return ret_x.transpose(0, 1).reshape(
            ret_x.shape[1], 1, ret_x.shape[0]
        ), ret_y.transpose(0, 1).reshape(ret_y.shape[1], 1, ret_y.shape[0])

    def __len__(self) -> int:
        if self.train:
            return self.trainlen
        else:
            return self.testlen


# Dataloader dispatcher

def stock_dataloader_dispatcher(
    data_path: Union[str, Path] = "./",
    which_financial: Union[List[int], Tuple] = (k for k in range(92)),
    which_contextual: Union[List[int], Tuple] = (k for k in range(8)),
    time_lookback: int = 120,
    time_predict: int = 5,
    window_stride: int = 1,
    ttsr: float = 0.8,
    train_bs: int = 32,
    test_bs: int = 512,
    shuffle_train: bool = True
):
    train_ds = StockDataset(
        data_path,
        which_financial,
        which_contextual,
        time_lookback,
        time_predict,
        window_stride,
        True,
        ttsr,
    )
    test_ds = StockDataset(
        data_path,
        which_financial,
        which_contextual,
        time_lookback,
        time_predict,
        window_stride,
        False,
        ttsr,
    )
    train_dl = DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=shuffle_train)
    test_dl = DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)
    totr_dl = DataLoader(dataset=train_ds, batch_size=test_bs, shuffle=False)

    return train_dl, test_dl, totr_dl, AttrDict({"fin_size": train_ds.fin_size, "ctx_size": train_ds.ctx_size})
