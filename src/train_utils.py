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

def train_epoch(
    model,
    device,
    train_loader,
    loss_fn,
    optimizer,
    epoch,
    print_every_nep,
    train_acc_avgmeter,
    inner_scheduler=None,
    accelerator=None,
    quiet=False,
):
    train_acc_avgmeter.reset()
    model.train()
    for batch_idx, batched_datapoint in enumerate(train_loader):

        # NOTE: Data-specific! -->
        data, target_ = batched_datapoint
        target = th.flatten(target_, start_dim=2, end_dim=3).transpose(-1, -2)
        # <--

        if accelerator is None:
            data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)

        if accelerator is None:
            loss.backward()
        else:
            accelerator.backward(loss)

        optimizer.step()
        if inner_scheduler is not None:
            inner_scheduler.step()

        train_acc_avgmeter.update(loss.item())

        if not quiet and batch_idx % print_every_nep == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage {}: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    train_acc_avgmeter.name,
                    train_acc_avgmeter.avg,
                )
            )


def test(model, device, test_loader, loss_fn, test_acc_avgmeter, quiet=False):

    test_acc_avgmeter.reset()
    model.eval()
    with th.no_grad():

        for batched_datapoint in test_loader:
            # NOTE: Data-specific! -->
            data, target_ = batched_datapoint
            target = th.flatten(target_, start_dim=2, end_dim=3).transpose(-1, -2)
            # <--

            data, target = data.to(device), target.to(device)
            output = model(data)

            loss = loss_fn(output, target)
            test_acc_avgmeter.update(loss.item())

    if not quiet:
        print("Average {}: {:.4f}".format(test_acc_avgmeter.name, test_acc_avgmeter.avg))
    return test_acc_avgmeter.avg
