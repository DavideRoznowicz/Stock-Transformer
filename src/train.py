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
from torch.nn import MSELoss, ModuleList, Identity, L1Loss, Tanh
from torch.optim.lr_scheduler import MultiStepLR

from data import stock_dataloader_dispatcher
from architectures import StockTransformerModel

from ebtorch.optim import Lookahead, RAdam
from ebtorch.logging import AverageMeter
from ebtorch.nn import Mish, mishlayer_init

from train_utils import train_epoch, test

from accelerate import Accelerator

from torchinfo import summary

# DATA:
train_loader, test_loader, totr_loader, data_props = stock_dataloader_dispatcher(
    data_path="../data/",
    which_financial=(range(97)),    # <-- Memory-bound
    which_contextual=(0, 1, 2),     # Whole, Year, Month, Whatever
    time_lookback=30,               # Reasonable: 30
    time_predict=5,                 # Almost surely in [5, 10]
    window_stride=1,                # Different from 1 does not make sense!
    ttsr=0.9,                       # Reasonably in [0.5, 0.9]
    train_bs=32,                    # Start at 32 (Luschi e Masters)
    test_bs=512,                    # Default: 512
    shuffle_train=False             # Tame overfitting to our advantage! :)
)

# MODEL PARAMETERS:
A = [
    (
        data_props.fin_size,
        (1, 250),
        (250, 3),
        (5, 5),
        (1, 3),
    ),
    {"batchnorm": True, "causal": False, "activation_fx": Mish()},
]
B = [([291, 100, 50], 16), {"batchnorm": True, "activation_fx": Mish()}]
C = [(310, 2, 512), {"activation": "gelu", "batch_first": True}]
D = [
    (),
    {"encoder_layer": "_", "num_layers": 3},
]
E = [([2480], 5*97), {"batchnorm": False, "activation_fx": Tanh()}]

F = data_props.ctx_size
G = data_props.fin_size

################################################################################

ACCELERATOR: bool = False
AUTODETECT: bool = True
DRY_VALIDATE: bool = False

nrepochs = 70

model = StockTransformerModel(A, B, C, D, E, F, G)

# Weights initialization
for submodel in (model.conv_featurizer, model.mlp_correlator, model.decoder):
    for layr in submodel.modules():
        mishlayer_init(layr)


if not ACCELERATOR:
    device = th.device("cuda" if th.cuda.is_available() and AUTODETECT else "cpu")
    model = model.to(device)
    accelerator = None
else:
    device = None
    accelerator = Accelerator()

#criterion = MSELoss(reduction="mean")
criterion = L1Loss(reduction="mean")
optimizer = RAdam(model.parameters(), lr=4e-3)

train_acc_avgmeter = AverageMeter("batchwise training loss")
test_acc_avgmeter = AverageMeter("epochwise testing loss")
totr_acc_avgmeter = AverageMeter("epochwise training loss")

base_optimizer = optimizer
optimizer = Lookahead(base_optimizer, la_steps=3)   # la_steps

# SCHEDULING:
sched_milestones=[2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
sched_gamma=0.5

if not isinstance(optimizer, Lookahead):
    scheduler = MultiStepLR(optimizer, milestones=sched_milestones, gamma=sched_gamma)
else:
    scheduler = MultiStepLR(base_optimizer, milestones=sched_milestones, gamma=sched_gamma)

if ACCELERATOR:
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

################################################################################

if DRY_VALIDATE:
    for _, dry_data in enumerate(train_loader):
        dry_x, dry_y_ = dry_data
        dry_y = th.flatten(dry_y_, start_dim=2, end_dim=3).transpose(-1, -2)
        break

    print("VALIDATING...")
    print(summary(model, input_data=dry_x))

else:
    for epoch in range(1, nrepochs + 1):

        # Training
        #print("TRAINING...")

        train_epoch(
            model=model,
            device=device,
            train_loader=train_loader,
            loss_fn=criterion,
            optimizer=optimizer,
            epoch=epoch,
            print_every_nep=15,
            train_acc_avgmeter=train_acc_avgmeter,
            inner_scheduler=None,
            accelerator=accelerator,
            quiet=False,
        )

        # Tweaks for the Lookahead optimizer (before testing)
        if isinstance(optimizer, Lookahead):
            optimizer._backup_and_load_cache()
        
        # Testing: on training and testing set
        print("\n")
        #print("TESTING...")
        #print("\nON TRAINING SET:")
        _ = test(model, device, totr_loader, criterion, totr_acc_avgmeter, quiet=False)
        #print("\nON TEST SET:")
        _ = test(model, device, test_loader, criterion, test_acc_avgmeter, quiet=False)
        print("\n\n\n")
        
        # Tweaks for the Lookahead optimizer (after testing)
        if isinstance(optimizer, Lookahead):
            optimizer._clear_and_load_backup()
        
        # Scheduling step (outer)
        scheduler.step()
