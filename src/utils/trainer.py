# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import importlib
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import SGD, Adam, RMSprop, lr_scheduler


def get_scheduler(cfg_opt, opt):
    """Return the scheduler object.

    Args:
        cfg_opt (obj): Config for the specific optimization module (gen/dis).
        opt (obj): PyTorch optimizer object.

    Returns:
        (obj): Scheduler
    """
    if cfg_opt.lr_policy.type == 'step':
        scheduler = lr_scheduler.StepLR(
            opt,
            step_size=cfg_opt.lr_policy.step_size,
            gamma=cfg_opt.lr_policy.gamma)
    elif cfg_opt.lr_policy.type == 'constant':
        scheduler = lr_scheduler.LambdaLR(opt, lambda x: 1)
    elif cfg_opt.lr_policy.type == 'linear':
        # Start linear decay from here.
        decay_start = cfg_opt.lr_policy.decay_start
        # End linear decay here.
        # Continue to train using the lowest learning rate till the end.
        decay_end = cfg_opt.lr_policy.decay_end
        # Lowest learning rate multiplier.
        decay_target = cfg_opt.lr_policy.decay_target

        def sch(x):
            return min(
                max(((x - decay_start) * decay_target + decay_end - x) / (
                    decay_end - decay_start
                ), decay_target), 1.
            )
        scheduler = lr_scheduler.LambdaLR(opt, lambda x: sch(x))
    else:
        return NotImplementedError('Learning rate policy {} not implemented.'.
                                   format(cfg_opt.lr_policy.type))
    return scheduler