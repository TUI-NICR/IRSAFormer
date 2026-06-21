# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import os
import random
from typing import Tuple

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    value = int(seed)
    os.environ["PYTHONHASHSEED"] = str(value)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def compute_optimizer_step_counts(
    args,
    train_steps_per_epoch: int
) -> Tuple[int, int]:
    n_epochs = max(1, int(args.n_epochs))
    steps_per_epoch = max(1, int(train_steps_per_epoch))
    accum_steps = max(1, int(args.gradient_accumulation_steps))
    optimizer_steps_per_epoch = max(
        1,
        (steps_per_epoch + accum_steps - 1) // accum_steps,
    )
    optimizer_total_steps = max(1, n_epochs * optimizer_steps_per_epoch)
    return optimizer_steps_per_epoch, optimizer_total_steps
