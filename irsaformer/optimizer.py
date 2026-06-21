# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Union

from torch.optim import Adam
from torch.optim import AdamW
from torch.optim import RAdam
from torch.optim import SGD


KNOWN_OPTIMIZERS = ('adam', 'adamw', 'radam', 'sgd')


OptimizerType = Union[Adam, AdamW, RAdam, SGD]


def get_optimizer(args, parameters) -> OptimizerType:
    name = args.optimizer
    lr = args.learning_rate
    weight_decay = args.weight_decay
    momentum = args.momentum

    name = name.lower()
    if name not in KNOWN_OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: '{name}'")

    if 'sgd' == name:
        optimizer = SGD(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=True
        )
    elif 'adam' == name:
        optimizer = Adam(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
    elif 'adamw' == name:
        optimizer = AdamW(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
    elif 'radam' == name:
        optimizer = RAdam(
            parameters,
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
        )

    return optimizer
