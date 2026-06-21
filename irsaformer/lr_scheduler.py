# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Iterable
from typing import List
from typing import Tuple

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import OneCycleLR

KNOWN_LR_SCHEDULERS = ('onecycle', 'warmup_poly')


LrSchedulerType = LRScheduler


class WarmupPolyScheduler(LRScheduler):
    def __init__(
        self,
        optimizer,
        configs: Iterable[Tuple[int, int]],
        total_steps: int,
        power: float
    ) -> None:
        self._configs: List[Tuple[int, int]] = [
            (int(start), int(warmup)) for start, warmup in configs
        ]
        self._total_steps = max(1, int(total_steps))
        self._power = float(power)
        assert len(self._configs) == len(optimizer.param_groups)
        super().__init__(optimizer)

    def get_lr(self):
        step = max(0, self.last_epoch)
        max_step = self._total_steps
        lrs = []
        for base_lr, (start_cfg, warmup_cfg) in zip(
            self.base_lrs, self._configs
        ):
            start = max(0, min(start_cfg, max_step))
            warmup = max(0, warmup_cfg)
            decay_window = max(1, max_step - start - warmup)
            if step < start:
                lr = 0.0
            elif (warmup > 0 and step < start + warmup):
                lr = base_lr * (step - start) / warmup
            else:
                elapsed = min(step, max_step) - start - warmup
                progress = max(0.0, min(1.0, elapsed / decay_window))
                lr = base_lr * (1.0 - progress) ** self._power
            lrs.append(max(lr, 0.0))
        return lrs


def _get_group_start_and_warmup(group_name: str, args) -> Tuple[float, float]:
    if group_name.startswith('backbone_rgb'):
        start_frac = args.rgb_encoder_backbone_lr_start_frac
        warmup_frac = args.rgb_encoder_backbone_lr_warmup_frac
    elif group_name.startswith('backbone_depth'):
        start_frac = args.depth_encoder_backbone_lr_start_frac
        warmup_frac = args.depth_encoder_backbone_lr_warmup_frac
    elif group_name.startswith('backbone_rgbd'):
        start_frac = args.rgbd_encoder_backbone_lr_start_frac
        warmup_frac = args.rgbd_encoder_backbone_lr_warmup_frac
    else:
        start_frac = args.non_backbone_lr_start_frac
        warmup_frac = args.non_backbone_lr_warmup_frac

    start_frac = float(start_frac)
    warmup_frac = float(warmup_frac)
    assert 0.0 <= start_frac <= 1.0
    assert 0.0 <= warmup_frac <= 1.0

    return start_frac, warmup_frac


def get_lr_scheduler(args, optimizer, total_steps: int) -> LrSchedulerType:
    name = args.learning_rate_scheduler.lower()
    assert name in KNOWN_LR_SCHEDULERS

    total_steps = max(1, int(total_steps))

    if name == 'onecycle':
        return OneCycleLR(
            optimizer,
            max_lr=[pg['lr'] for pg in optimizer.param_groups],
            total_steps=total_steps,
            div_factor=25,
            pct_start=0.1,
            anneal_strategy='cos',
            final_div_factor=1e4
        )

    assert name == 'warmup_poly'
    configs: List[Tuple[int, int]] = []
    for group in optimizer.param_groups:
        group_name = group['group_name']
        start_frac, warmup_frac = _get_group_start_and_warmup(group_name, args)
        start_steps = int(round(total_steps * start_frac))
        warmup_steps = int(round(total_steps * warmup_frac))
        configs.append((max(0, start_steps), max(0, warmup_steps)))

    return WarmupPolyScheduler(
        optimizer=optimizer,
        configs=configs,
        total_steps=total_steps,
        power=args.lr_warmup_poly_power
    )
