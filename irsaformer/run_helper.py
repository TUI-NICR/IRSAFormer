# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Tuple

import torch
from torchmetrics import MeanMetric

from nicr_mt_scene_analysis.data import infer_batch_size
from nicr_mt_scene_analysis.data import move_batch_to_device
from nicr_mt_scene_analysis.task_helper.token_based import TokenMatchingCache

from irsaformer.data import cast_to_dtype
from irsaformer.loss_weighting import get_loss_weighting_module
from irsaformer.task_helper import TokenTaskHelperType


class RunHelper:

    def __init__(
        self,
        args,
        model: torch.nn.Module,
        task_helpers: Tuple[TokenTaskHelperType],
        matching_cache: TokenMatchingCache,
        optimizer_steps_per_epoch: int,
        optimizer_total_steps: int,
        device: torch.device,
        move_model_to_device: bool = True,
    ) -> None:
        super().__init__()
        self.args = args

        if move_model_to_device:
            self.model = model.to(device)
        else:
            self.model = model

        self._task_helpers = task_helpers
        for task_helper in self._task_helpers:
            task_helper.initialize(device)
        self._matching_cache = matching_cache

        self._device = device
        self._validation_best_metrics_cache = {}
        self._accumulated_step_metrics = {}

        self._loss_weighting_module = get_loss_weighting_module(args)
        self._loss_amp = torch.float32
        if args.loss_amp == "fp16":
            self._loss_amp = torch.float16
        elif args.loss_amp == "bfp16":
            self._loss_amp = torch.bfloat16
        self._steps_per_epoch = max(1, int(optimizer_steps_per_epoch))
        self._total_steps = max(1, int(optimizer_total_steps))
        self._current_step = 0

    def reset(self) -> None:
        self._loss_weighting_module.reset_weights()
        self._validation_best_metrics_cache = {}
        self._accumulated_step_metrics = {}

    def _update_accumulated_step_metrics(self, logs, batch_size):
        metrics = self._accumulated_step_metrics
        for key, value in logs.items():
            if key not in metrics:
                metrics[key] = MeanMetric().to(self._device)
            metrics[key].update(value, weight=batch_size)

    def set_training_mode(self) -> None:
        torch.set_grad_enabled(True)
        self.model.train()

    def set_inference_mode(self) -> None:
        torch.set_grad_enabled(False)
        self.model.eval()

    def set_optimizer_step_from_epoch(self, epoch: int) -> None:
        self._current_step = int(self._steps_per_epoch * epoch)

    def _update_mask_attention_controller(self, progress: float) -> None:
        self.model.update_mask_attention_controller(progress)

    def training_step(self, batch, batch_idx):
        assert self.model.training
        progress = min(
            1.0,
            max(0.0, self._current_step / max(1, self._total_steps - 1))
        )
        self._update_mask_attention_controller(progress)

        batch = move_batch_to_device(batch, device=self._device)
        predictions_post = self.model(batch, do_postprocessing=True)

        losses = {}
        logs = {}

        if self.args.decoder_amp != self.args.loss_amp:
            predictions_post = cast_to_dtype(predictions_post, self._loss_amp)

        with torch.amp.autocast(
            device_type=self.args.device,
            enabled=self._loss_amp != torch.float32,
            dtype=self._loss_amp,
        ):
            for task_helper in self._task_helpers:
                task_loss_dict, task_logs = task_helper.training_step(
                    batch=batch,
                    batch_idx=batch_idx,
                    predictions_post=predictions_post,
                )
                losses.update(task_loss_dict)
                logs.update(task_logs)
            loss = self._loss_weighting_module.reduce_losses(losses, batch_idx)
        self._matching_cache.clear(batch_idx)

        logs["total_loss"] = loss.detach().clone()

        self._update_accumulated_step_metrics(
            logs={f"train_{key}": value for key, value in logs.items()},
            batch_size=infer_batch_size(batch),
        )
        return loss

    def on_optimizer_step(self) -> None:
        self._current_step += 1

    def training_get_artifacts_and_metrics(self):
        artifacts, metrics = {}, {}
        for key, metric in self._accumulated_step_metrics.items():
            if "train" not in key:
                continue
            metrics[key] = metric.compute()
            metric.reset()
        return artifacts, metrics

    def validation_step(self, batch, batch_idx):
        assert not self.model.training
        progress = min(
            1.0,
            max(0.0, self._current_step / max(1, self._total_steps - 1))
        )
        self._update_mask_attention_controller(progress)
        batch = move_batch_to_device(batch, device=self._device)
        predictions_post = self.model(batch, do_postprocessing=True)

        losses = {}
        logs = {}
        for task_helper in self._task_helpers:
            task_loss_dict, task_logs = task_helper.validation_step(
                batch=batch,
                batch_idx=batch_idx,
                predictions_post=predictions_post,
            )
            losses.update(task_loss_dict)
            logs.update(task_logs)
        self._matching_cache.clear(batch_idx)

        loss = self._loss_weighting_module.reduce_losses(losses, batch_idx)
        logs["total_loss"] = loss.detach().clone()

        self._update_accumulated_step_metrics(
            logs={f"valid_{key}": value for key, value in logs.items()},
            batch_size=infer_batch_size(batch),
        )
        return loss, predictions_post

    def validation_get_artifacts_examples_metrics(self):
        artifacts, examples, metrics = {}, {}, {}
        for key, metric in self._accumulated_step_metrics.items():
            if "valid" not in key:
                continue
            metrics[key] = metric.compute()
            metric.reset()

        for task_helper in self._task_helpers:
            task_result = task_helper.validation_epoch_end()
            task_artifacts, task_examples, task_logs = task_result
            metrics.update({
                f"valid_{key}": value for key, value in task_logs.items()
            })
            artifacts.update({
                f"valid_{key}": value
                for key, value in task_artifacts.items()
            })
            examples.update({
                f"valid_{key}": value
                for key, value in task_examples.items()
            })

        def force_tensor(value):
            if isinstance(value, torch.Tensor):
                return value
            return torch.tensor(value)

        cache = self._validation_best_metrics_cache
        for key in metrics:
            if any(m in key for m in ("miou", "acc", "rq", "sq", "pq")):
                fn = torch.greater
                default = torch.tensor(-torch.inf)
            elif "mae" in key or "rmse" in key:
                fn = torch.less
                default = torch.tensor(torch.inf)
            else:
                continue
            key_best = f"{key}_best"
            value_cur = metrics[key]
            value_best = cache.get(key_best, default)
            if fn(force_tensor(value_cur), force_tensor(value_best)).item():
                cache[key_best] = value_cur

        metrics.update(cache)
        return artifacts, examples, metrics
