#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
"""
import sys
import os
sys.path.append(os.getcwd())

from typing import Dict
from copy import deepcopy
from datetime import datetime
import json
import os
from pprint import pprint
import shlex
import sys
from time import time
import traceback
import warnings

import numpy as np
import PIL.Image
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
from tqdm import tqdm as tqdm_
import wandb

from nicr_mt_scene_analysis.checkpointing import CheckpointHelper
from nicr_mt_scene_analysis.logging import CSVLogger
from nicr_mt_scene_analysis.testing.onnx import export_onnx_model
from nicr_mt_scene_analysis.task_helper.token_based import TokenMatchingCache
from nicr_mt_scene_analysis.utils import cprint
from nicr_mt_scene_analysis.utils import cprint_step
from nicr_scene_analysis_datasets import ConcatDataset
from irsaformer.args import ArgParserIRSAFormer
from irsaformer.data import DatasetConfig
from irsaformer.data import DatasetConfigWithAuxiliary
from irsaformer.data import get_datahelper
from irsaformer.lr_scheduler import get_lr_scheduler
from irsaformer.model import IRSAFormer
from irsaformer.optimizer import get_optimizer
from irsaformer.optimizer_groups import build_optimizer_param_groups
from irsaformer.optimizer_groups import describe_param_groups
from irsaformer.preprocessing import get_preprocessor
from irsaformer.task_helper import get_task_helpers
from irsaformer.visualization import setup_shared_color_generators
from irsaformer.visualization import visualize
from irsaformer.weights import load_weights
from irsaformer.run_helper import RunHelper
from irsaformer.runtime import compute_optimizer_step_counts
from irsaformer.runtime import seed_everything


def main():
    # Args & General Stuff -----------------------------------------------------
    parser = ArgParserIRSAFormer()
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.seed is not None:
        print(f"Setting global seed to {args.seed}.")
        seed_everything(args.seed)

    if args.enable_tf32:
        # Enables TensorFloat32 for matmul operations if supported by the
        # hardware for faster training.
        torch.set_float32_matmul_precision('high')

    disable_progress_bars = args.disable_progress_bars
    if disable_progress_bars:
        # dummy tqdm function that only prints the description and step number
        def tqdm(obj, **kwargs):
            if 'desc' in kwargs:
                print(kwargs['desc'],
                      f"({kwargs.get('total', 'unknown number of')} steps)")
            return obj
    else:
        # use tqdm
        tqdm = tqdm_

    if args.dataset_path is None:
        raise ValueError(
            "Please provide `--dataset-path` for the selected dataset(s)."
        )

    # prepare results paths
    if not args.is_resumed_training:
        starttime = datetime.now().strftime('%Y_%m_%d-%H_%M_%S-%f')
        results_path = os.path.abspath(os.path.join(
            args.results_basepath,
            '_debug_runs' if args.debug else '',
            args.dataset.replace(':', '+'),
            f'run_{starttime}'
        ))
    else:
        # write results to same folder as in previous training
        results_path = args.resume_path
    os.makedirs(results_path, exist_ok=args.is_resumed_training)
    artifacts_path = os.path.join(results_path, 'artifacts')
    os.makedirs(artifacts_path, exist_ok=args.is_resumed_training)
    checkpoints_path = os.path.join(results_path, 'checkpoints')
    os.makedirs(checkpoints_path, exist_ok=args.is_resumed_training)
    examples_path = os.path.join(results_path, 'examples')
    os.makedirs(examples_path, exist_ok=args.is_resumed_training)
    print(f"Writing results to '{results_path}'.")

    # append some information to args
    args.results_path = results_path
    args.artifacts_path = artifacts_path
    args.checkpoints_path = checkpoints_path
    args.examples_path = examples_path
    args.start_timestamp = int(time())

    wandb_run = None

    if not args.validation_only:
        # set up wandb

        # convert tuples/lists to let them appear in parallel coordinate plots
        w_args = deepcopy(args)
        for k, v in dict(vars(w_args)).items():
            if isinstance(v, (list, tuple)):
                v_str = ', '.join(str(v_) for v_ in v)
                if not isinstance(v[0], str):
                    # prepend 's ' to make sure wandb handles it correctly
                    v_str = f's {v_str}'
                setattr(w_args, f'{k}_str', v_str)

        if args.wandb_mode != 'disabled':
            wandb_run = wandb.init(
                dir=results_path,
                entity='nicr',
                config=w_args,
                mode=args.wandb_mode,
                project=args.wandb_project,
                settings=wandb.Settings(start_method='fork')
            )
            # set epoch as default x axis
            wandb_run.define_metric('epoch')
            wandb_run.define_metric("*", step_metric='epoch', step_sync=True)

            # append some information to args
            args.wandb_name = wandb_run.name
            args.wandb_id = wandb_run.id
            args.wandb_url = wandb_run.url

        # dump args ------------------------------------------------------------
        if not args.is_resumed_training:
            # argv only if not resuming
            with open(os.path.join(args.results_path, 'argsv.txt'), 'w') as f:
                f.write(shlex.join(sys.argv))
                f.write('\n')

        with open(os.path.join(results_path, 'args.json'), 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)

    # Data & Model -------------------------------------------------------------
    cprint_step("Get model and dataset")
    # get datahelper
    data = get_datahelper(args)

    optimizer_steps_per_epoch, optimizer_total_steps = \
        compute_optimizer_step_counts(args, len(data.train_dataloader))

    if args.weights_filepath is not None:
        args.no_pretrained_backbone = True

    def _dataset_name(ds):
        meta = ds.load('meta', 0)
        # Auxiliary datasets store the wrapped original dataset class in meta.
        # Plain datasets do not need this entry. Their own class is the type.
        dataset_type = meta.get('dataset_type', ds.__class__)
        return dataset_type.__name__

    dataset_configs: Dict[str, DatasetConfig | DatasetConfigWithAuxiliary] = {}
    if isinstance(data.train_dataloader.dataset, ConcatDataset):
        for ds in data.train_dataloader.dataset.datasets:
            dataset_configs[_dataset_name(ds)] = ds.config
    else:
        dataset_configs = {
            _dataset_name(data.dataset_train): data.dataset_config
        }

    # get model
    model = IRSAFormer(args, dataset_configs=dataset_configs)

    # load weights (account for renamed or missing keys, specific dataset
    # combinations, pretraining configurations)
    checkpoint_epoch = None
    if args.weights_filepath is not None:
        print(f"Loading (pretrained) weights from: '{args.weights_filepath}'.")
        checkpoint = torch.load(args.weights_filepath,
                                map_location=torch.device('cpu'))
        if 'epoch' in checkpoint:
            print(f"-> Epoch: {checkpoint['epoch']}")
            checkpoint_epoch = int(checkpoint['epoch'])
        if args.debug and 'logs' in checkpoint:
            print("-> Logs/Metrics:")
            pprint(checkpoint['logs'])

        state_dict = checkpoint['state_dict']
        # `_delta` marks a checkpoint that stores `(trained - pretrained)`
        # backbone tensors instead of the trained values themselves. when
        # present, load_weights re-fetches the upstream timm backbone and
        # adds it on top before the strict load.
        delta_meta = checkpoint.get('_delta')
        load_weights(args, model, state_dict, verbose=True,
                     delta_meta=delta_meta)

    if args.compile_model:
        model.compile()

    model = model.to(device)
    data.set_train_preprocessor(
        get_preprocessor(
            args,
            dataset=data.dataset_train,
            phase='train'
        )
    )

    data.set_valid_preprocessor(
        get_preprocessor(
            args,
            dataset=data.datasets_valid[0],
            phase='test'
        )
    )

    # export onnx model to be able to debug the model's structure
    if args.debug:
        cprint_step("Export ONNX model")
        # use 'EXPORT_ONNX_MODELS=true python ...' to export the model
        from torch.onnx import TrainingMode

        # get some valid data
        batch = next(iter(data.train_dataloader))
        batch = {k: v for k, v in batch.items() if torch.is_tensor(v)}
        fp = os.path.join(results_path, 'model.onnx')
        # TODO: export for Dropout2D (feature_dropout) to enable mode PRESERVE
        if export_onnx_model(fp, model, (batch, {}),
                             training_mode=TrainingMode.EVAL,
                             force_export=False,
                             use_fallback=True,
                             opset_version=18):
            print(f"Wrote ONNX model to '{fp}'.")
        else:
            print("Export skipped. Set `EXPORT_ONNX_MODELS=true` to enable.")

    if not args.validation_only and args.overfit_n_batches > 0:
        # force overfitting (training+validation) to overfit_n_batches batches
        # of the valid set
        data.enable_overfitting_mode(n_valid_batches=args.overfit_n_batches)
        optimizer_steps_per_epoch, optimizer_total_steps = \
            compute_optimizer_step_counts(args, len(data.train_dataloader))

    # Training Stuff -----------------------------------------------------------
    # logging (note, appends to existing metrics file)
    csv_logger = None
    csv_logger = CSVLogger(filepath=os.path.join(results_path, 'metrics.csv'),
                           write_interval=1)

    # optimizer and lr scheduler
    param_groups = build_optimizer_param_groups(model, args)
    if args.debug_print_optimizer_groups:
        print(describe_param_groups(
            param_groups, title="Optimizer groups (IRSAFormer)"
        ))
    optimizer = get_optimizer(args, param_groups)
    lr_scheduler = get_lr_scheduler(
        args,
        optimizer,
        total_steps=optimizer_total_steps,
    )
    scaler = None
    if set([args.encoder_amp, args.decoder_amp]) != {'disabled'}:
        # Enable scaler to improve numerical stability
        scaler = torch.amp.GradScaler()

    # get task helper
    matching_cache = TokenMatchingCache()
    task_helpers = tuple(
        get_task_helpers(args, data.dataset_train, matching_cache)
    )

    # wrap model in run helper
    run = RunHelper(
        args,
        model=model,
        task_helpers=task_helpers,
        matching_cache=matching_cache,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        optimizer_total_steps=optimizer_total_steps,
        device=device,
        move_model_to_device=True
    )

    # check for resumed training
    if args.resume_ckpt_filepath is not None:
        cprint_step("Resume training")
        checkpoint = torch.load(args.resume_ckpt_filepath,
                                map_location=torch.device('cpu'))
        print(f"Checkpoint: '{args.resume_ckpt_filepath}'")
        next_epoch = checkpoint['epoch'] + 1
        print(f"Last epoch: {checkpoint['epoch']}, next epoch: {next_epoch}")
        print("Replacing state dicts for model, optimizer, and lr scheduler.")

        model_state = checkpoint.get('state_dict')
        if model_state is not None:
            _load_model_state_dict(args, model, model_state)
        optimizer_state = checkpoint.get('optimizer')
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        if 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if scaler is not None and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
    else:
        # training starts from scratch
        next_epoch = 0

    # checkpointing
    if args.checkpointing_metrics is None:
        warnings.warn(
            "No checkpoints will be saved. Please provide the metrics by which "
            "you want to checkpoint the model weights with "
            "`--checkpointing-metrics`."
        )
    checkpoint_helper = CheckpointHelper(
        metric_names=args.checkpointing_metrics,
        debug=True    # args.debug
    )

    # Simple Sanity Check ------------------------------------------------------
    if not args.skip_sanity_check:
        # ensure that crucial parts (data, forward, metrics, ...) are working
        # as expected, the check is done by forwarding a single batch of all
        # dataloaders WITHOUT backpropagation.
        cprint_step("Perform sanity check")

        # disable forward stats tracking (e.g., batchnorm)
        for m in model.modules():
            if hasattr(m, 'track_running_stats'):
                m.track_running_stats = False

        # check training (single batch)
        batch = next(iter(data.train_dataloader))
        assert isinstance(run.training_step(batch, 0), torch.Tensor)
        assert run.training_get_artifacts_and_metrics()

        # re-enable forward stats tracking (e.g., batchnorm)
        for m in model.modules():
            if hasattr(m, 'track_running_stats'):
                m.track_running_stats = True

        # check validation (single batch for all valid sets)
        run.set_inference_mode()
        for valid_dataloader in data.valid_dataloaders:
            batch = next(iter(valid_dataloader))
            validation_result, _ = run.validation_step(batch, 0)
            assert isinstance(validation_result, torch.Tensor)
        result = run.validation_get_artifacts_examples_metrics()  # also resets
        assert result

        # check metrics for checkpointing
        artifacts, examples, metrics = result
        for ckpt_metric in args.checkpointing_metrics or []:
            assert checkpoint_helper._determine_checkpoint_metrics(
                ckpt_metric, metrics
            )

        # reset run helper states (loss weighting module and metric caches)
        run.reset()

        # everything seems to work
        print("Fine.")

    # Validation ---------------------------------------------------------------
    if args.validation_only:
        cprint_step("Run validation only")

        if args.visualize_validation:
            print("Writing visualizations to: "
                  f"'{args.visualization_output_path}'.")
            os.makedirs(args.visualization_output_path, exist_ok=True)

            # dump args
            with open(os.path.join(args.visualization_output_path,
                                   'args.json'), 'w') as f:
                json.dump(vars(args), f, sort_keys=True, indent=4)
            with open(os.path.join(args.visualization_output_path,
                                   'argsv.txt'), 'w') as f:   # should be argv
                f.write(shlex.join(sys.argv))
                f.write('\n')

        # use shared color generators to ensure consistent colors and to speed
        # up visualization
        setup_shared_color_generators(data.dataset_train.config)

        run.set_inference_mode()
        # Validation needs a schedule position for mask-attention annealing.
        # If the checkpoint has no epoch metadata, assume a final checkpoint.
        validation_schedule_epoch = checkpoint_epoch
        if validation_schedule_epoch is None:
            validation_schedule_epoch = args.n_epochs - 1
        run.set_optimizer_step_from_epoch(validation_schedule_epoch)
        batch_idx = 0
        for i, valid_dataloader in enumerate(data.valid_dataloaders):
            tqdm_desc = f'Validation {i+1}/{len(data.valid_dataloaders)}'
            tqdm_desc += f' ({valid_dataloader.dataset.camera})'
            for batch in tqdm(valid_dataloader,
                              total=len(valid_dataloader),
                              desc=tqdm_desc):
                _, predictions = run.validation_step(batch, batch_idx)
                if args.visualize_validation:
                    output_path = os.path.join(
                        args.visualization_output_path,
                        args.validation_split.replace(':', '+')
                    )
                    visualize(
                        output_path=output_path,
                        batch=batch,
                        predictions=predictions,
                        dataset_config=data.dataset_train.config
                    )

                batch_idx += 1

        # get and print validation metrics
        _, _, metrics = run.validation_get_artifacts_examples_metrics()
        metrics = _to_float_dict(metrics)
        print("Validation results:")
        pprint(metrics)
        filepath = os.path.join(results_path, 'validation_metrics.json')
        with open(filepath, 'w') as f:
            json.dump(metrics, f, sort_keys=True, indent=2)

        # stop here
        return

    # Training -----------------------------------------------------------------
    cprint_step("Start training")
    # training loop
    grad_clip_norm = max(
        0.0,
        float(getattr(args, "gradient_clip_norm", 0.0) or 0.0)
    )
    try:
        for epoch in range(next_epoch, args.n_epochs):
            run.set_optimizer_step_from_epoch(epoch)
            cprint(f"Epoch: {epoch:04d}/{args.n_epochs-1:04d}",
                   color='cyan', attrs=('bold',))
            epoch_logs = {'epoch': epoch}
            wandb_examples = {}

            # training
            run.set_training_mode()
            train_dataloader = data.train_dataloader
            num_train_batches = len(train_dataloader)
            grad_accum_steps = max(1, int(args.gradient_accumulation_steps))
            n_forwards_done = 0
            n_forwards_per_step = 0
            for batch_idx, batch in tqdm(
                    enumerate(train_dataloader),
                    total=num_train_batches,
                    desc='Training'):
                if n_forwards_done == 0:
                    optimizer.zero_grad(set_to_none=True)
                    # Last accumulation block may be shorter.
                    n_forwards_per_step = min(
                        grad_accum_steps,
                        num_train_batches - batch_idx
                    )
                loss = run.training_step(batch, batch_idx)
                n_forwards_done += 1
                loss_to_backward = loss / n_forwards_per_step
                if scaler is not None:
                    scaler.scale(loss_to_backward).backward()
                else:
                    loss_to_backward.backward()
                if n_forwards_done < n_forwards_per_step:
                    continue
                if scaler is not None:
                    if grad_clip_norm > 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), grad_clip_norm
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if grad_clip_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), grad_clip_norm
                        )
                    optimizer.step()
                run.on_optimizer_step()
                n_forwards_done = 0
                lr_scheduler.step()

            # get training metrics
            _, metrics = run.training_get_artifacts_and_metrics()
            metrics = _to_float_dict(metrics)
            epoch_logs.update(metrics)
            current_global_step = min(
                optimizer_total_steps,
                (epoch + 1) * optimizer_steps_per_epoch,
            )
            epoch_logs['global_step'] = current_global_step
            lr_values = lr_scheduler.get_last_lr()
            epoch_logs['lr'] = lr_values[0] if lr_values else None
            for idx, group in enumerate(optimizer.param_groups):
                group_name = group.get('group_name', f'group_{idx}')
                key = f"lr_group_{group_name}"
                if key in epoch_logs:
                    key = f"{key}_{idx}"
                epoch_logs[key] = group.get('lr')
            controller = model.mask_attention_controller
            if controller is not None:
                epoch_logs["attn_mask_progress"] = float(controller._progress)
                start_idx = int(controller._start_stage_idx)
                end_idx = int(controller._end_stage_idx)
                stage_indices = list(range(start_idx, end_idx + 1))
                for idx, stage_idx in enumerate(stage_indices):
                    if stage_idx not in controller._stage_schedule:
                        continue
                    epoch_logs[f"attn_mask_prob_{idx}"] = float(
                        controller._probability_for_stage(stage_idx)
                    )

            # validation
            force = False
            if args.validation_force_interval is not None:
                # force validation at given interval
                force = ((epoch + 1) % args.validation_force_interval) == 0
            if (epoch + 1) == args.n_epochs:
                # it is the last epoch, force validation
                force = True

            if ((epoch + 1) >= (args.n_epochs * args.validation_skip)) or force:
                artifacts: Dict[str, torch.Tensor] = {}
                examples = {}
                metrics = {}
                do_create_checkpoint = {}

                run.set_inference_mode()
                if args.visualize_validation:
                    setup_shared_color_generators(data.dataset_train.config)
                    visualization_root = (
                        args.visualization_output_path
                        or os.path.join(results_path, 'visualization')
                    )
                    visualization_epoch = os.path.join(
                        visualization_root, f'epoch_{epoch:04d}'
                    )
                    os.makedirs(visualization_epoch, exist_ok=True)
                # we have multiple valid datasets due to multiple resolutions
                batch_idx = 0
                for i, valid_dataloader in enumerate(data.valid_dataloaders):
                    if isinstance(valid_dataloader.dataset,
                                  torch.utils.data.Subset):
                        # overfitting mode (dataset is wrapped using Subset)
                        camera = valid_dataloader.dataset.dataset.camera
                    else:
                        camera = valid_dataloader.dataset.camera
                    tqdm_desc = (f'Validation {i+1}/'
                                 f'{len(data.valid_dataloaders)} ({camera})')
                    for batch in tqdm(valid_dataloader,
                                      total=len(valid_dataloader),
                                      desc=tqdm_desc):
                        if args.visualize_validation:
                            _, predictions = run.validation_step(
                                batch, batch_idx
                            )
                            output_path = os.path.join(
                                visualization_epoch,
                                args.validation_split.replace(':', '+')
                            )
                            visualize(
                                output_path=output_path,
                                batch=batch,
                                predictions=predictions,
                                dataset_config=data.dataset_train.config
                            )
                        else:
                            _ = run.validation_step(batch, batch_idx)
                        batch_idx += 1

                # get validation artifacts and metrics
                artifacts_all, examples_all, metrics = \
                    run.validation_get_artifacts_examples_metrics()
                metrics = _to_float_dict(metrics)
                epoch_logs.update(metrics)

                artifacts = artifacts_all
                examples = examples_all
                # checkpointing
                do_create_checkpoint = checkpoint_helper.check_for_checkpoint(
                    logs=epoch_logs,
                    add_checkpoint_metrics_to_logs=True
                )

                pending_checkpoint = any(do_create_checkpoint.values())
                state_dict = model.state_dict() if pending_checkpoint else None

                if epoch >= (args.n_epochs * args.checkpointing_skip) or force:
                    for ckpt_metric, create_checkpoint in (
                        do_create_checkpoint.items()
                    ):
                        if not create_checkpoint:
                            # no new best value, skip checkpointing
                            continue

                        # create new checkpoint
                        if args.checkpointing_best_only:
                            suffix = '_best'
                        else:
                            suffix = f'_epoch_{epoch:04d}'

                        mapped_name = \
                            checkpoint_helper.metric_mapping_joined[ckpt_metric]
                        ckpt_filepath = os.path.join(
                            checkpoints_path, f'ckpt_{mapped_name}{suffix}.pth')
                        # save checkpoint
                        if state_dict is None:
                            continue
                        ckpt = {
                            'state_dict': state_dict,
                            'epoch': epoch,
                            'logs': epoch_logs
                        }
                        if scaler is not None:
                            ckpt['scaler'] = scaler.state_dict()
                        torch.save(ckpt, ckpt_filepath)
                        print(f"Wrote checkpoint to: '{ckpt_filepath}'.")

                # store artifacts
                for key, value in artifacts.items():
                    fn = f'{key}__epoch_{epoch:04d}.npy'
                    if isinstance(value, torch.Tensor):
                        value = value.cpu().numpy()
                    np.save(os.path.join(artifacts_path, fn), value)

                # store / log examples
                for key, value in examples.items():
                    fn = f'{key}__epoch_{epoch:04d}'
                    if isinstance(value, PIL.Image.Image):
                        value.save(os.path.join(examples_path, fn+'.png'),
                                   'PNG')
                        if wandb_run is not None:
                            wandb_examples[key] = wandb.Image(value)

            # resume checkpoint
            if (
                ((epoch + 1) % args.resume_ckpt_interval) == 0 or
                ((epoch + 1) == (args.n_epochs))
            ):
                ckpt_filepath = os.path.join(checkpoints_path,
                                             'ckpt_resume.pth')

                model_state = model.state_dict()
                optimizer_state = optimizer.state_dict()

                ckpt = {
                    'state_dict': model_state,
                    'optimizer': optimizer_state,
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'logs': epoch_logs
                }
                if scaler is not None:
                    ckpt['scaler'] = scaler.state_dict()

                torch.save(ckpt, ckpt_filepath+'.tmp')
                if os.path.isfile(ckpt_filepath):
                    os.remove(ckpt_filepath)
                os.rename(ckpt_filepath+'.tmp', ckpt_filepath)

                print(f"Wrote resume checkpoint to: '{ckpt_filepath}'.")

            # logging
            if csv_logger is not None:
                csv_logger.log(epoch_logs)
            if wandb_run is not None:
                wandb_logs = {**epoch_logs, **wandb_examples}
                wandb_logs = dict(sorted(wandb_logs.items()))
                wandb.log(wandb_logs, commit=True)
            if args.debug:
                print("Epoch logs:")
                pprint(epoch_logs)

    except Exception:
        # something went wrong -.-
        # store checkpoint
        ckpt_filepath = os.path.join(checkpoints_path,
                                     f'ckpt_error__epoch_{epoch:04d}.pth')

        model_state = model.state_dict()
        optimizer_state = optimizer.state_dict()

        ckpt = {
            'state_dict': model_state,
            'optimizer': optimizer_state,
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'logs': epoch_logs
        }
        torch.save(ckpt, ckpt_filepath)
        print(f"Wrote checkpoint to: '{ckpt_filepath}'.")
        # log error
        log_filepath = os.path.join(results_path, 'error.log')
        with open(log_filepath, 'w') as f:
            traceback.print_exc(file=f)
        print(f"Wrote error log to: '{log_filepath}'.")

        # reraise error -> let the run crash
        raise

    # training done
    with open(os.path.join(results_path, 'finished'), 'w') as f:
        pass
    if csv_logger is not None:
        csv_logger.write()
    cprint_step("Done")


def _to_float_dict(metrics):
    float_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(
                    f"Metric '{key}' must be scalar, got shape {value.shape}."
                )
            float_metrics[key] = float(value.detach().cpu())
        elif isinstance(value, str):
            float_metrics[key] = value
        else:
            float_metrics[key] = float(value)
    return float_metrics


def _load_model_state_dict(args, model, state_dict):
    # preserves a tiny shape-adaptation hop for token-panoptic checkpoints
    # (some older checkpoints have a 1-off class head); falls back to the
    # plain strict load otherwise.
    if state_dict is not None and 'token-panoptic' in args.tasks:
        model_state = model.state_dict()
        for key, weight in list(state_dict.items()):
            if all(n in key for n in ('decoders.panoptic_decoder', 'head')):
                if (
                    key in model_state
                    and weight.shape != model_state[key].shape
                ):
                    target = model_state[key]
                    if (
                        weight.shape[0] + 1 == target.shape[0]
                        and weight.shape[1:] == target.shape[1:]
                    ):
                        merged = target.clone()
                        merged[1:weight.shape[0] + 1] = weight
                        state_dict[key] = merged
                    else:
                        state_dict[key] = target
    model.load_state_dict(state_dict)


if __name__ == '__main__':
    main()
