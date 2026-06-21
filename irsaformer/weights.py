# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
import torch

from nicr_mt_scene_analysis.model.backbone import get_token_backbone
from nicr_scene_analysis_datasets import ScanNet


def _infer_backbone_input_channels(state_dict, prefix):
    # Standard timm patch embeddings store one projection tensor. Our RGB-D
    # patch embedding stores separate RGB and depth projections instead.
    proj_w = state_dict.get(prefix + 'patch_embed.proj.weight')
    rgb_w = state_dict.get(prefix + 'patch_embed.proj.rgb_proj.weight')
    depth_w = state_dict.get(prefix + 'patch_embed.proj.depth_proj.weight')
    if proj_w is not None:
        return int(proj_w.shape[1])
    if rgb_w is not None and depth_w is not None:
        return int(rgb_w.shape[1]) + int(depth_w.shape[1])
    if rgb_w is not None:
        return int(rgb_w.shape[1])
    if depth_w is not None:
        return int(depth_w.shape[1])
    raise RuntimeError(
        f"Cannot infer n_input_channels for prefix '{prefix}'; "
        "no patch_embed.proj{,.rgb_proj,.depth_proj}.weight found.")


def load_weights(args, model, state_dict, verbose=True, delta_meta=None):
    # this function accounts for:
    # - missing keys, e.g., different number of scene classes
    # - specific dataset or pretraining combinations, e.g., Hypersim -> SUNRGB-D
    # - delta checkpoints (e.g. DINOv3): when `delta_meta` is given, the 
    #   backbone tensors in `state_dict` are stored as `(trained - pretrained)`
    #   and must be re-added on top of the upstream pretrained weights before
    #   the strict load.

    print_ = print if verbose else lambda *a, **k: None

    # get current model state dict
    model_state_dict = model.state_dict()

    # if the model was compiled, the keys contain .orig_mod which need
    # to be removed from state dict to load uncompiled model
    state_dict = {
        k.replace('_orig_mod.', ''): v
        for k, v in state_dict.items()
    }

    if len(state_dict) != len(model_state_dict):
        # loaded state dict is different, run a deeper analysis
        # this can happen if a model trained with deviating tasks is loaded
        # (e.g., pre-training on hypersim with normals)
        # we try to remove the extra keys
        for key in list(state_dict.keys()):
            if key not in model_state_dict:
                print_(f"Removing '{key}' from loaded state dict as the "
                       "current model does not contain such key.")
                _ = state_dict.pop(key)

    # scene classes may differ, e.g., when using pretrained weights on
    # Hypersim for a subsequent training, we skip loading these pretrained
    # weights
    for key in list(state_dict.keys()):
        if all(n in key for n in ('scene_decoder', 'head')):
            n_classes_pretraining = model_state_dict[key].shape[0]
            n_classes_current = state_dict[key].shape[0]
            if n_classes_current != n_classes_pretraining:
                print_(f"Skipping '{key}' as the number of scene classes "
                       f"differs {n_classes_current} (current) vs. "
                       f"{n_classes_pretraining} (pretraining).")
                # we simply use the random weights of the current model
                state_dict[key] = model_state_dict[key]

    semantic_like_head_prefixes = []
    if 'token-semantic' in args.tasks:
        semantic_like_head_prefixes.append('semantic_decoder')
    if 'token-panoptic' in args.tasks:
        semantic_like_head_prefixes.append('panoptic_decoder')

    if semantic_like_head_prefixes:
        if args.dataset.startswith('sunrgbd'):  # first (main) dataset
            # sunrgbd has only 37 semantic classes, however these classes
            # match the first 37 classes of nyuv2, scannet and hypersim
            # (40 classes), so, if we detect weights with 40 output
            # channels (filter and bias) in a semantic head, we keep the
            # first 37 channels
            for key, weight in list(state_dict.items()):
                if any(prefix in key and 'head' in key
                       for prefix in semantic_like_head_prefixes):
                    if weight.shape[0] == 40:
                        print_(f"Removing last 3 channels in '{key}'.")
                        state_dict[key] = weight[:37, ...]

        elif args.dataset.startswith('scannet'):  # first (main) dataset
            # check if training (e.g., pretraining on hypersim) was done
            # with more classes, we can handle two cases 40 -> 20 and
            # 549 -> 200
            if not args.validation_scannet_benchmark_mode:
                # otherwise, we already would have 20 / 200 classes

                # get mapping and mask
                if 20 == args.scannet_semantic_n_classes:
                    mapping = ScanNet.SEMANTIC_CLASSES_40_MAPPING_TO_BENCHMARK
                else:
                    mapping = (
                        ScanNet.SEMANTIC_CLASSES_549_MAPPING_TO_BENCHMARK200
                    )

                mask = torch.tensor([
                    c_benchmark != 0   # class is not ignored
                    for c_data, c_benchmark in mapping.items()
                    if c_data != 0     # skip void class
                ], dtype=torch.bool)

                # check weights of semantic heads and remove ignored classes
                for key, weight in list(state_dict.items()):
                    if any(prefix in key and 'head' in key
                           for prefix in semantic_like_head_prefixes):
                        if weight.shape[0] == mask.shape[0]:
                            print_("Removing channels for ignored classes "
                                   f"in '{key}'.")
                            state_dict[key] = weight[mask, ...]

        # remove all semantic weights if shape still does not match,
        # happens, e.g., when using pretrained weights from scannet with
        # 20 classes for sunrbgd or nyuv2
        for key, weight in list(state_dict.items()):
            if any(prefix in key and 'head' in key
                   for prefix in semantic_like_head_prefixes):
                if weight.shape != model_state_dict[key].shape:
                    print_(f"Removing '{key}' from loaded state dict as"
                           f"the shape does not match: {weight.shape} "
                           f"vs. {model_state_dict[key].shape}.")
                    # we simply use the random weights of the current
                    # model
                    state_dict[key] = model_state_dict[key]

    if args.dataset.startswith('scannet'):
        if 20 == args.scannet_semantic_n_classes:
            mapping = ScanNet.SEMANTIC_CLASSES_40_MAPPING_TO_BENCHMARK
        elif 200 == args.scannet_semantic_n_classes:
            mapping = ScanNet.SEMANTIC_CLASSES_549_MAPPING_TO_BENCHMARK200
        else:
            mapping = None

        if mapping is not None:
            mask = torch.tensor([
                c_benchmark != 0
                for c_data, c_benchmark in mapping.items()
                if c_data != 0
            ], dtype=torch.bool)

            lp_prefixes = (
                'decoders.visual_embedding_decoder._linear_probing_head',
                'decoders.image_embedding_decoder._linear_probing_head',
            )
            for prefix in lp_prefixes:
                for suffix in ('weight', 'bias'):
                    key = f'{prefix}.ScanNet.{suffix}'
                    if key not in state_dict or key not in model_state_dict:
                        continue
                    weight = state_dict[key]
                    expected = model_state_dict[key]
                    if weight.shape == expected.shape:
                        continue
                    if weight.shape[0] == mask.shape[0]:
                        print_("Removing channels for ignored classes "
                               f"in '{key}'.")
                        state_dict[key] = weight[mask, ...]
                        weight = state_dict[key]
                    if weight.shape != expected.shape:
                        print_(f"Removing '{key}' from loaded state dict as "
                               f"the shape does not match: {weight.shape} "
                               f"vs. {expected.shape}.")
                        state_dict[key] = expected

    if delta_meta is not None:
        # delta checkpoint: backbone tensors are stored as (trained -
        # pretrained). fetch the matching upstream backbone(s) via timm and
        # add their values on top of the deltas before the strict load. this
        # reconstruct the full checkpoint weights by reversing the release
        # delta conversion.
        backbone_name = delta_meta.get('backbone_name')
        if not backbone_name:
            raise RuntimeError(
                "Delta checkpoint metadata missing 'backbone_name'.")
        delta_prefixes = tuple(delta_meta.get('delta_prefixes', ()))
        if not delta_prefixes:
            raise RuntimeError(
                "Delta checkpoint metadata missing 'delta_prefixes'; cannot "
                "recover backbone weights.")

        n_added = 0
        for prefix in delta_prefixes:
            prefix_keys = [k for k in state_dict if k.startswith(prefix)]
            if not prefix_keys:
                continue
            n_input_channels = _infer_backbone_input_channels(
                state_dict, prefix
            )
            print_(f"Fetching upstream '{backbone_name}' "
                   f"(n_input_channels={n_input_channels}) for prefix "
                   f"'{prefix}'.")
            upstream = get_token_backbone(
                name=backbone_name,
                n_input_channels=n_input_channels,
                pretrained=True,
            )
            upstream_sd = upstream.state_dict()
            # the wrapper exposes the timm model under self.model, so its
            # state_dict prefixes inner tensors with 'model.'. our delta keys
            # had `prefix` stripped, so prepend that back when looking up.
            for key in prefix_keys:
                sub_key = 'model.' + key[len(prefix):]
                if sub_key not in upstream_sd:
                    raise RuntimeError(
                        f"Delta key '{key}' has no upstream counterpart "
                        f"(looked up '{sub_key}').")
                state_dict[key] = upstream_sd[sub_key] + state_dict[key]
                n_added += 1
            del upstream, upstream_sd
        print_(f"Loaded delta checkpoint: re-added {n_added} backbone "
               f"tensors from upstream '{backbone_name}'.")

    model.load_state_dict(state_dict, strict=True)
