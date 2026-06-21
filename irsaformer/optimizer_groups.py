from typing import Dict, List, Sequence

import torch.nn as nn


def _get_patch_embed_params(
    backbone: nn.Module
) -> Dict[str, List[nn.Parameter]]:
    # Only multimodal ViT backbones expose separate RGB/depth patch projections
    if not hasattr(backbone, "model"):
        return {}
    if not hasattr(backbone.model, "patch_embed"):
        return {}
    if not hasattr(backbone.model.patch_embed, "proj"):
        return {}

    patch_embedder = backbone.model.patch_embed.proj
    if (
        not hasattr(patch_embedder, "rgb_proj")
        or not hasattr(patch_embedder, "depth_proj")
    ):
        return {}

    return {
        "rgb": list(patch_embedder.rgb_proj.parameters()),
        "depth": list(patch_embedder.depth_proj.parameters()),
    }


def build_optimizer_param_groups(model, args) -> List[Dict]:
    all_params = list(model.parameters())

    base_lr = args.learning_rate
    param_groups: List[Dict] = []
    seen_ids = set()

    lr_mult_map = {
        "rgb": args.rgb_encoder_backbone_lr_mult,
        "depth": args.depth_encoder_backbone_lr_mult,
        "rgbd": args.rgbd_encoder_backbone_lr_mult,
    }
    for modality in args.input_modalities:
        lr_mult = lr_mult_map[modality]
        params_all = list(model.get_backbone_parameters(modality))
        base_group_name = f"backbone_{modality}"
        separated_ids = set()

        # RGB-D ViT patch embedding has separate RGB/depth projections. Put the
        # depth projection into its own LR group and keep all remaining backbone
        # parameters in the regular RGB-D backbone group.
        if modality == "rgbd":
            patch_embed_params = _get_patch_embed_params(
                model._encoder_backbone_refs[modality]
            )
            depth_patch_params = patch_embed_params.get("depth", [])

            depth_lr_mult = args.rgbd_encoder_backbone_depth_embed_lr_mult

            param_groups.append(
                {
                    "params": depth_patch_params,
                    "lr": base_lr * lr_mult * depth_lr_mult,
                    "group_name": f"{base_group_name}_depth_patch_embed",
                }
            )
            ids = {id(p) for p in depth_patch_params}
            separated_ids.update(ids)
            seen_ids.update(ids)

        residual = [p for p in params_all if id(p) not in separated_ids]
        if residual:
            param_groups.append(
                {
                    "params": residual,
                    "lr": base_lr * lr_mult,
                    "group_name": base_group_name,
                }
            )
            seen_ids.update(id(p) for p in residual)

    remaining = [p for p in all_params if id(p) not in seen_ids]
    if remaining:
        param_groups.append(
            {
                "params": remaining,
                "lr": base_lr,
                "group_name": "non_backbone",
            }
        )

    return param_groups


def describe_param_groups(
    param_groups: Sequence[Dict],
    title: str = "Optimizer parameter groups",
) -> str:
    lines = [title]
    for idx, group in enumerate(param_groups):
        name = group['group_name']
        lr = group['lr']
        count = sum(p.numel() for p in group["params"])
        lines.append(f"  [{idx:02d}] {name}: lr={lr:.6e} | params={count:,}")
    return "\n".join(lines)
