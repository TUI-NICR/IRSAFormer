# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from nicr_mt_scene_analysis.model.attention_controller import EOMTAttentionMaskController
from nicr_mt_scene_analysis.model.backbone import get_token_backbone
from nicr_mt_scene_analysis.model.token_encoder import get_token_encoder
from .data import DatasetConfig
from .data import DatasetConfigWithAuxiliary
from .data import cast_to_dtype
from .decoder import get_token_based_decoders


def _get_token_backbone_kwargs(
    backbone_name: str,
    mlp_ratio: float,
    n_attention_heads: Optional[int]
) -> Dict[str, Any]:
    kwargs = {}
    if n_attention_heads is not None:
        kwargs['num_heads'] = n_attention_heads

    # DINOv3 is EVA-based in timm, but still accepts the ViT kwargs here.
    # Only actual EVA02 backbone names skip mlp_ratio.
    if 'eva02' not in backbone_name.lower():
        kwargs['mlp_ratio'] = mlp_ratio
    return kwargs


class IRSAFormer(torch.nn.Module):
    def __init__(
        self,
        args,
        dataset_configs: Dict[str, DatasetConfig],
    ) -> None:
        super().__init__()

        # store args and dataset parameters
        self.args = args

        # use first config as the main one
        self.dataset_config = next(iter(dataset_configs.values()))
        dataset_config = self.dataset_config
        
        # store auxiliary embeddings if available
        self.semantic_text_embeddings = None
        self.mean_visual_embeddings_per_semantic_class = None
        self.scene_text_embeddings = None
        self.mean_image_embedding_per_scene_class = None

        # get some dataset properties
        semantic_labels = dataset_config.semantic_label_list_without_void
        semantic_n_classes = len(semantic_labels)
        semantic_labels_with_void = dataset_config.semantic_label_list
        semantic_n_classes_with_void = len(semantic_labels_with_void)
        text_embeddings_per_semantic_class = None
        mean_visual_embedding_per_semantic_class = None

        semantic_n_classes_per_dataset = {
            ds_name: len(ds_conf.semantic_label_list_without_void)
            for ds_name, ds_conf in dataset_configs.items()
        }
        scene_n_classes_per_dataset = {
            ds_name: len(ds_conf.scene_label_list_without_void)
            for ds_name, ds_conf in dataset_configs.items()
        }
        scene_n_classes = len(dataset_config.scene_label_list_without_void)
        # TODO: still needed? Can be retrieved from meta in batch?
        panoptic_semantic_classes_is_thing = tuple(
            semantic_labels.classes_is_thing
        )
        self._semantic_n_classes_without_void = semantic_n_classes
        self._semantic_n_classes_with_void = semantic_n_classes_with_void
        # only available with auxiliary data
        if isinstance(dataset_config, DatasetConfigWithAuxiliary):
            text_embeddings_per_semantic_class = (
                dataset_config.semantic_text_embeddings
            )
            if len(text_embeddings_per_semantic_class) <= 0:
                # If dataset was loaded without data, an empty list will be
                # returned. This makes problem with calls to partial_class
                # later on which is why we set to None.
                text_embeddings_per_semantic_class = None
            # determine a tensor of shape (n_classes, embedding_dim) which
            # contains the text embeddings per semantic class
            if text_embeddings_per_semantic_class is not None:
                # With void!
                assert (
                    len(text_embeddings_per_semantic_class)
                    == semantic_n_classes + 1
                )
                text_embeddings_per_semantic_class = torch.tensor(
                    np.array(text_embeddings_per_semantic_class),
                    dtype=torch.float32
                )
                text_embeddings_per_semantic_class = (
                    text_embeddings_per_semantic_class[1:]
                )
                text_embeddings_per_semantic_class = (
                    text_embeddings_per_semantic_class.to(args.device)
                )
                self.semantic_text_embeddings = (
                    text_embeddings_per_semantic_class
                )

            text_embeddings_per_scene_class = (
                dataset_config.scene_text_embeddings
            )
            if len(text_embeddings_per_scene_class) <= 0:
                # If dataset was loaded without data, an empty list will be
                # returned. This makes problem with calls to partial_class
                # later on which is why we set to None.
                text_embeddings_per_scene_class = None
            # determine a tensor of shape (n_classes, embedding_dim) which
            # contains the text embeddings per scene class
            if text_embeddings_per_scene_class is not None:
                n_text = len(text_embeddings_per_scene_class)
                if n_text == scene_n_classes + 1:
                    # drop void
                    text_embeddings_per_scene_class = (
                        text_embeddings_per_scene_class[1:]
                    )
                elif n_text != scene_n_classes:
                    raise ValueError(
                        "Scene text embeddings mismatch: "
                        f"got {n_text}, expected {scene_n_classes} "
                        f"(or {scene_n_classes + 1} with void)."
                )
                text_embeddings_per_scene_class = torch.tensor(
                    np.array(text_embeddings_per_scene_class),
                    dtype=torch.float32
                ).to(args.device)
                self.scene_text_embeddings = text_embeddings_per_scene_class

            # compute mean visual embedding per semantic class and adjust
            # embedding with image_embedding to account for scene context.
            # shape is (n_classes, embedding_dim).
            # the resulting tensor can be used to determine the semantic
            # classes based on the predicted visual embedding
            mean_embedding_per_semantic_class = (
                dataset_config.mean_embedding_per_semantic_class
            )
            mean_image_embedding_per_semantic_class = (
                dataset_config.mean_image_embedding_per_semantic_class
            )
            if all((
                mean_embedding_per_semantic_class is not None,
                mean_image_embedding_per_semantic_class is not None
            )):
                mean_visual_embedding_per_semantic_class = torch.zeros_like(
                    text_embeddings_per_semantic_class
                ) if text_embeddings_per_semantic_class is not None else None

                if mean_visual_embedding_per_semantic_class is not None:
                    for semantic_idx, semantic_embedding in (
                        mean_embedding_per_semantic_class.items()
                    ):
                        mean_image_embedding = torch.from_numpy(
                            mean_image_embedding_per_semantic_class[
                                semantic_idx
                            ]
                        )
                        semantic_embedding = torch.from_numpy(
                            semantic_embedding
                        )

                        semantic_embedding_with_adjusted_scene_context = (
                            semantic_embedding
                            - args.token_visual_embedding_diff_factor
                            * mean_image_embedding
                        )
                        semantic_embedding_with_adjusted_scene_context /= (
                            semantic_embedding_with_adjusted_scene_context.norm(
                                dim=-1, keepdim=True
                            )
                        )
                        # -1 because we do not have embedding for void class
                        mean_visual_embedding_per_semantic_class[
                            semantic_idx - 1
                        ] = (
                            semantic_embedding_with_adjusted_scene_context
                        )
                    self.mean_visual_embeddings_per_semantic_class = \
                        mean_visual_embedding_per_semantic_class

            # normalize mean image embeddings per scene class and convert to
            # tensor otherwise we have problems with partial_class later on
            mean_embedding_per_scene_class_dict = None
            if 'mean_image_embedding_per_scene_class' in dir(dataset_config):
                mean_embedding_per_scene_class_dict = \
                    dataset_config.mean_image_embedding_per_scene_class
            if mean_embedding_per_scene_class_dict:
                embedding_dim = next(iter(
                    mean_embedding_per_scene_class_dict.values()
                )).shape[-1]
                mean_image_embedding_per_scene_class = torch.zeros(
                    (scene_n_classes, embedding_dim), dtype=torch.float32
                )
                for scene_class, mean_image_embedding in (
                    mean_embedding_per_scene_class_dict.items()
                ):
                    mean_image_embedding = torch.from_numpy(
                        mean_image_embedding
                    )
                    mean_image_embedding /= mean_image_embedding.norm(
                        dim=-1, keepdim=True
                    )
                    mean_image_embedding_per_scene_class[
                        scene_class - 1
                    ] = mean_image_embedding
                mean_image_embedding_per_scene_class = (
                    mean_image_embedding_per_scene_class.to(args.device)
                )
                self.mean_image_embedding_per_scene_class = (
                    mean_image_embedding_per_scene_class
                )

        # create encoder(s)
        backbone_rgb = None
        backbone_depth = None
        backbone_rgbd = None
        if 'rgb' in args.input_modalities:
            rgb_backbone_kwargs = _get_token_backbone_kwargs(
                args.rgb_encoder_backbone,
                args.rgb_encoder_mlp_ratio,
                args.rgb_encoder_num_attention_heads
            )
            backbone_rgb = get_token_backbone(
                name=args.rgb_encoder_backbone,
                n_input_channels=3,
                pretrained=not args.no_pretrained_backbone,
                pretrained_filepath=(
                    args.rgb_encoder_backbone_pretrained_weights_filepath
                ),
                **rgb_backbone_kwargs,
            )
            backbone_rgb.set_input_size(
                args.input_height,
                args.input_width,
                patch_height=args.input_patch_height,
                patch_width=args.input_patch_width
            )
        if 'depth' in args.input_modalities:
            depth_backbone_kwargs = _get_token_backbone_kwargs(
                args.depth_encoder_backbone,
                args.depth_encoder_mlp_ratio,
                args.depth_encoder_num_attention_heads
            )
            backbone_depth = get_token_backbone(
                name=args.depth_encoder_backbone,
                n_input_channels=1,
                pretrained=not args.no_pretrained_backbone,
                pretrained_filepath=(
                    args.depth_encoder_backbone_pretrained_weights_filepath
                ),
                **depth_backbone_kwargs,
            )
            backbone_depth.set_input_size(
                args.input_height,
                args.input_width,
                patch_height=args.input_patch_height,
                patch_width=args.input_patch_width
            )
        if 'rgbd' in args.input_modalities:
            rgbd_backbone_kwargs = _get_token_backbone_kwargs(
                args.rgbd_encoder_backbone,
                args.rgbd_encoder_mlp_ratio,
                args.rgbd_encoder_num_attention_heads
            )
            backbone_rgbd = get_token_backbone(
                name=args.rgbd_encoder_backbone,
                n_input_channels=3+1,
                pretrained=not args.no_pretrained_backbone,
                pretrained_filepath=(
                    args.rgbd_encoder_backbone_pretrained_weights_filepath
                ),
                **rgbd_backbone_kwargs,
            )
            backbone_rgbd.set_input_size(
                args.input_height,
                args.input_width,
                patch_height=args.input_patch_height,
                patch_width=args.input_patch_width
            )
        assert any([
            backbone_rgb is not None,
            backbone_depth is not None,
            backbone_rgbd is not None,
        ])
        self._encoder_backbone_refs = {
            modality: backbone
            for modality, backbone in (
                ('rgb', backbone_rgb),
                ('depth', backbone_depth),
                ('rgbd', backbone_rgbd),
            )
            if backbone is not None
        }
        token_modality = self.args.token_modality
        token_backbone = self._encoder_backbone_refs[token_modality]

        # Public args select transformer blocks. The token encoder addresses
        # stages, where stage 0 is the patch embedding and stage i maps to
        # transformer block i - 1.
        reference_backbone = backbone_rgb or backbone_rgbd or backbone_depth
        n_stages = len(reference_backbone.stages)
        self._n_stages = n_stages
        n_transformer_blocks = n_stages - 1
        token_start_block = self.args.token_injection_start_block
        # Allow negative selections.
        if token_start_block < 0:
            token_start_block = n_transformer_blocks + token_start_block
        if not 0 <= token_start_block < n_transformer_blocks:
            raise ValueError(
                "`--token-injection-start-block` must select an existing "
                "transformer block."
            )

        extra_tokens_start_stage_idx = token_start_block + 1
        extra_tokens_end_stage_idx = n_stages - 1
        n_injection_blocks = (
            extra_tokens_end_stage_idx - extra_tokens_start_stage_idx + 1
        )
        block_schedule = self.args.token_attention_mask_block_schedule
        if (
            block_schedule is not None
            and len(block_schedule) != n_injection_blocks
        ):
            raise ValueError(
                "`--token-attention-mask-block-schedule` must match the "
                f"number of injected blocks ({n_injection_blocks})."
            )
        skip_stage_indices: list[int] = []
        if (
            not self.args.no_multistage_supervision
            and n_injection_blocks > 0
        ):
            first_aux_idx = max(extra_tokens_start_stage_idx - 1, 0)
            skip_stage_indices = list(
                range(first_aux_idx, extra_tokens_end_stage_idx + 1)
            )
        encoder_skip_stage_indices = tuple(skip_stage_indices)
        self.multistage_stage_indices = tuple(skip_stage_indices)

        fusion_stage_indices = None
        if backbone_rgb is not None and backbone_depth is not None:
            if self.args.encoder_fusion_stage_indices is not None:
                fusion_stage_indices = tuple(
                    self.args.encoder_fusion_stage_indices
                )
            elif self.args.encoder_fusion_stage_interval is not None:
                interval = int(self.args.encoder_fusion_stage_interval)
                fusion_stage_indices = tuple(
                    range(interval, n_stages, interval)
                )

        # fuse encoder(s) in a shared module
        self.encoder = get_token_encoder(
            backbone_rgb=backbone_rgb,
            backbone_depth=backbone_depth,
            backbone_rgbd=backbone_rgbd,
            fusion=args.encoder_fusion,
            skip_downsamplings=encoder_skip_stage_indices,
            fusion_stage_indices=fusion_stage_indices,
            extra_tokens_start_stage_idx=extra_tokens_start_stage_idx,
            extra_tokens_end_stage_idx=extra_tokens_end_stage_idx
        )
        self._extra_tokens_start_stage_idx = extra_tokens_start_stage_idx
        self._extra_tokens_end_stage_idx = extra_tokens_end_stage_idx

        # build modality-specific query embeddings
        # 0 queries is valid for token-scene (the scene head reads globally
        # pooled backbone features and never consumes the query tokens). mask
        # and panoptic tasks still need queries.
        assert self.args.token_num_queries >= 0
        token_embed_dim = token_backbone.embed_dim

        # select tokens for token-based decoder based on modality
        # however, in theory it should be possible for all modalities
        self._embed_dim = token_embed_dim
        self.token_query_modality = token_modality
        self.token_query_embedding = nn.Embedding(
            num_embeddings=self.args.token_num_queries,
            embedding_dim=token_embed_dim,
        )

        # compute the number of learned x2 mask upsampling blocks so the mask
        # features reach at least roughly 1/4 input resolution. Patch size 14
        # therefore uses two blocks, same as patch size 16. The mask decoder
        # still resizes predictions to the exact patch-grid size for
        # non-power-of-two patches.
        meta = token_backbone.backbone_meta()
        patch_h, patch_w = meta['patch_size']
        max_patch = max(patch_h, patch_w)
        mask_n_upsampling_blocks = max(
            1,
            math.ceil(math.log2(max_patch / 4))
        )

        # the embedding decoders need to know the width of the external
        # reference embeddings they will be matched against (e.g. CLIP-style
        # vectors from the dataset's auxiliary data). only compute it when
        # an embedding task is enabled, and fail loudly if no reference is
        # available rather than silently using the ViT token dim.
        reference_embedding_dim = None
        if any(t in args.tasks
               for t in ('token-visual-embedding', 'token-image-embedding')):
            refs = (
                self.semantic_text_embeddings,
                self.mean_visual_embeddings_per_semantic_class,
                self.scene_text_embeddings,
                self.mean_image_embedding_per_scene_class,
            )
            ref = next((r for r in refs if r is not None), None)
            if ref is None:
                raise RuntimeError(
                    "Token embedding tasks require external reference "
                    "embeddings from the dataset's auxiliary data; none of "
                    "{semantic,scene}_text_embeddings or "
                    "mean_{visual,image}_embedding_per_{semantic,scene}_class "
                    "are available."
                )
            reference_embedding_dim = ref.shape[-1]

        self.decoders = get_token_based_decoders(
            token_dim=token_embed_dim,
            modality=token_modality,
            semantic_n_classes=semantic_n_classes,
            semantic_classes_is_thing=panoptic_semantic_classes_is_thing,
            auxiliary_stage_indices=self.multistage_stage_indices or None,
            mask_n_upsampling_blocks=mask_n_upsampling_blocks,
            mask_upsampling_mode=args.mask_decoder_upsampling,
            prediction_upsampling=args.upsampling_prediction,
            tasks=args.tasks,
            scene_n_classes=scene_n_classes,
            scene_n_classes_per_dataset=scene_n_classes_per_dataset,
            semantic_n_classes_per_dataset=semantic_n_classes_per_dataset,
            token_panoptic_mask_threshold=args.token_panoptic_mask_threshold,
            token_panoptic_mask_pixel_threshold=(
                args.token_panoptic_mask_pixel_threshold
            ),
            token_panoptic_overlap_threshold=(
                args.token_panoptic_overlap_threshold
            ),
            token_visual_embedding_mask_threshold=(
                args.token_visual_embedding_mask_threshold
            ),
            token_visual_embedding_mask_pixel_threshold=(
                args.token_visual_embedding_mask_pixel_threshold
            ),
            token_visual_embedding_overlap_threshold=(
                args.token_visual_embedding_overlap_threshold
            ),
            token_visual_embedding_query_mask_score_exponent=(
                args.token_visual_embedding_query_mask_score_exponent
            ),
            token_visual_embedding_similarity_score_scale=(
                args.token_visual_embedding_similarity_score_scale
            ),
            reference_embedding_dim=reference_embedding_dim,
            mean_image_embeddings_per_scene_class=(
                self.mean_image_embedding_per_scene_class
            ),
            scene_text_embeddings=self.scene_text_embeddings,
            mean_visual_embeddings_per_semantic_class=(
                self.mean_visual_embeddings_per_semantic_class
            ),
            semantic_text_embeddings=self.semantic_text_embeddings,
            token_visual_embedding_linear_probing=True,
            token_visual_embedding_mlp_hidden_dim=(
                args.token_visual_embedding_mlp_hidden_dim
            ),
            token_visual_embedding_mlp_dropout=(
                args.token_visual_embedding_mlp_dropout
            ),
        )
        self.mask_attention_controller = self._build_mask_attention_controller()

        # Setup amp (automatic mixed precision)
        self.encoder_amp = torch.float32
        if args.encoder_amp == 'fp16':
            self.encoder_amp = torch.float16
        elif args.encoder_amp == 'bfp16':
            self.encoder_amp = torch.bfloat16
        else:
            assert args.encoder_amp == 'disabled'

        self.decoder_amp = torch.float32
        if args.decoder_amp == 'fp16':
            self.decoder_amp = torch.float16
        elif args.decoder_amp == 'bfp16':
            self.decoder_amp = torch.bfloat16
        else:
            assert args.decoder_amp == 'disabled'

    def compile(self):
        # Compile the model if requested. We only compile individual parts
        # instead of the whole model as if we compile the whole model, we
        # get some compiler issues.
        # Raise the dynamo cache so the per-stage_idx graph specialization in
        # `forward_stage` does not hit the default recompile limit of 8.
        torch._dynamo.config.cache_size_limit = max(64, self._n_stages + 16)
        self.encoder = torch.compile(self.encoder)
        compiled_decoders = nn.ModuleDict()
        for name, decoder in self.decoders.items():
            compiled_decoders[name] = torch.compile(decoder)
        self.decoders = compiled_decoders
        # Rebuild the mask attention controller so it references the freshly
        # compiled decoder modules (shared mask head must stay in sync).
        self.mask_attention_controller = self._build_mask_attention_controller()

    def forward(self, batch, do_postprocessing=False) -> Dict[str, Any]:
        # Setup encoder amp
        with torch.amp.autocast(
            device_type=self.args.device,
            enabled=self.encoder_amp != torch.float32 and self.training,
            dtype=self.encoder_amp if self.training else torch.float32
        ):
            # determine input
            encoder_inputs = {}
            if 'rgbd' in self.args.input_modalities:
                rgb = batch['rgb']
                depth = batch['depth']
                rgbd = torch.cat([rgb, depth], dim=1)
                encoder_inputs['rgbd'] = rgbd
            else:
                if 'rgb' in self.args.input_modalities:
                    encoder_inputs['rgb'] = batch['rgb']
                if 'depth' in self.args.input_modalities:
                    encoder_inputs['depth'] = batch['depth']
            assert len(encoder_inputs) > 0

            # materialize modality-specific token queries
            batch_size = next(iter(encoder_inputs.values())).shape[0]
            extra_tokens = {}
            tokens = self.token_query_embedding.weight[None, :, :].expand(
                batch_size, -1, -1
            )
            extra_tokens[self.token_query_modality] = tokens

            encoder_outputs = self.encoder(
                encoder_inputs,
                extra_tokens=extra_tokens,
                extra_tokens_controller=self.mask_attention_controller,
            )

        assert len(encoder_outputs) == 3
        enc_features, enc_dec_skips, token_sequences = encoder_outputs

        if self.encoder_amp != self.decoder_amp:
            enc_features = cast_to_dtype(enc_features, self.decoder_amp)
            enc_dec_skips = cast_to_dtype(enc_dec_skips, self.decoder_amp)
            token_sequences = cast_to_dtype(token_sequences, self.decoder_amp)

        token_input = (enc_features, enc_dec_skips, token_sequences)

        # Setup decoder amp
        with torch.amp.autocast(
            device_type=self.args.device,
            enabled=self.decoder_amp != torch.float32 and self.training,
            dtype=self.decoder_amp if self.training else torch.float32
        ):
            # forward decoder(s)
            if do_postprocessing:
                outputs = {}
                for decoder in self.decoders.values():
                    outputs = decoder(
                        token_input, enc_dec_skips, batch,
                        do_postprocessing=do_postprocessing,
                        out_dict=outputs
                    )
            else:
                outputs = []
                for decoder in self.decoders.values():
                    outputs.append(
                        decoder(
                            token_input, enc_dec_skips, batch,
                            do_postprocessing=do_postprocessing
                        )
                    )
        # TODO: Should we return outputs with mixed precision, or should we
        # cast everything to float32?

        return outputs

    def get_backbone_parameters(
        self,
        modality: str
    ) -> Tuple[nn.Parameter, ...]:
        return self._encoder_backbone_refs[modality].parameters()

    def _build_mask_attention_controller(
        self
    ) -> Optional[EOMTAttentionMaskController]:
        mode = self.args.token_attention_mask_mode
        if mode == 'none':
            # No special attention control. The encoder uses plain attention.
            return

        if 'mask_decoder' not in self.decoders:
            raise RuntimeError(
                "Mask attention requires a token mask decoder."
            )

        stage_indices = list(range(
            self._extra_tokens_start_stage_idx,
            self._extra_tokens_end_stage_idx + 1
        ))
        schedule_windows = self.args.token_attention_mask_block_schedule
        if schedule_windows is None:
            schedule_windows = tuple((0.0, 1.0) for _ in stage_indices)
        else:
            schedule_windows = tuple(
                (float(start), float(end)) for start, end in schedule_windows
            )
        stage_schedule = {}
        for stage_idx, (start, end) in zip(stage_indices, schedule_windows):
            stage_schedule[stage_idx] = (float(start), float(end))

        if mode == 'eomt':
            return EOMTAttentionMaskController(
                modality=self.token_query_modality,
                mask_logits_fn=(
                    self.decoders['mask_decoder'].compute_mask_logits
                ),
                start_stage_idx=self._extra_tokens_start_stage_idx,
                end_stage_idx=self._extra_tokens_end_stage_idx,
                stage_schedule=stage_schedule or None,
            )
        else:
            raise ValueError('Only eomt implemented for now.')

    def update_mask_attention_controller(self, progress: float) -> None:
        if self.mask_attention_controller is None:
            return
        self.mask_attention_controller.update_progress(progress)
