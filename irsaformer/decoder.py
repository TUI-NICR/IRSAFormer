# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from nicr_mt_scene_analysis.model.decoder import TokenMaskDecoder
from nicr_mt_scene_analysis.model.decoder import TokenPanopticDecoder
from nicr_mt_scene_analysis.model.decoder import TokenSemanticDecoder
from nicr_mt_scene_analysis.model.decoder import TokenSceneDecoder
from nicr_mt_scene_analysis.model.decoder import TokenOrientationDecoder
from nicr_mt_scene_analysis.model.decoder import TokenImageEmbeddingDecoder
from nicr_mt_scene_analysis.model.decoder import TokenEmbeddingDecoder
from nicr_mt_scene_analysis.model.postprocessing import get_postprocessing_class


def get_token_based_decoders(
    *,
    token_dim: int,
    modality: str,
    semantic_n_classes: int,
    semantic_classes_is_thing: Optional[Tuple[bool, ...]] = None,
    auxiliary_stage_indices: Optional[Tuple[int, ...]] = None,
    mask_n_upsampling_blocks: int,
    mask_upsampling_mode: str,
    prediction_upsampling: str,
    tasks: Sequence[str],
    scene_n_classes: int,
    scene_n_classes_per_dataset: dict[str, int],
    semantic_n_classes_per_dataset: Optional[dict[str, int]] = None,
    token_panoptic_mask_threshold: float = 0.1,
    token_panoptic_mask_pixel_threshold: float = 0.2,
    token_panoptic_overlap_threshold: float = 0.6,
    token_visual_embedding_mask_threshold: float = 0.0,
    token_visual_embedding_mask_pixel_threshold: float = 0.70,
    token_visual_embedding_overlap_threshold: float = 0.60,
    token_visual_embedding_query_mask_score_exponent: float = 0.75,
    token_visual_embedding_similarity_score_scale: float = 20.0,
    reference_embedding_dim: Optional[int] = None,
    mean_image_embeddings_per_scene_class: Optional[torch.Tensor] = None,
    scene_text_embeddings: Optional[torch.Tensor] = None,
    mean_visual_embeddings_per_semantic_class: Optional[torch.Tensor] = None,
    semantic_text_embeddings: Optional[torch.Tensor] = None,
    token_visual_embedding_linear_probing: bool = True,
    token_visual_embedding_mlp_hidden_dim: int = 1024,
    token_visual_embedding_mlp_dropout: float = 0.0,
) -> nn.ModuleDict:
    decoders = {}

    # token mask must be first so other token tasks can reuse the matching.
    if 'token-mask' in tasks:
        decoders['mask_decoder'] = TokenMaskDecoder(
            embed_dim=token_dim,
            n_upsampling_blocks=mask_n_upsampling_blocks,
            modality=modality,
            upsampling_mode=mask_upsampling_mode,
            prediction_upsampling=prediction_upsampling,
            upsample_side_outputs=False,
            side_output_stage_indices=auxiliary_stage_indices,
        )

    if 'token-semantic' in tasks:
        decoders['semantic_decoder'] = TokenSemanticDecoder(
            embed_dim=token_dim,
            modality=modality,
            n_classes=semantic_n_classes,
            side_output_stage_indices=auxiliary_stage_indices,
        )
    elif 'token-panoptic' in tasks:
        assert semantic_classes_is_thing is not None
        decoders['panoptic_decoder'] = TokenPanopticDecoder(
            embed_dim=token_dim,
            modality=modality,
            n_classes=semantic_n_classes,
            postprocessing=get_postprocessing_class(
                'token-panoptic',
                n_classes=semantic_n_classes,
                semantic_classes_is_thing=semantic_classes_is_thing,
                mask_threshold=token_panoptic_mask_threshold,
                mask_pixel_threshold=token_panoptic_mask_pixel_threshold,
                overlap_threshold=token_panoptic_overlap_threshold,
            ),
            side_output_stage_indices=auxiliary_stage_indices,
        )

    if 'token-scene' in tasks:
        decoders['scene_decoder'] = TokenSceneDecoder(
            embed_dim=token_dim,
            n_classes=scene_n_classes,
            modality=modality
        )

    if 'token-orientation' in tasks:
        decoders['orientation_decoder'] = TokenOrientationDecoder(
            embed_dim=token_dim,
            modality=modality,
        )

    if 'token-image-embedding' in tasks:

        dec = TokenImageEmbeddingDecoder(
            embed_dim=token_dim,
            embedding_dim=reference_embedding_dim,
            modality=modality,
            n_scene_classes_per_dataset=scene_n_classes_per_dataset,
            postprocessing=get_postprocessing_class(
                'token-image-embedding',
                mean_image_embeddings_per_scene_class=(
                    mean_image_embeddings_per_scene_class
                ),
                scene_text_embeddings=scene_text_embeddings,
            )
        )

        decoders['image_embedding_decoder'] = dec

    if 'token-visual-embedding' in tasks:
        linear_probe_classes = (
            semantic_n_classes
            if token_visual_embedding_linear_probing
            else None
        )
        linear_probe_classes_per_dataset = None
        if (
            token_visual_embedding_linear_probing
            and semantic_n_classes_per_dataset is not None
        ):
            linear_probe_classes_per_dataset = semantic_n_classes_per_dataset
        decoders['visual_embedding_decoder'] = TokenEmbeddingDecoder(
            embed_dim=token_dim,
            embedding_dim=reference_embedding_dim,
            modality=modality,
            linear_probing_classes=linear_probe_classes,
            linear_probing_classes_per_dataset=linear_probe_classes_per_dataset,
            mlp_hidden_dim=token_visual_embedding_mlp_hidden_dim,
            mlp_dropout=token_visual_embedding_mlp_dropout,
            postprocessing=get_postprocessing_class(
                'token-visual-embedding',
                text_embeddings_per_class=semantic_text_embeddings,
                mean_visual_embedding_per_class=(
                    mean_visual_embeddings_per_semantic_class
                ),
                semantic_classes_is_thing=semantic_classes_is_thing,
                mask_threshold=token_visual_embedding_mask_threshold,
                mask_pixel_threshold=(
                    token_visual_embedding_mask_pixel_threshold
                ),
                overlap_threshold=token_visual_embedding_overlap_threshold,
                query_mask_score_exponent=(
                    token_visual_embedding_query_mask_score_exponent
                ),
                similarity_score_scale=(
                    token_visual_embedding_similarity_score_scale
                ),
            ),
        )

    assert len(decoders) > 0

    return nn.ModuleDict(decoders)
