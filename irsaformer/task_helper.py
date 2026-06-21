# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Tuple

from nicr_mt_scene_analysis.task_helper import TokenTaskHelperType
from nicr_mt_scene_analysis.task_helper.token_based import TokenImageEmbeddingTaskHelper
from nicr_mt_scene_analysis.task_helper.token_based import TokenEmbeddingTaskHelper
from nicr_mt_scene_analysis.task_helper.token_based import TokenMaskTaskHelper
from nicr_mt_scene_analysis.task_helper.token_based import TokenMatchingCache
from nicr_mt_scene_analysis.task_helper.token_based import TokenOrientationTaskHelper
from nicr_mt_scene_analysis.task_helper.token_based import TokenPanopticTaskHelper
from nicr_mt_scene_analysis.task_helper.token_based import TokenSceneTaskHelper
from nicr_mt_scene_analysis.task_helper.token_based import TokenSemanticTaskHelper

from .data import DatasetType


def get_task_helpers(
    args,
    dataset: DatasetType,
    matching_cache: TokenMatchingCache,
) -> Tuple[TokenTaskHelperType, ...]:
    task_helper = []
    if 'token-mask' in args.tasks:
        mask_mode = None
        if 'token-semantic' in args.tasks:
            mask_mode = 'semantic'
        elif 'token-panoptic' in args.tasks:
            # only semantic or panoptic mask supported for now (exclusive).
            # ensure that tasks don't contain both.
            assert mask_mode is None
            mask_mode = 'panoptic'
        elif 'token-visual-embedding' in args.tasks:
            assert mask_mode is None
            mask_mode = 'panoptic'
        assert mask_mode is not None
        mask_kwargs = {}
        if args.token_mask_match_with_semantic_class and (
            'token-semantic' in args.tasks
            or 'token-panoptic' in args.tasks
            or 'token-visual-embedding' in args.tasks
        ):
            class_labels_key = {
                'semantic': 'semantic_token_labels',
                'panoptic': 'panoptic_token_labels',
            }[mask_mode]
            class_label_offset = {
                'semantic': 1,
                'panoptic': 0,
            }[mask_mode]
            class_scores_prefix = 'token_semantic'
            if 'token-visual-embedding' in args.tasks:
                class_scores_prefix = 'token_visual_embedding_matcher'
            mask_kwargs.update(
                class_scores_prefix=class_scores_prefix,
                class_labels_key=class_labels_key,
                class_label_offset=class_label_offset,
                class_weight=args.token_mask_semantic_matcher_class_weight,
            )
        mask_kwargs['unmatched_mask_coefficient'] = (
            args.token_mask_unmatched_weight
        )
        mask_kwargs['unmatched_mask_topk_frac'] = (
            args.token_mask_unmatched_topk_frac
        )
        task_helper.append(
            TokenMaskTaskHelper(
                mask_mode=mask_mode,
                matching_cache=matching_cache,
                **mask_kwargs,
            )
        )

    if 'token-semantic' in args.tasks:
        class_weights = dataset.semantic_compute_class_weights(
            weight_mode=args.token_semantic_class_weighting,
            c=args.token_semantic_class_weighting_logarithmic_c,
            ignore_first_class=True,
            n_threads=4,
            debug=False
        )
        if args.debug:
            print("Semantic class weights:", class_weights)

        task_helper.append(
            TokenSemanticTaskHelper(
                n_classes=dataset.semantic_n_classes_without_void,
                class_weights=class_weights,
                label_smoothing=args.token_semantic_loss_label_smoothing,
                matching_cache=matching_cache,
                examples_cmap=dataset.semantic_class_colors_without_void,
                semantic_coefficient=4.0,
            )
        )

    if 'token-panoptic' in args.tasks:
        semantic_n_classes = dataset.semantic_n_classes_without_void
        semantic_classes_is_thing = tuple(
            dataset.config.semantic_label_list.classes_is_thing
        )
        assert len(semantic_classes_is_thing) == semantic_n_classes + 1
        task_helper.append(
            TokenPanopticTaskHelper(
                semantic_n_classes_without_void=semantic_n_classes,
                semantic_classes_is_thing=semantic_classes_is_thing,
                mask_threshold=args.token_panoptic_mask_threshold,
                mask_pixel_threshold=args.token_panoptic_mask_pixel_threshold,
                overlap_threshold=args.token_panoptic_overlap_threshold,
                matching_cache=matching_cache,
                label_smoothing=args.token_panoptic_loss_label_smoothing,
                examples_cmap=tuple(
                    tuple(c) for c in dataset.semantic_class_colors
                ),
                norm_margin=args.token_panoptic_norm_margin,
                norm_coefficient=args.token_panoptic_norm_coefficient,
            )
        )

    if 'token-scene' in args.tasks:
        task_helper.append(
            TokenSceneTaskHelper(
                n_classes=dataset.scene_n_classes_without_void,
                class_weights=None,
                label_smoothing=args.token_scene_loss_label_smoothing
            )
        )

    if 'token-orientation' in args.tasks:
        # supervised exactly like the per-query class head: reuses the mask
        # helper's query<->segment matching cache, regresses one biternion
        # orientation per matched query, and only on queries whose segment
        # carries a valid orientation annotation.
        task_helper.append(
            TokenOrientationTaskHelper(
                matching_cache=matching_cache,
            )
        )

    if 'token-image-embedding' in args.tasks:
        task_helper.append(
            TokenImageEmbeddingTaskHelper(
                n_classes=dataset.scene_n_classes_without_void,
            )
        )

    if 'token-visual-embedding' in args.tasks:
        semantic_classes_is_thing = tuple(
            dataset.config.semantic_label_list.classes_is_thing
        )
        assert (
            len(semantic_classes_is_thing)
            == dataset.semantic_n_classes_without_void + 1
        )
        task_helper.append(
            TokenEmbeddingTaskHelper(
                n_classes=dataset.semantic_n_classes_without_void,
                semantic_classes_is_thing=semantic_classes_is_thing,
                matching_cache=matching_cache,
                examples_cmap=tuple(
                    tuple(c) for c in dataset.semantic_class_colors
                ),
                embedding_coefficient=(
                    args.token_visual_embedding_loss_coefficient
                ),
                negative_coefficient=(
                    args.token_visual_embedding_negative_coefficient
                ),
                negative_margin=args.token_visual_embedding_negative_margin,
                norm_margin=args.token_visual_embedding_norm_margin,
                norm_coefficient=args.token_visual_embedding_norm_coefficient,
                disable_aux_loss=args.token_visual_embedding_disable_aux_loss,
                enable_linear_probing=True,
            )
        )
    return tuple(task_helper)
