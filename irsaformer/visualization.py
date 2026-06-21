# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
from typing import Any, Dict, Optional, Sequence, Union

import os
import warnings

import cv2
import numpy as np
import PIL

from nicr_mt_scene_analysis.data.preprocessing.clone import DEFAULT_CLONE_KEY
from nicr_mt_scene_analysis.data.preprocessing.resize import get_fullres_key
from nicr_mt_scene_analysis.types import BatchType
from nicr_mt_scene_analysis.visualization import InstanceColorGenerator
from nicr_mt_scene_analysis.visualization import PanopticColorGenerator
from nicr_mt_scene_analysis.visualization import visualize_depth
from nicr_mt_scene_analysis.visualization import visualize_heatmap
from nicr_mt_scene_analysis.visualization import visualize_instance
from nicr_mt_scene_analysis.visualization import visualize_instance_center
from nicr_mt_scene_analysis.visualization import visualize_instance_offset
from nicr_mt_scene_analysis.visualization import visualize_instance_orientations
from nicr_mt_scene_analysis.visualization import visualize_instance_pil
from nicr_mt_scene_analysis.visualization import visualize_orientation
from nicr_mt_scene_analysis.visualization import visualize_panoptic
from nicr_mt_scene_analysis.visualization import visualize_semantic_pil
from nicr_scene_analysis_datasets.dataset_base import DatasetConfig
from nicr_scene_analysis_datasets.utils.img import get_visual_distinct_colormap

KWARGS_INSTANCE_ORIENTATION = {
    'thickness': 3,
    'font_size': 45,
    'bg_color': 0,
    'bg_color_font': 'black'
}

KWARGS_INSTANCE_ORIENTATION_WHITEBG = {
    'thickness': 3,
    'font_size': 45,
    'bg_color': 255,
    'bg_color_font': 'white'
}

CV_WRITE_FLAGS = (cv2.IMWRITE_PNG_COMPRESSION, 9)


_shared_color_generators = {
    'instance': None,
    'panoptic': None,
}


def setup_shared_color_generators(
    dataset_config: DatasetConfig,
    max_instances_per_category: int = 1 << 16
) -> None:
    # instance color generator
    instance_shg = InstanceColorGenerator(
        cmap_without_void=get_visual_distinct_colormap(with_void=False)
    )
    _shared_color_generators['instance'] = instance_shg

    # panoptic color generator
    sem_labels = dataset_config.semantic_label_list
    panoptic_shg = PanopticColorGenerator(
        classes_colors=sem_labels.colors,
        classes_is_thing=sem_labels.classes_is_thing,
        max_instances=max_instances_per_category,
        void_label=0
    )
    _shared_color_generators['panoptic'] = panoptic_shg


def visualize(
    output_path: str,
    batch: BatchType,
    predictions: Dict[str, Any],
    dataset_config: DatasetConfig,
    use_shared_color_generators: bool = True,
    max_instances_per_category: int = 1 << 16,
) -> None:

    # color generators
    if use_shared_color_generators:
        instance_color_generator = _shared_color_generators['instance']
        panoptic_color_generator = _shared_color_generators['panoptic']
        if instance_color_generator is None or panoptic_color_generator is None:
            warnings.warn(
                "Shared color generators are not ready. Please call "
                "'setup_shared_color_generators' first."
            )
    else:
        instance_color_generator = None
        panoptic_color_generator = None

    # visualize ground truth
    gt_path = os.path.join(output_path, 'gt')
    use_token_panoptic = 'token_panoptic_segmentation' in predictions
    batch_visualization = visualize_batches(
        batch=batch,
        dataset_config=dataset_config,
        instance_color_generator=instance_color_generator,
        panoptic_color_generator=panoptic_color_generator,
        token_panoptic=use_token_panoptic,
        max_instances_per_category=max_instances_per_category,
    )
    save_visualization_result_dict(
        visualization_dict=batch_visualization,
        output_path=gt_path
    )

    # visualize ground truth for side outputs (downscaled images)
    additional_keys = ['_down_8', '_down_16', '_down_32']
    for key in additional_keys:
        if key not in batch:
            # we do not have side outputs
            continue

        # get batch dict for side output and copy identifier
        so_batch = batch[key]
        so_batch['identifier'] = so_batch['identifier']

        # visualize side output
        so_batch_visualization = visualize_batches(
            batch=so_batch,
            dataset_config=dataset_config,
            instance_color_generator=instance_color_generator,
            panoptic_color_generator=panoptic_color_generator,
            token_panoptic=use_token_panoptic,
            max_instances_per_category=max_instances_per_category,
        )
        save_visualization_result_dict(
            visualization_dict=so_batch_visualization,
            output_path=os.path.join(gt_path, key)
        )

    # visualize predictions
    prediction_visualization = visualize_predictions(
        predictions=predictions,
        batch=batch,
        dataset_config=dataset_config,
        instance_color_generator=instance_color_generator,
        panoptic_color_generator=panoptic_color_generator,
        max_instances_per_category=max_instances_per_category
    )
    save_visualization_result_dict(
        visualization_dict=prediction_visualization,
        output_path=os.path.join(output_path, 'pred')
    )


def save_visualization_result_dict(
    visualization_dict: Dict[str, Any],
    output_path: str
) -> None:
    os.makedirs(output_path, exist_ok=True)
    for key, value in visualization_dict.items():
        if key == 'identifier':
            continue
        for i, v in enumerate(value):
            out_filepath = os.path.join(
                output_path,
                key,
                *visualization_dict['identifier'][i]
            )
            os.makedirs(os.path.dirname(out_filepath), exist_ok=True)

            if isinstance(v, PIL.Image.Image):
                # value is a PIL image
                v.save(out_filepath + '.png')
            elif isinstance(v, np.ndarray):
                # value is an image given as numpy array, write with OpenCV
                if v.ndim == 3:
                    v = cv2.cvtColor(v, cv2.COLOR_RGB2BGR, CV_WRITE_FLAGS)
                cv2.imwrite(out_filepath + '.png', v)
            else:
                # scene label
                with open(out_filepath + '.txt', 'w') as f:
                    f.write(str(v))


def _apply_mask(
    img: np.ndarray,
    mask: np.ndarray,
    value: Union[np.ndarray, Sequence]
) -> None:
    # apply mask inplace
    img[mask, ...] = value
    return img


def _copy_and_apply_mask(
    img: np.ndarray,
    mask: np.ndarray,
    value: Union[np.ndarray, Sequence]
) -> np.ndarray:
    # copy img and apply mask
    return _apply_mask(img.copy(), mask, value)


def visualize_batches(
    batch: BatchType,
    dataset_config: DatasetConfig,
    instance_color_generator: Optional[InstanceColorGenerator] = None,
    panoptic_color_generator: Optional[PanopticColorGenerator] = None,
    token_panoptic: bool = False,
    max_instances_per_category: int = 1 << 16,
) -> Dict[str, Any]:
    # note, we use PIL whenever an image with palette is useful

    # semantic colors
    colors = dataset_config.semantic_label_list.colors_array    # with void

    # create dict storing the result
    result_dict = {}
    result_dict['identifier'] = batch['identifier']

    # dump inputs and targets without preprocessing ----------------------------
    if DEFAULT_CLONE_KEY in batch:
        batch_np = batch[DEFAULT_CLONE_KEY]
        # inputs
        if 'rgb' in batch_np:
            result_dict[f'{DEFAULT_CLONE_KEY}_rgb'] = list(batch_np['rgb'])
        if 'depth' in batch_np:
            result_dict[f'{DEFAULT_CLONE_KEY}_depth'] = [
                visualize_depth(img) for img in batch_np['depth']
            ]

        # semantic
        if 'semantic' in batch_np:
            result_dict[f'{DEFAULT_CLONE_KEY}_semantic'] = [
                visualize_semantic_pil(img, colors=colors)
                for img in batch_np['semantic']
            ]

        # instance
        if 'instance' in batch_np:
            result_dict[f'{DEFAULT_CLONE_KEY}_instance'] = [
                visualize_instance_pil(
                    instance_img=img,
                    shared_color_generator=instance_color_generator
                )
                for img in batch_np['instance']
            ]

        # orientation
        if 'orientations' in batch_np:
            result_dict[f'{DEFAULT_CLONE_KEY}_orientations'] = [
                visualize_instance_orientations(
                    *data,
                    shared_color_generator=instance_color_generator,
                    **KWARGS_INSTANCE_ORIENTATION
                ) for data in zip(batch_np['instance'],
                                  batch_np['orientations'])
            ]
            result_dict[f'{DEFAULT_CLONE_KEY}_orientations_white_bg'] = [
                visualize_instance_orientations(
                    *data,
                    shared_color_generator=instance_color_generator,
                    **KWARGS_INSTANCE_ORIENTATION_WHITEBG
                ) for data in zip(batch_np['instance'],
                                  batch_np['orientations'])
            ]

        # scene classification
        if 'scene' in batch_np:
            result_dict[f'{DEFAULT_CLONE_KEY}_scene'] = [
                dataset_config.scene_label_list[s].class_name
                for s in batch_np['scene']
            ]

    else:
        # we do not have the batch data without preprocessing
        batch_np = {}

    # semantic -----------------------------------------------------------------
    if 'semantic' in batch:
        # semantic may have changed due to mapping some classes to void
        result_dict['semantic'] = [
            visualize_semantic_pil(img, colors=colors)
            for img in batch['semantic'].cpu().numpy()
        ]

    # instance -----------------------------------------------------------------
    if 'instance' in batch:
        # instance may have changed due to selecting thing classes
        result_dict['instance'] = [
            visualize_instance_pil(
                instance_img=img,
                shared_color_generator=instance_color_generator
            )
            for img in batch['instance'].cpu().numpy()
        ]

        result_dict['instance_white_bg'] = [
            # use foreground mask to change background color to white
            _apply_mask(
                img=visualize_instance(
                    instance_img=img,
                    shared_color_generator=instance_color_generator
                ),
                mask=np.logical_not(fg),
                value=(255, 255, 255)
            )
            for img, fg in zip(batch['instance'].cpu().numpy(),
                               batch['instance_foreground'].cpu().numpy())
        ] if 'instance_foreground' in batch else []

        if 'instance_center' in batch:
            result_dict['instance_center'] = [
                visualize_instance_center(center_img=img)
                for img in batch['instance_center'].cpu().numpy()
            ]

        if 'instance_offset' in batch and 'instance_foreground' in batch:
            result_dict['instance_offset'] = [
                visualize_instance_offset(
                    offset_img=img.transpose(1, 2, 0),
                    foreground_mask=fg
                )
                for img, fg in zip(batch['instance_offset'].cpu().numpy(),
                                   batch['instance_foreground'].cpu().numpy())
            ]

    # orientation --------------------------------------------------------------
    if 'orientation' in batch:
        # instance orientation may have changed due to selecting thing classes
        # 2d dense orientation with black/white background
        if 'orientation_foreground' in batch:
            result_dict['orientation'] = [
                # use foreground mask to change background color to black
                _apply_mask(
                    img=visualize_orientation(o.transpose(1, 2, 0)),
                    mask=np.logical_not(fg),
                    value=(0, 0, 0)
                )
                for o, fg in zip(batch['orientation'].cpu().numpy(),
                                 batch['orientation_foreground'].cpu().numpy())
            ]
            result_dict['orientation_white_bg'] = [
                # change background color to white
                _copy_and_apply_mask(
                    img=o_img,
                    mask=np.logical_not(fg),
                    value=(255, 255, 255)
                )
                for o_img, fg in zip(
                    result_dict['orientation'],
                    batch['orientation_foreground'].cpu().numpy()
                )
            ]

        # orientation with outline
        if 'orientations_present' in batch and 'instance' in batch:
            result_dict['orientations'] = [
                visualize_instance_orientations(
                    *data,
                    shared_color_generator=instance_color_generator,
                    draw_outline=True,
                    **KWARGS_INSTANCE_ORIENTATION
                )
                for data in zip(batch['instance'].cpu().numpy(),
                                batch['orientations_present'])
            ]
            result_dict['orientations_white_bg'] = [
                visualize_instance_orientations(
                    *data,
                    shared_color_generator=instance_color_generator,
                    draw_outline=True,
                    **KWARGS_INSTANCE_ORIENTATION_WHITEBG
                )
                for data in zip(batch['instance'].cpu().numpy(),
                                batch['orientations_present'])
            ]

    # panoptic -----------------------------------------------------------------
    if 'panoptic' in batch:
        sem_labels = dataset_config.semantic_label_list
        if token_panoptic:
            token_colors = [(0, 0, 0)] + list(sem_labels.colors[1:])
            token_is_thing = (False,) + tuple(sem_labels.classes_is_thing[1:])
            token_void_label = 0
            result_dict['panoptic'] = [
                visualize_panoptic(
                    panoptic_img=img,
                    semantic_classes_colors=token_colors,
                    semantic_classes_is_thing=token_is_thing,
                    max_instances=max_instances_per_category,
                    void_label=token_void_label,
                    shared_color_generator=None,
                )
                for img in batch['panoptic'].cpu().numpy()
            ]
        else:
            result_dict['panoptic'] = [
                visualize_panoptic(
                    panoptic_img=img,
                    semantic_classes_colors=sem_labels.colors,
                    semantic_classes_is_thing=sem_labels.classes_is_thing,
                    max_instances=max_instances_per_category,
                    void_label=0,
                    shared_color_generator=panoptic_color_generator
                )
                for img in batch['panoptic'].cpu().numpy()
            ]

    # panoptic + orientation ---------------------------------------------------
    # panoptic image overlayed with orientation as text
    if 'panoptic' in batch and 'orientations_present' in batch:
        result_dict['panoptic_orientations'] = [
            _copy_and_apply_mask(
                img=panoptic_img,
                mask=visualize_instance_orientations(
                    instance_img=instance,
                    orientations=orientations,
                    shared_color_generator=instance_color_generator,
                    draw_outline=False,
                    thickness=3,
                    font_size=45,
                    bg_color=0,
                    bg_color_font='black'
                ).any(axis=-1),   # text mask
                value=(255, 255, 255)    # white text color
            )
            for panoptic_img, instance, orientations in zip(
                result_dict['panoptic'],
                batch['instance'].cpu().numpy(),
                batch['orientations_present']
            )
        ]

    # semantic embedding -------------------------------------------------------
    if 'embedding_indices' in batch:
        result_dict['embedding_indices'] = [
            visualize_instance_pil(lut.cpu().numpy())
            for lut in batch['embedding_indices']
        ]

    return result_dict


def visualize_predictions(
    predictions: Dict[str, Any],
    batch: BatchType,
    dataset_config: DatasetConfig,
    instance_color_generator: Optional[InstanceColorGenerator] = None,
    panoptic_color_generator: Optional[PanopticColorGenerator] = None,
    max_instances_per_category: int = 1 << 16,
) -> Dict[str, Any]:
    # note, we use PIL whenever an image with palette is useful

    # semantic colors
    colors = dataset_config.semantic_label_list.colors_array

    # create dict for results
    result_dict = {}
    result_dict['identifier'] = batch['identifier']

    # semantic -----------------------------------------------------------------
    # -> predicted class
    key = 'semantic_segmentation_idx'
    if key in predictions:
        for k in (key, get_fullres_key(key)):  # plain output and fullres
            result_dict[k] = [
                visualize_semantic_pil(img, colors=colors[1:])
                for img in predictions[k].cpu().numpy()
            ]
    # -> predicted class score
    key = 'semantic_segmentation_score'
    if key in predictions:
        for k in (key, get_fullres_key(key)):  # plain output and fullres
            result_dict[k] = [
                visualize_heatmap(img, cmap='jet')
                for img in predictions[k].cpu().numpy()
            ]

    # token-based semantic segmentation ---------------------------------------
    key = 'token_semantic_dense_idx'
    if key in predictions:
        for k in (key, get_fullres_key(key)):
            if k not in predictions:
                continue
            result_dict[k] = [
                visualize_semantic_pil(img, colors=colors[1:])
                for img in predictions[k].cpu().numpy()
            ]
    key = 'token_semantic_dense_score'
    if key in predictions:
        for k in (key, get_fullres_key(key)):
            if k not in predictions:
                continue
            result_dict[k] = [
                visualize_heatmap(img, cmap='jet')
                for img in predictions[k].cpu().numpy()
            ]

    # instance -----------------------------------------------------------------
    # -> instance segmentation using gt foreground mask (dataset eval only)
    key = 'instance_segmentation_gt_foreground'
    if key in predictions:
        for k in (key, get_fullres_key(key)):  # plain output and fullres
            result_dict[k] = [
                visualize_instance_pil(
                    instance_img=img,
                    shared_color_generator=instance_color_generator
                )
                for img in predictions[k].cpu().numpy()
            ]

    # panoptic segmentation ----------------------------------------------------
    sem_labels = dataset_config.semantic_label_list

    # -> token-based panoptic label
    key = 'token_panoptic_segmentation'
    if key in predictions:
        token_colors = [(0, 0, 0)] + list(sem_labels.colors[1:])
        token_is_thing = (False,) + tuple(sem_labels.classes_is_thing[1:])
        token_void_label = 0
        result_dict[key] = [
            visualize_panoptic(
                panoptic_img=img,
                semantic_classes_colors=token_colors,
                semantic_classes_is_thing=token_is_thing,
                max_instances=max_instances_per_category,
                void_label=token_void_label,
                shared_color_generator=None,
            )
            for img in predictions[key].cpu().numpy()
        ]

    token_panoptic_prefixes = (
        'token_visual_embedding_text_based',
        'token_visual_embedding_visual_mean_based',
        'token_visual_embedding_linear_probing',
    )
    for prefix in token_panoptic_prefixes:
        panoptic_key = f'{prefix}_panoptic_segmentation'
        if panoptic_key in predictions:
            token_colors = [(0, 0, 0)] + list(sem_labels.colors[1:])
            token_is_thing = (False,) + tuple(sem_labels.classes_is_thing[1:])
            token_void_label = 0
            result_dict[panoptic_key] = [
                visualize_panoptic(
                    panoptic_img=img,
                    semantic_classes_colors=token_colors,
                    semantic_classes_is_thing=token_is_thing,
                    max_instances=max_instances_per_category,
                    void_label=token_void_label,
                    shared_color_generator=None,
                )
                for img in predictions[panoptic_key].cpu().numpy()
            ]


    # token visual embedding ---------------------------------------------------
    semantic_prediction_prefixes = [
        'token_visual_embedding_text_based_semantic',
        'token_visual_embedding_visual_mean_based_semantic',
        'token_visual_embedding_linear_probing_semantic',
    ]

    for prefix in semantic_prediction_prefixes:
        idx_key = f'{prefix}_semantic_idx'
        if idx_key in predictions:
            result_dict[idx_key] = [
                visualize_semantic_pil(img, colors=colors[1:])
                for img in predictions[idx_key].cpu().numpy()
            ]

        fullres_idx_key = get_fullres_key(idx_key)
        if fullres_idx_key in predictions:
            result_dict[fullres_idx_key] = [
                visualize_semantic_pil(img, colors=colors[1:])
                for img in predictions[fullres_idx_key].cpu().numpy()
            ]

        score_key = f'{prefix}_semantic_score'
        if score_key in predictions:
            result_dict[score_key] = [
                visualize_heatmap(img, cmap='jet')
                for img in predictions[score_key].cpu().numpy()
            ]

        fullres_score_key = get_fullres_key(score_key)
        if fullres_score_key in predictions:
            result_dict[fullres_score_key] = [
                visualize_heatmap(img, cmap='jet')
                for img in predictions[fullres_score_key].cpu().numpy()
            ]

    # scene classification -----------------------------------------------------
    if 'scene_class_idx' in predictions:
        result_dict['scene'] = [
            dataset_config.scene_label_list_without_void[s].class_name
            for s in predictions['scene_class_idx']
        ]

    return result_dict
