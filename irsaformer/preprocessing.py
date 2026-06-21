# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Optional, Tuple

from torchvision.transforms import Compose

from nicr_mt_scene_analysis.data.preprocessing import DenseVisualEmbeddingTargetGenerator
from nicr_mt_scene_analysis.data.preprocessing import FullResCloner
from nicr_mt_scene_analysis.data.preprocessing import InstanceClearStuffIDs
from nicr_mt_scene_analysis.data.preprocessing import KeyCleaner
from nicr_mt_scene_analysis.data.preprocessing import NormalizeDepth
from nicr_mt_scene_analysis.data.preprocessing import NormalizeRGB
from nicr_mt_scene_analysis.data.preprocessing import OrientationTargetGenerator
from nicr_mt_scene_analysis.data.preprocessing import PanopticTargetGenerator
from nicr_mt_scene_analysis.data.preprocessing import PanopticTokenMaskTargetGenerator
from nicr_mt_scene_analysis.data.preprocessing import RandomCrop
from nicr_mt_scene_analysis.data.preprocessing import RandomHorizontalFlip
from nicr_mt_scene_analysis.data.preprocessing import RandomHSVJitter
from nicr_mt_scene_analysis.data.preprocessing import RandomResize
from nicr_mt_scene_analysis.data.preprocessing import Resize
from nicr_mt_scene_analysis.data.preprocessing import ScaleDepth
from nicr_mt_scene_analysis.data.preprocessing import SemanticClassMapper
from nicr_mt_scene_analysis.data.preprocessing import SemanticTokenMaskTargetGenerator
from nicr_mt_scene_analysis.data.preprocessing import ToTorchTensors
from nicr_scene_analysis_datasets import ScanNet

from .data import DatasetType
from .data import parse_datasets


def get_preprocessor(
    args,
    dataset: DatasetType,
    phase: str,
    multiscale_downscales: Optional[Tuple[int, ...]] = None,
) -> Compose:
    assert phase in ('train', 'test')

    dataset_config = dataset.config
    sample_keys = dataset.sample_keys
    max_instances_per_category = dataset.instance_max_instances_per_image

    transforms = []

    # check if ScanNet benchmark mode is enabled -> handle remapping
    if 'test' == phase and args.validation_scannet_benchmark_mode:
        # enable ScanNet benchmark mode for validation ONLY, i.e., mapping
        # ignored classes to void (40 -> 20, 549 -> 200) to ignore them in
        # metrics
        assert args.scannet_semantic_n_classes in (40, 549)
        if 40 == args.scannet_semantic_n_classes:
            mapping = ScanNet.SEMANTIC_CLASSES_40_MAPPING_TO_BENCHMARK
        else:
            mapping = ScanNet.SEMANTIC_CLASSES_549_MAPPING_TO_BENCHMARK200
        classes_to_ignore = tuple(
            c_data
            for c_data, c_benchmark in mapping.items()
            if c_benchmark == 0 and c_data != 0     # ignore void
        )
        assert len(classes_to_ignore) in (40-20, 549-200)

        transforms.append(
            SemanticClassMapper(
                classes_to_map=classes_to_ignore,
                new_label=0,
            )
        )

    # check if SUNRGB-D is combined as main dataset with NYUv2, ScanNet or
    # Hypersim -> ignore last three classes (other*)
    datasets = tuple(ds['name'] for ds in parse_datasets(args.dataset))
    if 'sunrgbd' == datasets[0]:
        if any(d in ('nyuv2', 'hypersim', 'scannet') for d in datasets[1:]):
            # map last three classes to void (ignore these classes in training/
            # validation)
            transforms.append(
                SemanticClassMapper(
                    classes_to_map=(38, 39, 40),
                    new_label=0,
                )
            )

    # instance preprocessing
    if 'instance' in sample_keys:
        # depending on the dataset and the applied division into stuff and
        # thing classes, the data may contain valid instance ids for instances
        # of stuff classes, we force id=0 (= no instance) for all stuff classes
        # including void to ensure that each stuff class is considered as a
        # single segment later
        # note that this preprocessor should be applied before resizing to
        # ensure that this requirement also applies to the full resolution
        # images that may be used for determining evaluation metrics
        transforms.append(
            InstanceClearStuffIDs(
                use_is_thing_from_meta=True,
            )
        )

    if 'panoptic_embedding' in sample_keys:
        # note: we need the panoptic targets in full resolution, so it is
        # important to have this preprocessor before the resize
        assert 'semantic' in sample_keys and 'instance' in sample_keys, \
            "Panoptic embedding targets require semantic and instance targets!"
        transforms.append(
            PanopticTargetGenerator(
                use_is_thing_from_meta=True,
            )
        )
    elif 'token-panoptic' in args.tasks:
        # Some datasets (e.g., NYUv2) do not provide panoptic targets. Build
        # them from semantic + instance maps when needed.
        if 'panoptic' not in sample_keys:
            assert 'semantic' in sample_keys and 'instance' in sample_keys, \
                "Token panoptic targets require semantic and instance maps."
            transforms.append(
                PanopticTargetGenerator(
                    use_is_thing_from_meta=True,
                )
            )

    if 'train' == phase:
        # augmentation
        transforms.extend([
            RandomResize(
                min_scale=args.aug_scale_min,
                max_scale=args.aug_scale_max,
            ),
            RandomCrop(
                crop_height=args.input_height,
                crop_width=args.input_width,
            ),
            RandomHSVJitter(
                hue_jitter=10/(360/2),     # +-10 degree
                saturation_jitter=20/255,     # +- ~8%
                value_jitter=50/255,     # +- ~16%
            ),
            RandomHorizontalFlip(p=0.5),
        ])
    else:
        # create full-resolution copies of the relevant inputs (required for
        # resizing in inference and metrics)
        transforms.append(
            FullResCloner(
                keys_to_keep_fullres=(
                    'rgb', 'depth',    # resizing in inference
                    'semantic', 'normal', 'instance', 'panoptic'),    # eval!
                ignore_missing_keys=True    # not all keys may be available
            )
        )

        if not args.validation_full_resolution:
            # resize input images to network input resolution
            # validation_full_resolution means no resizing at all
            transforms.append(
                Resize(
                    height=args.validation_input_height,
                    width=args.validation_input_width,
                    keep_aspect_ratio=args.validation_resize_keep_aspect_ratio,
                    padding_mode=args.validation_resize_padding_mode,
                    keys_to_ignore=(
                        'image_embedding',
                    )
                )
            )
            
    if 'semantic' in sample_keys and 'token-semantic' in args.tasks:
            transforms.append(
                SemanticTokenMaskTargetGenerator(
                    semantic_key='semantic',
                    masks_key='semantic_token_masks',
                    labels_key='semantic_token_labels',
                    ignore_labels=(0,),
                )
            )

    if (
        'token-panoptic' in args.tasks
        or 'token-visual-embedding' in args.tasks
    ) and (
        'panoptic' in sample_keys
        or ('semantic' in sample_keys and 'instance' in sample_keys)
    ):
        transforms.append(
            PanopticTokenMaskTargetGenerator(
                panoptic_key='panoptic',
                masks_key='panoptic_token_masks',
                labels_key='panoptic_token_labels',
                ids_key='panoptic_token_ids',
                void_label=(
                    dataset.semantic_n_classes_without_void
                    * max_instances_per_category
                ),
                max_instances_per_category=max_instances_per_category,
            )
        )

    if 'orientations' in sample_keys:
        estimate_orientation = \
            dataset_config.semantic_label_list.classes_use_orientations
        transforms.append(
            OrientationTargetGenerator(
                semantic_classes_estimate_orientation=estimate_orientation
            )
        )

    if 'token-visual-embedding' in args.tasks:
        transforms.append(DenseVisualEmbeddingTargetGenerator(
            diff_factor=args.token_visual_embedding_diff_factor,
        ))

    # default preprocessing
    if 'rgb' in args.input_modalities or 'rgbd' in args.input_modalities:
        transforms.append(NormalizeRGB())
    if 'depth' in args.input_modalities or 'rgbd' in args.input_modalities:
        if args.scale_depth:
            # simply scale depth values - each sample is scaled to [0, 1]
            # independently
            transforms.append(
                ScaleDepth(
                    new_min=0.0,
                    new_max=1.0,
                    raw_depth=args.raw_depth,
                    invalid_depth_value=0
                )
            )
        else:
            # standardize depth values - similar to NormalizeRGB but with
            # depth mean and std
            transforms.append(
                NormalizeDepth(
                    depth_mean=dataset_config.depth_stats.mean,
                    depth_std=dataset_config.depth_stats.std,
                    raw_depth=args.raw_depth,
                    invalid_depth_value=0
                )
            )

    keys_to_clean = []
    if phase == 'train':
        # Full-resolution clones are only used for validation metrics.
        # Thus we drop it to save some vram.
        keys_to_clean.extend(
            key for key in sample_keys if key.endswith('_fullres')
        )
    if 'instance' not in sample_keys:
        keys_to_clean.append('instance')
    if 'panoptic' not in sample_keys and 'token-panoptic' not in args.tasks:
        keys_to_clean.append('panoptic')
    if (
        'token-visual-embedding' not in args.tasks
    ):
        keys_to_clean.append('panoptic_embedding')
    transforms.append(KeyCleaner(keys_to_clean=keys_to_clean))

    transforms.append(ToTorchTensors())

    # stack all transforms into a single preprocessor object
    preprocessor = Compose(transforms=transforms)

    if args.debug:
        print(f"Preprocessor for phase: '{phase}':\n{preprocessor}")

    return preprocessor
