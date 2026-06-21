# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from copy import deepcopy
from dataclasses import asdict
from functools import partial
import re
import warnings

import numpy as np
from nicr_mt_scene_analysis.data import CollateIgnoredDict
from nicr_mt_scene_analysis.data import mt_collate
from nicr_mt_scene_analysis.data import RandomSamplerSubset
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from nicr_scene_analysis_datasets.dataset_base import build_dataset_config
from nicr_scene_analysis_datasets.dataset_base import DatasetConfig
from nicr_scene_analysis_datasets.auxiliary_data import DatasetConfigWithAuxiliary
from nicr_scene_analysis_datasets.pytorch import get_dataset_class
from nicr_scene_analysis_datasets.pytorch import wrap_dataset_with_auxiliary_data
from nicr_scene_analysis_datasets.dataset_base import MetaDict
from nicr_scene_analysis_datasets.dataset_base import OrientationDict
from nicr_scene_analysis_datasets.dataset_base import PanopticEmbeddingDict
from nicr_scene_analysis_datasets.dataset_base import SampleIdentifier
from nicr_scene_analysis_datasets.dataset_base import SemanticLabel
from nicr_scene_analysis_datasets.dataset_base import SemanticLabelList
from nicr_scene_analysis_datasets.pytorch import DatasetType
from nicr_scene_analysis_datasets.pytorch import KNOWN_DATASETS    # noqa: F401
from nicr_scene_analysis_datasets.pytorch import KNOWN_CLASS_WEIGHTINGS    # noqa: F401
from nicr_scene_analysis_datasets.pytorch import ConcatDataset
from nicr_scene_analysis_datasets.pytorch import ScanNet


class ScanNetWithOrientations(ScanNet):
    def __init__(self, **kwargs):
        # ScanNet does provide annotations for instance orientations. However,
        # we need support for orientations when combining ScanNet with other
        # datasets, e.g., NYUv2 or SUNRGB-D. This class is a workaround to
        # provide empty OrientationDicts, indicating that the instances should
        # be ignored for fitting the orientation estimation task.
        # To further use ScanNet as main dataset, this class provides a
        # function to copy the 'use_orientations' information from another
        # dataset.
        sample_keys = kwargs['sample_keys']

        # call super without orientations key
        kwargs['sample_keys'] = tuple(
            sk for sk in kwargs['sample_keys'] if sk != 'orientations'
        )
        super().__init__(**kwargs)

        # restore sample keys and re-register loaders
        self._sample_keys = sample_keys
        self.auto_register_sample_key_loaders()

        self._use_orientations_replaced = False

    def _load_orientations(self, idx):
        # we do not have instance orientations for ScanNet
        return OrientationDict({})

    @staticmethod
    def get_available_sample_keys(split: str) -> Tuple[str]:
        return ScanNet.SPLIT_SAMPLE_KEYS[split] + ('orientations',)

    def __getitem__(self, idx):
        return super().__getitem__(idx)

    def copy_use_orientations_from(self, other_dataset):
        # we need another dataset to copy the 'use_orientations' information
        # from for each semantic class
        other_semantic_label_list = other_dataset.config.semantic_label_list

        # create new semantic label list
        new_semantic_label_list = SemanticLabelList()
        missing_classes = []
        for sl in self.config.semantic_label_list:
            if sl.class_name not in other_semantic_label_list:
                # we do not have this class in other_dataset
                new_semantic_label_list.add_label(sl)
                missing_classes.append(sl.class_name)
                continue

            # create new semantic label with copied 'use_orientations'
            idx = other_semantic_label_list.index(sl.class_name)
            other_sl = other_semantic_label_list[idx]
            sl_dict = asdict(sl)
            sl_dict['use_orientations'] = other_sl.use_orientations
            new_semantic_label_list.add_label(SemanticLabel(**sl_dict))

        # print warning for missing classes
        if len(missing_classes) > 0:
            warnings.warn(
                f"{self.__class__.__name__}: Could not copy 'use_orientations' "
                f"information for classes: {missing_classes} from dataset "
                f"{other_dataset.__class__.__name__}."
            )

        # replace current dataset config
        self._config = build_dataset_config(
            semantic_label_list=new_semantic_label_list,
            scene_label_list=self.config.scene_label_list,
            depth_stats=self.config.depth_stats
        )
        self._use_orientations_replaced = True


def parse_datasets(
    datasets_str: str,
    datasets_path_str: Optional[str] = None,
    datasets_split_str: Optional[str] = None
) -> List[Dict[str, Any]]:
    misconfiguration_error = ValueError(
        "Detected dataset misconfiguration, i.e., different number of "
        f"datasets, paths or splits. Datasets: '{datasets_str}', paths: "
        f"'{datasets_path_str}', splits: '{datasets_split_str}'."
    )

    # ':' indicates joined datasets
    dataset_specifiers = datasets_str.lower().split(':')
    if datasets_path_str is not None:
        dataset_paths = datasets_path_str.split(':')
        if len(dataset_paths) != len(dataset_specifiers):
            raise misconfiguration_error
    if datasets_split_str is not None:
        dataset_splits = datasets_split_str.lower().split(':')
        if len(dataset_splits) != len(dataset_specifiers):
            raise misconfiguration_error

    datasets = []
    for i, dataset in enumerate(dataset_specifiers):
        # handle complex dataset specifiers (e.g., 'sunrgbd[kv1,kv2]' or
        # 'ade20k^depthanything_v2__indoor_large[640x480]')
        re_res = re.findall(
            '([a-z0-9\\_\\-]+)\\^?([a-z0-9\\_\\-]*)\\[?([a-z0-9\\_\\-]*)\\]?',
            dataset
        )
        assert len(re_res) == 1 and len(re_res[0]) == 3
        # parse results (dataset_name, cameras_str)
        ds_name, ds_depth_estimator, ds_cameras = re_res[0]
        # split cameras
        ds_cameras = ds_cameras.split(',') if ds_cameras else None

        # assert ds_name not in dataset_dict, f"Got same '{ds_name}' twice."
        datasets.append({
            'name': ds_name,
            'path': None if datasets_path_str is None else dataset_paths[i],
            'split': None if datasets_split_str is None else dataset_splits[i],
            'depth_estimator': ds_depth_estimator or None,
            'cameras': ds_cameras
        })

    return datasets


def get_dataset(args, split):
    # define default kwargs dict for all datasets
    dataset_depth_mode = 'raw' if args.raw_depth else 'refined'
    use_domestic_scene_labels = not args.use_original_scene_labels
    default_dataset_kwargs = {
        'ade20k': {},
        'cityscapes': {
            'depth_mode': dataset_depth_mode,
            'semantic_n_classes': 19,
            'disparity_instead_of_depth': False
        },
        'coco': {},
        'hypersim': {
            'depth_mode': dataset_depth_mode,
            'subsample': None,
            'scene_use_indoor_domestic_labels': use_domestic_scene_labels
        },
        'nyuv2': {
            'depth_mode': dataset_depth_mode,
            'semantic_n_classes': args.nyuv2_semantic_n_classes,
            'scene_use_indoor_domestic_labels': use_domestic_scene_labels
        },
        'scannet': {
            'depth_mode': dataset_depth_mode,
            'instance_semantic_mode': 'refined',    # use refined annotations
            'scene_use_indoor_domestic_labels': use_domestic_scene_labels,
            'semantic_n_classes': args.scannet_semantic_n_classes,
            'semantic_use_nyuv2_colors': (
                args.scannet_semantic_n_classes in (20, 40)
            )
        },
        'scenenetrgbd': {
            'depth_mode': dataset_depth_mode,
        },
        'sunrgbd': {
            'depth_mode': dataset_depth_mode,
            # note, EMSANet paper uses False for 'depth_force_mm'
            'depth_force_mm': not args.sunrgbd_depth_do_not_force_mm,
            'instances_version': args.sunrgbd_instances_version,
            'semantic_use_nyuv2_colors': True,
            'scene_use_indoor_domestic_labels': use_domestic_scene_labels
        },
    }

    # prepare names, paths, and splits
    # ':' indicates joined datasets
    dataset_split = split.lower()
    n_datasets = len(parse_datasets(args.dataset))
    if 'train' == dataset_split and n_datasets > 1:
        # backward compatibility: use 'train' split for all datasets
        dataset_split = ':'.join(['train'] * n_datasets)

    # parse full dataset information
    datasets = parse_datasets(
        datasets_str=args.dataset,
        datasets_path_str=args.dataset_path,
        datasets_split_str=dataset_split
    )

    # check if SUNRGB-D is combined with other datasets
    if 'sunrgbd' in datasets and len(datasets) > 1:
        # we need to force depth in mm
        warnings.warn(
            "Forcing `depth_force_mm` for SUNRGB-D, as it is combined with "
            f"other datasets. Datasets to load: {datasets}."
        )
        default_dataset_kwargs['sunrgbd']['depth_force_mm'] = True

    token_semantic = 'token-semantic' in args.tasks
    token_panoptic = 'token-panoptic' in args.tasks
    token_visual_embedding = 'token-visual-embedding' in args.tasks
    token_orientation = 'token-orientation' in args.tasks
    token_scene = 'token-scene' in args.tasks
    token_image_embedding = 'token-image-embedding' in args.tasks

    # determine sample keys
    sample_keys = list(args.input_modalities)
    if (
        token_semantic
        or token_panoptic
        or token_visual_embedding
        or token_orientation
    ):
        sample_keys.append('semantic')
    if token_panoptic or token_visual_embedding or token_orientation:
        # token-orientation targets are derived from the panoptic token
        # segments, so it needs the same panoptic / instance keys.
        sample_keys.extend(['panoptic', 'instance'])
    if token_visual_embedding:
        sample_keys.extend(['panoptic_embedding', 'image_embedding'])
    if token_orientation:
        sample_keys.append('orientations')
    if token_scene:
        sample_keys.append('scene')
    if token_image_embedding:
        # Needed only for validation of the learned image embeddings.
        sample_keys.extend(['image_embedding', 'scene'])

    # add identifier for easier debugging and plotting
    sample_keys.append('identifier')
    # start from nicr_mt_scene_analysis >= 0.3.0 some preprocessors
    # (e.g. PanopticTargetGenerator and InstanceClearStuffIDs) support
    # to get dataset specific information (usually semantic_label_list
    # to determine stuff and thing classes) directly from the sample.
    # this help to mix datasets with different semantic classes (e.g.
    # nyuv2 and ade20k) and still use the correct thing/stuff classes for
    # each sample and thus allows creating mixed dataset batches.
    if 'meta' not in sample_keys:
        sample_keys.append('meta')
    # fix sample key for orientation
    if 'orientation' in sample_keys:
        idx = sample_keys.index('orientation')
        sample_keys[idx] = 'orientations'
    # rgbd (single encoder) modality still require rgb and depth
    if 'rgbd' in sample_keys:
        if 'rgb' not in sample_keys:
            sample_keys.append('rgb')
        if 'depth' not in sample_keys:
            sample_keys.append('depth')
        # remove rgbd key
        sample_keys.remove('rgbd')

    sample_keys = tuple(sample_keys)

    # get dataset instances
    dataset_instances = []
    for idx, dataset in enumerate(datasets):
        if 'none' == dataset['split']:
            # indicates that this dataset should not be loaded (e.g., for
            # training on ScanNet and SunRGB-D but validation only on SunRGB-D)
            continue

        # Embedding tasks and estimated depth use auxiliary data which is not
        # provided by the original dataset classes. Setting this argument adds
        # the auxiliary-data wrapper around the original dataset.
        with_auxiliary_data = (
            token_visual_embedding
            or token_image_embedding
            or dataset['depth_estimator'] is not None
        )

        # get dataset class
        if 'scannet' == dataset['name'] and 'orientations' in sample_keys:
            # we do not have orientation annotations for ScanNet, use
            # ScanNetWithOrientations as a simple workaround to mimic empty
            # OrientationDicts, however, this only makes sense if ScanNet is
            # is combined with another dataset that provides orientation
            warnings.warn(
                "Detected ScanNet dataset in dataset configuration: "
                f"{datasets} and training with orientation estimation. "
                "Switching to 'ScanNetWithOrientations' to mimic orientations."
            )
            Dataset = ScanNetWithOrientations
            if with_auxiliary_data:
                # wrap the dataset to provide auxiliary data.
                Dataset = wrap_dataset_with_auxiliary_data(Dataset)
        else:
            Dataset = get_dataset_class(
                dataset['name'], with_auxiliary_data=with_auxiliary_data
            )

        # get default kwargs for dataset
        dataset_kwargs = deepcopy(default_dataset_kwargs[dataset['name']])

        # handle subsample for ScanNet
        if 'scannet' == dataset['name']:
            if 'train' == dataset['split']:
                dataset_kwargs['subsample'] = args.scannet_subsample
            else:
                dataset_kwargs['subsample'] = args.validation_scannet_subsample

        # handle subsample for Hypersim
        if 'hypersim' == dataset['name']:
            if 'train' == dataset['split']:
                dataset_kwargs['subsample'] = args.hypersim_subsample

        # handle depth estimation for ADE20k
        if dataset['name'] in ['ade20k', 'coco']:
            if dataset['depth_estimator'] is not None:
                dataset_kwargs['depth_estimator'] = dataset['depth_estimator']

        # check if all sample keys are available
        sample_keys_avail = Dataset.get_available_sample_keys(dataset['split'])
        sample_keys_missing = set(sample_keys) - set(sample_keys_avail)

        if sample_keys_missing:
            # this indicates a common problem, however, it also happens for
            # inference ScanNet on test split
            warnings.warn(
                f"Sample keys '{sample_keys_missing}' are not available for "
                f"dataset '{dataset['name']}' and split '{dataset['split']}'. "
                "Removing them from sample keys."
            )
            sample_keys = tuple(set(sample_keys) - sample_keys_missing)

        # Using panoptic-embedding or image-embedding requires specifying the
        # estimator to use for reference embeddings
        if 'panoptic_embedding' in sample_keys:
            dataset_kwargs['panoptic_embedding_estimator'] = \
                args.token_embedding_reference_estimator
            # If the dataset has specific number of semantic classes, we
            # also forward it to the wrapped dataset (wrapped for auxiliary
            # data) so we use the correct embeddings.
            if 'semantic_n_classes' in default_dataset_kwargs:
                dataset_kwargs['semantic_n_classes'] = \
                    default_dataset_kwargs['semantic_n_classes']
        if 'image_embedding' in sample_keys:
            dataset_kwargs['image_embedding_estimator'] = \
                args.token_embedding_reference_estimator

        # computing the visual_mean_based_miou during validation requires
        # reference embeddings which can be computed by iterating over all
        # ground truth embeddings and retrieve a mean visual embedding vector
        # per semantic class. as we only evaluate the main training dataset
        # during training we only compute the reference embeddings for the
        # first dataset in the list.
        # Note that this doesn't actually predict any new embeddings but
        # just combines them with a semantic class.
        if all([
            with_auxiliary_data,
            idx == 0,  # only for first dataset
            # train_panoptic_2017 for ade20k
            'train' in dataset['split'],  # only for training split
            #  (e.g. to speed up test cases)
            not args.token_embedding_do_not_compute_mean_embedding,
        ]):
            dataset_kwargs['compute_mean_visual_embeddings'] = \
                True
            mean_visual_embedding_types = []
            if token_visual_embedding:
                mean_visual_embedding_types.append('semantic')
            if token_image_embedding:
                mean_visual_embedding_types.append('scene')
            dataset_kwargs['mean_visual_embedding_types'] = (
                mean_visual_embedding_types
            )

        # instantiate dataset object
        dataset_instance = Dataset(
            dataset_path=dataset['path'],
            split=dataset['split'],
            sample_keys=sample_keys,
            use_cache=args.cache_dataset,
            cache_disable_deepcopy=False,  # False as we modify samples inplace
            cameras=dataset['cameras'],
            **dataset_kwargs
        )

        # TODO: can be removed from codebase later
        if 'hypersim' == dataset['name'] and args.hypersim_use_old_depth_stats:
            # patch dataset
            from nicr_scene_analysis_datasets import dataset_base
            dataset_instance._config = dataset_base.build_dataset_config(
                semantic_label_list=(
                    dataset_instance._config.semantic_label_list
                ),
                scene_label_list=dataset_instance._config.scene_label_list,
                depth_stats=dataset_instance._TRAIN_SPLIT_DEPTH_STATS_V030
            )
            train_depth_stats = dataset_instance._TRAIN_SPLIT_DEPTH_STATS_V030
            assert dataset_instance.depth_std == train_depth_stats.std
            assert dataset_instance.depth_mean == train_depth_stats.mean

        dataset_instances.append(dataset_instance)

    if 1 == len(dataset_instances):
        # single dataset
        return dataset_instances[0]

    if isinstance(dataset_instances[0], ScanNetWithOrientations):
        # we switched from ScanNet to ScanNetWithOrientations as it is combined
        # with other datasets, however, we do not have valid 'use_orientations'
        # information for ScanNet, so we copy it from the next dataset
        dataset_instances[0].copy_use_orientations_from(dataset_instances[1])

    # concatenated datasets
    return ConcatDataset(dataset_instances[0], *dataset_instances[1:])


class DataHelper:
    def __init__(
        self,
        dataset_train: DatasetType,
        batch_size_train: int,
        datasets_valid: Iterable[DatasetType],
        batch_size_valid: Optional[int] = None,
        subset_train: Union[float, Sequence[float]] = 1.0,
        subset_deterministic: bool = False,
        n_workers: int = 8,
        persistent_worker: bool = False,
    ) -> None:
        # use the OS default dataloader worker start method (fork on Linux,
        # spawn on macOS). forcing fork on macOS segfaults the workers, and
        # Linux already uses fork by default.
        self._mp_context = None

        # we use a modified collate function to handle elements of different
        # spatial resolution and to ignore numpy arrays, dicts containing
        # orientations (OrientationDict), and simple tuples storing shapes
        collate_fn = partial(mt_collate,
                             type_blacklist=(np.ndarray,
                                             CollateIgnoredDict,
                                             MetaDict,
                                             OrientationDict,
                                             PanopticEmbeddingDict,
                                             SampleIdentifier))

        # training split/set
        sampler = RandomSamplerSubset(
            data_source=dataset_train,
            subset=subset_train,
            deterministic=subset_deterministic
        )
        train_loader_kwargs = dict(
            batch_size=batch_size_train,
            sampler=sampler,
            drop_last=True,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=n_workers,
            persistent_workers=persistent_worker
        )
        if self._mp_context is not None and n_workers > 0:
            train_loader_kwargs['multiprocessing_context'] = self._mp_context
        self._dataloader_train = DataLoader(
            dataset_train, **train_loader_kwargs
        )

        # validation split/set
        self._dataloaders_valid = tuple(
            DataLoader(
                dataset_valid,
                batch_size=batch_size_valid or 3*batch_size_train,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn,
                pin_memory=True,
                # Note that a lot of datasets only have a single camera
                # resolution, and thus we don't want to always start n_workers
                # with the maximum number of workers.
                # However, the length of the dataset can only be determined
                # if it is loaded with data (e.g. not the case
                # for inference_samples.py)
                num_workers=(
                    min(n_workers, len(dataset_valid))
                    if dataset_valid.dataset_path is not None
                    else n_workers
                ),
                persistent_workers=persistent_worker
            )
            for dataset_valid in datasets_valid
        )

        self._overfitting_enabled = False
        # we use the (first) valid dataset when overfitting mode gets enabled,
        # copy the dataset here to ensure that no sample was drawn before
        self._overfitting_dataset = deepcopy(self.datasets_valid[0])

    def enable_overfitting_mode(self, n_valid_batches: int) -> None:
        self._overfitting_enabled = True

        batch_size = self._dataloader_train.batch_size
        n_samples = n_valid_batches * batch_size

        dataset = self._overfitting_dataset
        camera = dataset.cameras[0]
        if len(dataset.cameras) > 1:
            warnings.warn(
                "Overfitting dataset (valid split) contains multiple cameras. "
                f"Using first camera: '{camera}' to ensure samples of same "
                "spatial resolution."
            )
        dataset.filter_camera(camera)

        if n_samples > len(dataset):
            raise ValueError(
                f"Not enough data for overfitting. Tried to draw {n_samples} "
                f"samples from {len(dataset)}. Reduce the number of batches or "
                " the batch size for overfitting!"
            )

        self._overfitting_dataloader = DataLoader(
            Subset(dataset, tuple(range(n_samples))),
            batch_size=self._dataloader_train.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=self._dataloader_train.collate_fn,
            pin_memory=True,
            num_workers=self._dataloader_train.num_workers,
            persistent_workers=self._dataloader_train.persistent_workers
        )

        print("Enable overfitting mode (same data for training and validation) "
              f"with {n_valid_batches} batches (each with {batch_size} "
              "samples) from validation split.")

    @property
    def dataset_config(
        self
    ) -> Union[DatasetConfig, DatasetConfigWithAuxiliary]:
        # use config of train split
        return self._dataloader_train.dataset.config

    @property
    def dataset_train(self) -> DatasetType:
        return self._dataloader_train.dataset

    @property
    def datasets_valid(self) -> Tuple[DatasetType]:
        return tuple(loader.dataset for loader in self._dataloaders_valid)

    def set_train_preprocessor(self, preprocessor):
        self._dataloader_train.dataset.preprocessor = preprocessor

    def set_valid_preprocessor(self, preprocessor):
        for dataset in self.datasets_valid:
            dataset.preprocessor = preprocessor

        # apply preprocessor to overfitting dataset as well
        self._overfitting_dataset.preprocessor = deepcopy(preprocessor)

    @property
    def train_dataloader(self) -> DataLoader:
        if self._overfitting_enabled:
            return self._overfitting_dataloader

        return self._dataloader_train

    @property
    def valid_dataloaders(self) -> Tuple[DataLoader]:
        if self._overfitting_enabled:
            return tuple([self._overfitting_dataloader])

        return self._dataloaders_valid


def get_datahelper(args) -> DataHelper:
    # get datasets
    dataset_train = get_dataset(args, args.split)
    dataset_valid = get_dataset(args, args.validation_split)

    # create list of datasets for validation (each with only one camera ->
    # same resolution)
    dataset_valid_list = []
    for camera in dataset_valid.cameras:
        dataset_camera = deepcopy(dataset_valid).filter_camera(camera)
        dataset_valid_list.append(dataset_camera)

    # combine everything in a data helper
    # persistent workers can speed up training (especially for small
    # datasets), however, they also need more memory
    use_persistent_worker = (
        args.n_workers > 0 and not args.no_persistent_worker
    )
    return DataHelper(
        dataset_train=dataset_train,
        subset_train=args.subset_train,
        subset_deterministic=args.subset_deterministic,
        batch_size_train=args.batch_size,
        datasets_valid=dataset_valid_list,
        batch_size_valid=args.validation_batch_size,
        n_workers=args.n_workers,
        persistent_worker=use_persistent_worker,
    )


# simple helper function to cast values to a specific dtype
def cast_to_dtype(value, dtype):
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    elif isinstance(value, dict):
        return {k: cast_to_dtype(v, dtype) for k, v in value.items()}
    elif isinstance(value, list):
        return [cast_to_dtype(v, dtype) for v in value]
    elif isinstance(value, tuple):
        return tuple(cast_to_dtype(v, dtype) for v in value)
    return value
