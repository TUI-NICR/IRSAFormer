# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
import os
from dataclasses import replace

import numpy as np
import pytest
import torch

from nicr_mt_scene_analysis.data.preprocessing.base import APPLIED_PREPROCESSING_KEY
from nicr_mt_scene_analysis.data.preprocessing.resize import Resize
from nicr_mt_scene_analysis.testing.onnx import export_onnx_model

from nicr_scene_analysis_datasets.auxiliary_data import DatasetConfigWithAuxiliary

from irsaformer.args import ArgParserIRSAFormer
from irsaformer.data import get_dataset
from irsaformer.model import IRSAFormer


def _get_dataset_type(dataset):
    meta = dataset.load('meta', 0)
    # Auxiliary datasets store the wrapped original dataset class in meta.
    # Plain datasets do not need this entry; their own class is the type.
    return meta.get('dataset_type', dataset.__class__)


def _with_minimal_semantic_text_embeddings(dataset_config, embedding_dim=32):
    if not isinstance(dataset_config, DatasetConfigWithAuxiliary):
        return dataset_config
    if dataset_config.semantic_text_embeddings:
        return dataset_config

    embeddings = []
    for idx in range(len(dataset_config.semantic_label_list)):
        embedding = np.zeros((embedding_dim,), dtype='float32')
        embedding[idx % embedding_dim] = 1.0
        embeddings.append(embedding)

    return replace(dataset_config, semantic_text_embeddings=embeddings)


def model_test(tasks,
               modalities,
               backbone,
               do_postprocessing,
               training,
               tmp_path,
               additional_args=None):

    parser = ArgParserIRSAFormer()
    cli_args = [
        '--input-modalities', *modalities,
        '--tasks', *tasks,
        '--rgbd-encoder-backbone', backbone,
        '--rgb-encoder-backbone', backbone,
        '--depth-encoder-backbone', backbone,
        '--no-pretrained-backbone',
        '--dataset', 'nyuv2',
        '--device', 'cpu',
        '--token-modality', modalities[0],
        # Use small batch size to avoid OOM in CI
        '--batch-size', '2',
        '--validation-batch-size', '2',
    ]
    if additional_args:
        cli_args.extend(additional_args)

    args = parser.parse_args(cli_args, verbose=False)

    dataset = get_dataset(args, split='train')
    dataset_config = _with_minimal_semantic_text_embeddings(dataset.config)
    dataset_type = _get_dataset_type(dataset)

    # create model
    model = IRSAFormer(
        args,
        dataset_configs={dataset_type.__name__: dataset_config}
    )
    if not training:
        model.eval()

    # determine input
    batch_size = 3
    input_shape = (480, 640)
    batch = {}
    if 'rgb' in args.input_modalities or 'rgbd' in args.input_modalities:
        batch['rgb'] = torch.randn((batch_size, 3)+input_shape)
    if 'depth' in args.input_modalities or 'rgbd' in args.input_modalities:
        batch['depth'] = torch.randn((batch_size, 1)+input_shape)
    batch['meta'] = [{'dataset_type': dataset_type} for _ in range(batch_size)]

    # Add applied preprocessing to batch which is required for postprocessing
    batch[APPLIED_PREPROCESSING_KEY] = [
        [{
            'type': Resize.__name__,
            'valid_region_slice_y': slice(0, input_shape[0]),
            'valid_region_slice_x': slice(0, input_shape[1]),
        },]
    ]*batch_size

    if not training and do_postprocessing:
        # for inference postprocessing, inputs in full resolution are required
        if 'rgb' in batch:
            batch['rgb_fullres'] = batch['rgb'].clone()
        if 'depth' in batch:
            batch['depth_fullres'] = batch['depth'].clone()

    # apply model
    outputs = model(batch, do_postprocessing=do_postprocessing)

    # some simple checks for output
    if do_postprocessing:
        assert isinstance(outputs, dict)
    else:
        assert isinstance(outputs, list)
    assert outputs

    # export model to ONNX
    if not training and do_postprocessing:
        # stop here: inference postprocessing is challenging (no onnx export)
        return
    # determine filename and filepath
    tasks_str = '+'.join(tasks)
    modalities_str = '+'.join(modalities)
    filename = f'model_{modalities_str}_{tasks_str}'
    filename += f'__backbone_{backbone}'
    filename += f'__train{training}'
    filename += f'__post_{do_postprocessing}'
    filename += '.onnx'
    filepath = os.path.join(tmp_path, filename)
    # export
    # note, the last element in input tuple is interpreted as named args
    # if no named args should be passed use
    x = (batch, {'do_postprocessing': do_postprocessing})
    export_onnx_model(filepath, model, x)


@pytest.mark.parametrize('modalities', (('rgb',),
                                        ('depth',),
                                        ('rgbd',)))
@pytest.mark.parametrize('backbone', ('dinov3_small_plus_qkvb',))
@pytest.mark.parametrize('do_postprocessing', (False, True))
@pytest.mark.parametrize('training', (False, True))
def test_token_visual_embedding_model(modalities, backbone, do_postprocessing,
                                      training, tmp_path):
    """Test token-visual-embedding on a small DINOv3 backbone."""
    model_test(
        tasks=('token-mask', 'token-visual-embedding'),
        modalities=modalities,
        backbone=backbone,
        do_postprocessing=do_postprocessing,
        training=training,
        tmp_path=tmp_path
    )
