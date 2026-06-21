# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import pytest

from nicr_scene_analysis_datasets.utils.testing import DATASET_PATH_DICT

from irsaformer.args import ArgParserIRSAFormer
from irsaformer.data import get_datahelper
from irsaformer.data import KNOWN_DATASETS


@pytest.mark.parametrize('dataset', KNOWN_DATASETS)
def test_data_helper(dataset):
    """Test data helper"""
    # get args
    parser = ArgParserIRSAFormer()
    if 'coco' == dataset:
        input_modalities = ('rgb',)
    else:
        input_modalities = ('rgb', 'depth')
    split_train = 'train'
    split_valid = 'valid'
    if dataset == 'ade20k':
        split_train = 'train_panoptic_2017'
        split_valid = 'valid_panoptic_2017'
    args = parser.parse_args(
        ['--dataset', dataset,
         '--dataset-path', DATASET_PATH_DICT[dataset],
         '--input-modalities', *input_modalities,
         '--token-modality', 'rgb',
         '--split', split_train,
         '--validation-split', split_valid],
        verbose=False)

    data = get_datahelper(args)
    for idx, batch in enumerate(data.train_dataloader):
        assert batch is not None
        if idx == 10:
            break

    for idx, batch in enumerate(data.valid_dataloaders[0]):
        assert batch is not None
        if idx == 10:
            break
