# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from nicr_mt_scene_analysis.loss_weighting import FixedLossWeighting
from nicr_mt_scene_analysis.loss_weighting import LossWeightingType
from nicr_mt_scene_analysis.task_helper.base import get_total_loss_key


def get_loss_weighting_module(args) -> LossWeightingType:
    # we stick to fixed task weighting as none of the remaining was working well

    # assign weight to each task (based on positional order)
    tasks_weights = {}
    assert len(args.tasks) == len(args.tasks_weighting)
    tasks_weights = {
        # Loss keys use underscores while CLI task names use dashes, e.g.
        # token-visual-embedding -> token_visual_embedding_total_loss.
        task.replace('-', '_'): weight
        for task, weight in zip(args.tasks, args.tasks_weighting)
    }

    # convert task weights to loss weights (keys must match the later losses)
    # note, we consider only losses marked as total for weighting for now
    loss_weights = {}

    # handle token mask as the key depends on the task it is trained with
    if 'token_mask' in tasks_weights:
        if (
            'token_panoptic' in tasks_weights
            or 'token_visual_embedding' in tasks_weights
        ):
            joint_task = 'token_panoptic'
        else:
            joint_task = 'token_semantic'
        weight_mask = tasks_weights.pop('token_mask')
        loss_weights[f'{joint_task}_mask_total_loss'] = weight_mask

    # for the remaining tasks, simply append the total loss suffix
    loss_weights.update({
        get_total_loss_key(task): value
        for task, value in tasks_weights.items()
    })

    return FixedLossWeighting(weights=loss_weights)
