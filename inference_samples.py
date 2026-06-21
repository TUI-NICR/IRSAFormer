# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from glob import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from nicr_mt_scene_analysis.data import move_batch_to_device
from nicr_mt_scene_analysis.data import mt_collate

from irsaformer.args import ArgParserIRSAFormer
from irsaformer.data import get_datahelper
from irsaformer.model import IRSAFormer
from irsaformer.preprocessing import get_preprocessor
from irsaformer.visualization import visualize_predictions
from irsaformer.weights import load_weights


def _get_dataset_type(dataset):
    dataset_cls = dataset.__class__
    bases = dataset_cls.__bases__
    if len(bases) > 1 and bases[1].__name__ != 'Dataset':
        return bases[1]
    return dataset_cls


def _get_max_instances_per_category(model):
    if 'panoptic_decoder' in model.decoders:
        postprocessing = model.decoders['panoptic_decoder'].postprocessing
    elif 'visual_embedding_decoder' in model.decoders:
        postprocessing = (
            model.decoders['visual_embedding_decoder'].postprocessing
        )
    else:
        raise RuntimeError(
            "Sample visualization requires a panoptic-capable decoder."
        )
    return int(postprocessing.max_instances_per_category)


def _get_args():
    parser = ArgParserIRSAFormer()

    # add additional arguments
    group = parser.add_argument_group('Inference')
    group.add_argument(    # useful for appm context module
        '--inference-input-height',
        type=int,
        default=480,
        dest='validation_input_height',    # used in test phase
        help="Network input height for predicting on inference data."
    )
    group.add_argument(    # useful for appm context module
        '--inference-input-width',
        type=int,
        default=640,
        dest='validation_input_width',    # used in test phase
        help="Network input width for predicting on inference data."
    )
    group.add_argument(
        '--depth-max',
        type=float,
        default=None,
        help="Additional max depth values. Values above are set to zero as "
             "they are most likely not valid. Note, this clipping is applied "
             "before scaling the depth values."
    )
    group.add_argument(
        '--depth-scale',
        type=float,
        default=1.0,
        help="Additional depth scaling factor to apply."
    )

    default_samples_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'samples'
    )
    group.add_argument(
        '--samples-path',
        type=str,
        default=default_samples_dir,
        help="Directory containing the samples."
    )
    group.add_argument(
        '--output-path',
        type=str,
        default=None,
        help="Directory to save the results."
    )
    group.add_argument(
        '--show-results',
        action='store_true',
        default=False,
        help="Show results in a window."
    )
    group.add_argument(
        '--panel-output',
        type=str,
        default=None,
        help="If set, save a combined RGB | Depth | Semantic | Panoptic panel "
             "to this image path (the README sample figure)."
    )
    group.add_argument(
        '--model-label',
        type=str,
        default=None,
        help="Backbone name shown in the panel title, e.g. "
             "'DINOv3-Base, RGB-D'."
    )

    return parser.parse_args()


def _load_img(fp):
    img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def main():
    args = _get_args()

    device = torch.device(args.device)

    # data and model
    data = get_datahelper(args)
    dataset_type = _get_dataset_type(data.dataset_train)
    dataset_configs = {
        dataset_type.__name__: data.dataset_config
    }
    model = IRSAFormer(args, dataset_configs=dataset_configs)

    # load weights
    print(f"Loading checkpoint: '{args.weights_filepath}'")
    checkpoint = torch.load(args.weights_filepath,
                            map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    if 'epoch' in checkpoint:
        print(f"-> Epoch: {checkpoint['epoch']}")
    # `_delta` marks a delta checkpoint whose backbone is stored as
    # (trained - pretrained); pass it so load_weights re-adds the upstream
    # backbone (mirrors main.py).
    delta_meta = checkpoint.get('_delta')
    load_weights(args, model, state_dict, verbose=True, delta_meta=delta_meta)

    # put the EoMT attention schedule at the position the checkpoint was saved
    # at (main.py does this via run.set_epoch for --validation-only). without it
    # the masked attention the model uses at inference stays disabled, which
    # degrades the predictions. progress mirrors run_helper: epoch / n_epochs.
    if 'epoch' in checkpoint and model.mask_attention_controller is not None:
        progress = int(checkpoint['epoch']) / max(1, args.n_epochs)
        progress = min(1.0, max(0.0, progress))
        model.update_mask_attention_controller(progress)

    torch.set_grad_enabled(False)
    model.eval()
    model.to(device)
    max_instances_per_category = _get_max_instances_per_category(model)

    # Sample images have no semantic/panoptic targets. Temporarily disable
    # task-aware target generators while building the input preprocessor.
    tasks = args.tasks
    args.tasks = ()
    preprocessor = get_preprocessor(
        args,
        dataset=data.datasets_valid[0],
        phase='test',
        multiscale_downscales=None
    )
    args.tasks = tasks

    # get samples
    basepath = args.samples_path
    # Files are assumed to be in an rgb and depth folder
    rgb_filepaths = sorted(glob(os.path.join(basepath, 'rgb', '*.*')))
    depth_filepaths = sorted(glob(os.path.join(basepath, 'depth', '*.*')))
    assert len(rgb_filepaths) == len(depth_filepaths)
    basenames_rgb = [os.path.basename(os.path.splitext(x)[0])
                     for x in rgb_filepaths]
    basenames_depth = [os.path.basename(os.path.splitext(x)[0])
                       for x in depth_filepaths]
    assert basenames_rgb == basenames_depth

    if args.output_path is not None:
        os.makedirs(args.output_path, exist_ok=True)

    # which task outputs to save / preview
    save_keys = (
        'semantic_segmentation_idx_fullres',
        'token_semantic_dense_idx_fullres',
        'token_panoptic_segmentation',
        # open-vocabulary outputs from the visual-embedding checkpoint (only
        # present when the model has the token-visual-embedding head)
        'token_visual_embedding_text_based_semantic_idx_fullres',
        'token_visual_embedding_text_based_panoptic_segmentation',
        'token_visual_embedding_visual_mean_based_panoptic_segmentation',
        'token_visual_embedding_linear_probing_panoptic_segmentation',
    )

    for fp_rgb, fp_depth in tqdm(zip(rgb_filepaths, depth_filepaths),
                                 total=len(rgb_filepaths)):
        # load rgb and depth image
        img_rgb = _load_img(fp_rgb)

        img_depth = _load_img(fp_depth).astype('float32')
        if args.depth_max is not None:
            img_depth[img_depth > args.depth_max] = 0
        img_depth *= args.depth_scale

        # preprocess sample
        sample = preprocessor({
            'rgb': img_rgb,
            'depth': img_depth,
            'identifier': os.path.basename(os.path.splitext(fp_rgb)[0])
        })

        # add batch axis as there is no dataloader
        batch = mt_collate([sample])
        batch = move_batch_to_device(batch, device=device)
        # the visual-embedding decoder's per-dataset linear-probing and scene
        # heads select weights by meta[i]['dataset_type'].__name__. samples
        # carry no dataset meta, so attach the loaded validation dataset type.
        if 'meta' not in batch:
            batch['meta'] = [{
                'dataset_type': _get_dataset_type(data.datasets_valid[0])
            }]

        # apply model
        predictions = model(batch, do_postprocessing=True)

        # visualize predictions
        preds_viz = visualize_predictions(
            predictions=predictions,
            batch=batch,
            dataset_config=data.dataset_config,
            max_instances_per_category=max_instances_per_category
        )

        if 'scene' in preds_viz:
            sample_id = os.path.basename(os.path.splitext(fp_rgb)[0])
            print(f"  {sample_id} -> scene: {preds_viz['scene'][0]}")

        if args.output_path is not None:
            for key in save_keys:
                if key not in preds_viz:
                    continue
                image = preds_viz[key][0]
                fp_out = os.path.join(
                    args.output_path,
                    key,
                    os.path.basename(fp_rgb)
                )
                os.makedirs(os.path.dirname(fp_out), exist_ok=True)
                if isinstance(image, Image.Image):
                    image.save(fp_out)
                elif isinstance(image, np.ndarray):
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(fp_out, image)

        # save a combined panel (RGB | Depth | Semantic | Panoptic) for the
        # README sample figure
        if args.panel_output is not None:
            depth_vis = img_depth.astype('float32')
            valid_d = depth_vis > 0
            if valid_d.any():
                lo = np.percentile(depth_vis[valid_d], 2)
                hi = np.percentile(depth_vis[valid_d], 98)
                norm = np.clip((depth_vis - lo) / max(hi - lo, 1e-6), 0, 1)
                depth_vis = (
                    plt.cm.viridis(norm)[..., :3] * 255
                ).astype('uint8')
                depth_vis[~valid_d] = 0
            panel_items = [('RGB', img_rgb), ('Depth', depth_vis)]
            for key, label in (('token_semantic_dense_idx_fullres', 'Semantic'),
                               ('token_panoptic_segmentation', 'Panoptic')):
                if key in preds_viz:
                    image = preds_viz[key][0]
                    panel_items.append(
                        (label,
                         np.array(image.convert('RGB'))
                         if isinstance(image, Image.Image) else image))
            fig, axs = plt.subplots(1, len(panel_items),
                                    figsize=(4 * len(panel_items), 3.4))
            for ax, (label, image) in zip(
                np.atleast_1d(axs).ravel(), panel_items
            ):
                ax.imshow(image)
                ax.set_title(label, fontsize=12)
                ax.axis('off')
            title = 'IRSAFormer'
            if args.model_label:
                title += f' ({args.model_label})'
            title += ' on a sample'
            fig.suptitle(title, fontsize=12, y=1.02)
            fig.tight_layout()
            os.makedirs(
                os.path.dirname(os.path.abspath(args.panel_output)),
                exist_ok=True)
            fig.savefig(args.panel_output, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  saved panel -> {args.panel_output}")

        # show results
        if args.show_results:
            panels = [('RGB', img_rgb), ('Depth', img_depth)]
            for key, title in (
                ('semantic_segmentation_idx_fullres', 'Semantic'),
                ('token_semantic_dense_idx_fullres', 'Semantic (token)'),
                ('token_panoptic_segmentation', 'Panoptic (token)'),
            ):
                if key in preds_viz:
                    panels.append((title, preds_viz[key][0]))

            n = len(panels)
            cols = min(n, 3)
            rows = (n + cols - 1) // cols
            _, axs = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows),
                                  dpi=150, squeeze=False)
            for ax in axs.ravel():
                ax.set_axis_off()
            for (title, image), ax in zip(panels, axs.ravel()):
                ax.set_title(title)
                ax.imshow(image, interpolation='nearest')

            suptitle = (
                f"Image: {os.path.basename(fp_rgb)}, "
                f"Model: {os.path.basename(args.weights_filepath)}"
            )
            if 'scene' in preds_viz:
                suptitle += f", Scene: {preds_viz['scene'][0]}"
            plt.suptitle(suptitle, fontsize=9)
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    main()
