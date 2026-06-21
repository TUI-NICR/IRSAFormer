# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import argparse as ap
import json
import os
import shlex
import shutil

from nicr_mt_scene_analysis.model.backbone import KNOWN_TOKEN_BACKBONES
from nicr_mt_scene_analysis.model.encoder_fusion import KNOWN_ENCODER_FUSIONS
from nicr_mt_scene_analysis.model.upsampling import KNOWN_UPSAMPLING_METHODS
from nicr_mt_scene_analysis.multi_task import KNOWN_TOKEN_TASKS
from nicr_scene_analysis_datasets.auxiliary_data.embedding_estimation import KNOWN_EMBEDDING_ESTIMATORS

from .data import KNOWN_CLASS_WEIGHTINGS
from .data import KNOWN_DATASETS
from .lr_scheduler import KNOWN_LR_SCHEDULERS
from .optimizer import KNOWN_OPTIMIZERS


MASK_DEPENDENT_TASKS = (
    'token-semantic',
    'token-panoptic',
    'token-orientation',
    'token-visual-embedding'
)


class ArgParserIRSAFormer(ap.ArgumentParser):
    def __init__(self, *args, **kwargs):
        # force ArgumentDefaultsHelpFormatter as formatter_class is given
        formatter_class = kwargs.pop('formatter_class', None)
        formatter_class = formatter_class or ap.ArgumentDefaultsHelpFormatter

        super().__init__(*args, formatter_class=formatter_class, **kwargs)

        # paths ---------------------------------------------------------------
        group = self.add_argument_group('Paths')
        group.add_argument(
             '--results-basepath',
             type=str,
             default='./results',
             help="Path where to store training files."
        )
        group.add_argument(
            '--weights-filepath',
            type=str,
            default=None,
            help="Filepath to (last) checkpoint / weights for the entire model."
        )

        # network and multi-task -----------------------------------------------
        group = self.add_argument_group('Tasks')
        # -> multi-task parameters
        group.add_argument(
            '--tasks',
            nargs='+',
            type=str,
            choices=KNOWN_TOKEN_TASKS,
            default=('token-mask', 'token-semantic'),
            help="Task(s) to perform."
        )
        # -> input
        group = self.add_argument_group('Input')
        group.add_argument(
            '--input-height',
            type=int,
            default=480,
            help="Network input height. Images will be resized to this height."
        )
        group.add_argument(
            '--input-width',
            type=int,
            default=640,
            help="Network input width. Images will be resized to this width."
        )
        group.add_argument(
            '--input-patch-height',
            type=int,
            default=16,
            help="Patch height supplied to token backbones."
        )
        group.add_argument(
            '--input-patch-width',
            type=int,
            default=16,
            help="Patch width supplied to token backbones."
        )
        group.add_argument(
            '--input-modalities',
            nargs='+',
            type=str,
            choices=('rgb', 'depth', 'rgbd'),
            default=('rgbd',),
            help="Input modalities to instantiate. 'rgbd' means one RGB-D "
                 "encoder with early fusion, while 'rgb depth' uses separate "
                 "RGB and depth encoders."
        )

        # -> whole model
        group = self.add_argument_group('Model')
        group.add_argument(
            '--compile-model',
            default=False,
            action='store_true',
            help="Enables compilation of the model."
        )
        group.add_argument(
            '--enable-tf32',
            action='store_true',
            default=False,
            help="Enables tf32 precision for the model on supported hardware."
                 "This can result in a speedup of the model and usualy does "
                 "not affect the model performance or numerical stability."
        )

        # -> encoder related parameters
        group = self.add_argument_group('Model: Encoder(s)')
        group.add_argument(
            '--no-pretrained-backbone',
            action='store_true',
            default=False,
            help="Disables loading of pretrained weights for the "
                 "backbone(s). Useful for inference or inference timing."
        )
        group.add_argument(
            '--encoder-backbone-pretrained-weights-filepath',
            type=str,
            default=None,
            help="Path to pretrained weights for the encoder "
                 "backbones. Use this argument if you want to initialize all "
                 "encoder backbones with the same weights. "
                 "If `weights-filepath` is given, the specified weights are "
                 "loaded subsequently and may replace the pretrained weights."
        )
        group.add_argument(
            '--encoder-fusion',
            choices=KNOWN_ENCODER_FUSIONS,
            default='se-add-uni-rgb',
            help="Determines how features of the depth (rgb) encoder are "
                 "fused to features of the other encoder."
        )
        group.add_argument(
            '--encoder-fusion-stage-indices',
            nargs='+',
            type=int,
            default=None,
            help="Stage indices at which to fuse RGB and depth features. "
                 "Only used when both RGB and depth encoders are active. "
                 "Indices refer to encoder stages (0 = patch embed for ViTs). "
                 "If None, fuse at all stages."
        )
        group.add_argument(
            '--encoder-fusion-stage-interval',
            type=int,
            default=None,
            help="Fuse at every N-th encoder stage when using RGB-D fusion. "
                 "Cannot be combined with --encoder-fusion-stage-indices."
        )
        group.add_argument(
            '--encoder-amp',
            type=str,
            default='fp16',
            choices=('disabled', 'fp16', 'bfp16'),
            help="Use automatic mixed precision (AMP) for faster training. "
                 "If fp16 or bfp16 is used, the encoder will be trained in "
                 "half precision. Note that this may lead to numerical "
                 "instabilities."
        )
        group = self.add_argument_group('Model: Encoder(s) -> Token settings')
        group.add_argument(
            '--token-modality',
            type=str,
            choices=('rgb', 'depth', 'rgbd'),
            default='rgbd',
            help="Modality whose encoder receives the injected token queries. "
                 "This must be present in --input-modalities."
        )
        group.add_argument(
            '--token-num-queries',
            type=int,
            default=128,
            help="Number of token queries injected into the selected modality."
        )
        group.add_argument(
            '--token-injection-start-block',
            type=int,
            default=-3,
            help="First transformer block receiving token queries. Negative "
                 "values count from the end, e.g. -3 injects into the last "
                 "three blocks."
        )
        group.add_argument(
            '--token-attention-mask-mode',
            choices=('none', 'eomt'),
            default='eomt',
            help="Controls if/which attention mask controller is used. "
                 "'eomt' uses the masked-attention strategy from EoMT "
                 "(Kerssies et al., CVPR 2025)."
        )
        group.add_argument(
            '--token-attention-mask-block-schedule',
            nargs='+',
            default=('0.1:0.3', '0.3:0.6', '0.6:0.9'),
            help="Per-block windows for attention masking. Specify `start:end` "
                 "pairs per injected block, e.g. "
                 "`0.1:0.3 0.3:0.6 0.6:0.9`. Length must match the number "
                 "of injected transformer blocks. Values are fractions of "
                 "training progress in [0, 1], i.e. 0.1 means 10%% of total "
                 "optimizer steps."
        )
        # -> rgb encoder
        group = self.add_argument_group('Model: Encoder(s) -> RGB encoder')
        group.add_argument(
            '--rgb-encoder-backbone',
            type=str,
            choices=KNOWN_TOKEN_BACKBONES,
            default='dinov3_small_plus_qkvb',
            help="Backbone to use for RGB encoder."
        )
        group.add_argument(
            '--rgb-encoder-backbone-lr-mult',
            type=float,
            default=0.1,
            help="Learning rate multiplier for the RGB backbone parameters "
                 "relative to the base learning rate."
        )
        group.add_argument(
            '--rgb-encoder-backbone-lr-start-frac',
            type=float,
            default=0.1,
            help="Fraction of total optimizer steps (0-1, e.g. 0.1 = 10%%) "
                 "before the RGB backbone begins warming up with "
                 "'warmup_poly'."
        )
        group.add_argument(
            '--rgb-encoder-backbone-lr-warmup-frac',
            type=float,
            default=0.2,
            help="Warmup duration as fraction of total optimizer steps (0-1, "
                 "e.g. 0.1 = 10%%) for the RGB backbone with 'warmup_poly'."
        )
        group.add_argument(
            '--rgb-encoder-backbone-pretrained-weights-filepath',
            type=str,
            default=None,
            help="Path to pretrained weights for the rgb encoder "
                 "backbone. "
                 "If `weights-filepath` is given, the specified weights are "
                 "loaded subsequently and may replace the pretrained weights."
        )
        group.add_argument(
            '--rgb-encoder-mlp-ratio',
            type=float,
            default=4.0,
            help="MLP hidden-to-input dimension ratio for non-EVA token ViTs."
        )
        group.add_argument(
            '--rgb-encoder-num-attention-heads',
            type=int,
            default=None,
            help="Number of attention heads for token ViT backbones. If "
                 "None, use the selected backbone default."
        )

        # -> depth encoder
        group = self.add_argument_group('Model: Encoder(s) -> depth encoder')
        group.add_argument(
            '--depth-encoder-backbone',
            type=str,
            choices=KNOWN_TOKEN_BACKBONES,
            default='dinov3_small_plus_qkvb',
            help="Backbone to use for depth encoder."
        )
        group.add_argument(
            '--depth-encoder-backbone-lr-mult',
            type=float,
            default=0.2,
            help="Learning rate multiplier for the depth backbone parameters "
                 "relative to the base learning rate."
        )
        group.add_argument(
            '--depth-encoder-backbone-lr-start-frac',
            type=float,
            default=0.1,
            help="Fraction of total optimizer steps (0-1, e.g. 0.1 = 10%%) "
                 "before the depth backbone begins warming up with "
                 "'warmup_poly'."
        )
        group.add_argument(
            '--depth-encoder-backbone-lr-warmup-frac',
            type=float,
            default=0.2,
            help="Warmup duration as fraction of total optimizer steps (0-1, "
                 "e.g. 0.1 = 10%%) for the depth backbone with 'warmup_poly'."
        )
        group.add_argument(
            '--depth-encoder-backbone-pretrained-weights-filepath',
            type=str,
            default=None,
            help="Path to pretrained weights for the depth encoder "
                 "backbone. "
                 "If `weights-filepath` is given, the specified weights are "
                 "loaded subsequently and may replace the pretrained weights."
        )
        group.add_argument(
            '--depth-encoder-mlp-ratio',
            type=float,
            default=4.0,
            help="MLP hidden-to-input dimension ratio for non-EVA token ViTs."
        )
        group.add_argument(
            '--depth-encoder-num-attention-heads',
            type=int,
            default=None,
            help="Number of attention heads for token ViT backbones. If "
                 "None, use the selected backbone default."
        )

        # -> rgbd encoder
        group = self.add_argument_group('Model: Encoder(s) -> RGB-D encoder')
        group.add_argument(
            '--rgbd-encoder-backbone',
            type=str,
            choices=KNOWN_TOKEN_BACKBONES,
            default='dinov3_base_qkvb',
            help="Backbone to use for RGB-D encoder."
        )
        group.add_argument(
            '--rgbd-encoder-backbone-lr-mult',
            type=float,
            default=0.2,
            help="Learning rate multiplier for the RGB-D backbone parameters "
                 "relative to the base learning rate."
        )
        group.add_argument(
            '--rgbd-encoder-backbone-lr-start-frac',
            type=float,
            default=0.1,
            help="Fraction of total optimizer steps (0-1, e.g. 0.1 = 10%%) "
                 "before the RGB-D backbone begins warming up with "
                 "'warmup_poly'."
        )
        group.add_argument(
            '--rgbd-encoder-backbone-lr-warmup-frac',
            type=float,
            default=0.2,
            help="Warmup duration as fraction of total optimizer steps (0-1, "
                 "e.g. 0.1 = 10%%) for the RGB-D backbone with 'warmup_poly'."
        )
        group.add_argument(
            '--rgbd-encoder-backbone-pretrained-weights-filepath',
            type=str,
            default=None,
            help="Path to pretrained weights for the RGB-D encoder "
                 "backbone. "
                 "If `weights-filepath` is given, the specified weights are "
                 "loaded subsequently and may replace these pretrained weights."
        )
        group.add_argument(
            '--rgbd-encoder-mlp-ratio',
            type=float,
            default=4.0,
            help="MLP hidden-to-input dimension ratio for non-EVA token ViTs."
        )
        group.add_argument(
            '--rgbd-encoder-num-attention-heads',
            type=int,
            default=None,
            help="Number of attention heads for token ViT backbones. If "
                 "None, use the selected backbone default."
        )
        group.add_argument(
            '--rgbd-encoder-backbone-depth-embed-lr-mult',
            type=float,
            default=1.0,
            help="Learning rate multiplier for the depth convolution in the "
                 "patch embedding."
        )

        # -> decoder related parameters
        group = self.add_argument_group('Model: Decoder(s)')
        group.add_argument(
            '--upsampling-prediction',
            choices=KNOWN_UPSAMPLING_METHODS,
            default='bilinear',
            help="How features are upsampled after the last decoder module to "
                 "match the network input resolution."
        )
        group.add_argument(
            '--decoder-amp',
            type=str,
            default='fp16',
            choices=('disabled', 'fp16', 'bfp16'),
            help="Use automatic mixed precision (AMP) for faster training of "
                 "the decoder. If fp16 or bfp16 is used, the decoder will be "
                 "trained in half precision. Note that this may lead to "
                 "numerical instabilities and is not recommend for the trainig "
                 "of some tasks (e.g. semanitc segmentation) decoder!"
                 "Please verify at your own that it dosn't lead to "
                 "bad performance!"
        )
        # -> mask decoder related parameters
        group.add_argument(
            '--mask-decoder-upsampling',
            choices=('transposed-conv',),
            default='transposed-conv',
            help="Upsampling mode for each token mask decoder stage. "
                 "Currently only 'transposed-conv' is supported."
        )

        # -> token embedding related parameters
        group = self.add_argument_group('Model: Decoder(s) -> Token Embedding')
        group.add_argument(
            '--token-visual-embedding-diff-factor',
            type=float,
            default=0.65,
            help="Factor to use for subtracting image embedding from panoptic "
                 "embeddings. Set to 0 to disable this preprocessing step."
        )
        group.add_argument(
            '--token-embedding-reference-estimator',
            type=str,
            default='alpha_clip__l14-336-grit-20m',
            help="Embeddings vectors which where created by the given "
                 "estimator",
            choices=KNOWN_EMBEDDING_ESTIMATORS
        )
        # training ------------------------------------------------------------
        group = self.add_argument_group('Training')
        group.add_argument(
            '--loss-amp',
            type=str,
            default='disabled',
            choices=('disabled', 'fp16', 'bfp16'),
            help="Use automatic mixed precision (AMP) for faster training of "
                 "the loss functions. If fp16 or bfp16 is used, the loss "
                 "functions will be trained in half precision. Note that "
                 "this may lead to numerical instabilities."
        )

        group.add_argument(
            '--n-epochs',
            type=int,
            default=25,
            help="Number of epochs to train for."
        )
        group.add_argument(
            '--batch-size',
            type=int,
            default=8,
            help="Batch size to use for training."
        )
        group.add_argument(
            '--optimizer',
            type=str,
            choices=KNOWN_OPTIMIZERS,
            default='adamw',
            help="Optimizer to use."
        )
        group.add_argument(
            '--learning-rate',
            type=float,
            default=2.5e-4,
            help="Maximum learning rate for an effective batch size of 8 "
                 "(effective batch = `batch-size` * "
                 "`gradient-accumulation-steps`). When using a different "
                 "effective batch size, the learning rate is scaled "
                 "automatically: lr = `learning-rate` * effective_batch/8."
        )
        group.add_argument(
            '--learning-rate-scheduler',
            type=str,
            choices=KNOWN_LR_SCHEDULERS,
            default='warmup_poly',
            help="Learning rate scheduler to use. For parameters and details, "
                 "see implementation."
        )
        group = self.add_argument_group(
            'Learning rate scheduler -> WarmupPoly'
        )
        group.add_argument(
            '--lr-warmup-poly-power',
            type=float,
            default=0.9,
            help="Polynomial decay power applied after warmup when using "
                 "'warmup_poly' scheduler."
        )
        group.add_argument(
            '--non-backbone-lr-start-frac',
            type=float,
            default=0.0,
            help="Fraction of total optimizer steps (0-1, e.g. 0.1 = 10%%) "
                 "before non-backbone parameters begin warming up with "
                 "'warmup_poly'."
        )
        group.add_argument(
            '--non-backbone-lr-warmup-frac',
            type=float,
            default=0.15,
            help="Warmup duration as fraction of total optimizer steps (0-1, "
                 "e.g. 0.1 = 10%%) for non-backbone parameters with "
                 "'warmup_poly'."
        )
        group.add_argument(
            '--momentum',
            type=float,
            default=0.9,
            help="Momentum to use."
        )
        group.add_argument(
            '--weight-decay',
            type=float,
            default=5e-2,
            help="Weight decay to use for all network weights."
        )
        group.add_argument(
            '--gradient-clip-norm',
            type=float,
            default=0.0,
            help="Max L2 norm applied via gradient clipping (<=0 disables "
                 "clipping)."
        )
        group.add_argument(
            '--gradient-accumulation-steps',
            type=int,
            default=1,
            help="Number of mini-batches to accumulate before performing an "
                 "optimizer step."
        )
        group.add_argument(
            '--debug-print-optimizer-groups',
            action='store_true',
            default=False,
            help="Print learning-rate/parameter counts for each optimizer "
                 "group at startup."
        )
        group.add_argument(
            '--tasks-weighting',
            nargs='+',
            type=float,
            default=None,
            help="Task weighting to use for loss balancing. The tasks' "
                 "weights are assigned to the task in the order given by "
                 "`tasks`."
        )
        group.add_argument(
            '--no-multistage-supervision',
            action='store_true',
            default=False,
            help="Disables multi-stage supervision (auxiliary losses)."
        )

        # -> token semantic related parameters
        group = self.add_argument_group('Token Semantic')
        group.add_argument(
            '--token-semantic-class-weighting',
            type=str,
            choices=KNOWN_CLASS_WEIGHTINGS,
            default='none',
            help="Weighting mode to use for semantic classes to balance loss "
                 "during training"
        )
        group.add_argument(
            '--token-semantic-class-weighting-logarithmic-c',
            type=float,
            default=1.02,
            help="Parameter c for limiting the upper bound of the class "
                 "weights when class weighting is 'logarithmic'. "
                 "Logarithmic class weighting is defined as 1 / ln(c+p_class)."
        )
        group.add_argument(
            '--token-semantic-loss-label-smoothing',
            type=float,
            default=0.0,
            help="Label smoothing factor for the token semantic loss."
        )

        # -> token mask related parameters
        group = self.add_argument_group('Token Mask')
        group.add_argument(
            '--token-mask-match-with-semantic-class',
            action='store_true',
            default=False,
            help="Include semantic class logits in the token mask Hungarian "
                 "matcher to better mimic Mask2Former training."
        )
        group.add_argument(
            '--token-mask-semantic-matcher-class-weight',
            type=float,
            default=4.0,
            help="Classification cost assigned to semantic queries when mask "
                 "matching uses semantic logits."
        )
        group.add_argument(
            '--token-mask-unmatched-weight',
            type=float,
            default=0.6,
            help="Weight for the unmatched mask sink loss. "
                 "Setting this > 0 penalizes unmatched mask logits."
        )
        group.add_argument(
            '--token-mask-unmatched-topk-frac',
            type=float,
            default=0.2,
            help="Fraction of unmatched queries to penalize (top-k by mask "
                 "logit). Use values in (0, 1] to reduce compute, e.g. 0.25."
        )

        # -> token panoptic related parameters
        group = self.add_argument_group('Token Panoptic')
        group.add_argument(
            '--token-panoptic-mask-threshold',
            type=float,
            default=0.1,
            help="Mask probability threshold used by token panoptic "
                 "postprocessing. Pixels below this value stay void."
        )
        group.add_argument(
            '--token-panoptic-mask-pixel-threshold',
            type=float,
            default=0.2,
            help="Pixel-level sigmoid threshold for token panoptic masks. "
                 "Pixels below this value are ignored when forming segments."
        )
        group.add_argument(
            '--token-panoptic-overlap-threshold',
            type=float,
            default=0.6,
            help="Overlap threshold for token panoptic postprocessing."
        )
        group.add_argument(
            '--token-panoptic-loss-label-smoothing',
            type=float,
            default=0.0,
            help="Label smoothing factor for the token panoptic class loss."
        )
        group.add_argument(
            '--token-panoptic-norm-margin',
            type=float,
            default=1.0,
            help="Margin for the token panoptic norm sink loss."
        )
        group.add_argument(
            '--token-panoptic-norm-coefficient',
            type=float,
            default=5e-2,
            help="Weight for the token panoptic norm sink loss."
        )

        # -> token visual embedding related parameters
        group = self.add_argument_group('Token Visual Embedding')
        group.add_argument(
            '--token-visual-embedding-mask-threshold',
            type=float,
            default=0.0,
            help="Mask probability threshold used by token visual embedding "
                 "panoptic postprocessing. Pixels below this value stay void."
        )
        group.add_argument(
            '--token-visual-embedding-mask-pixel-threshold',
            type=float,
            default=0.70,
            help="Pixel-level sigmoid threshold for token visual embedding "
                 "panoptic masks. Pixels below this value are ignored when "
                 "forming segments. Retained NYUv2 default: 0.70."
        )
        group.add_argument(
            '--token-visual-embedding-overlap-threshold',
            type=float,
            default=0.60,
            help="Minimum retained-area ratio for token visual embedding "
                 "panoptic postprocessing after mask competition. "
                 "NYUv2 default: 0.60."
        )
        group.add_argument(
            '--token-visual-embedding-query-mask-score-exponent',
            type=float,
            default=0.75,
            help="Exponent applied to `mask_scores.clamp_min(0)` before it is "
                 "multiplied into token visual embedding panoptic query "
                 "scores. Set to 1.0 to disable the power transform. "
                 "Retained NYUv2 default: 0.75."
        )
        group.add_argument(
            '--token-visual-embedding-similarity-score-scale',
            type=float,
            default=20.0,
            help="Softmax scale applied to token visual embedding text/visual "
                 "similarity scores before panoptic query ranking and class "
                 "selection. Higher values sharpen class confidence."
        )
        group.add_argument(
            '--token-visual-embedding-loss-coefficient',
            type=float,
            default=2.0,
            help="Weight for the token visual embedding loss."
        )
        group.add_argument(
            '--token-visual-embedding-negative-coefficient',
            type=float,
            default=0.0,
            help="Weight for negative token visual embedding pairs."
        )
        group.add_argument(
            '--token-visual-embedding-negative-margin',
            type=float,
            default=0.0,
            help="Margin for negative token visual embedding pairs."
        )
        group.add_argument(
            '--token-visual-embedding-norm-margin',
            type=float,
            default=1.0,
            help="Margin for the token visual embedding norm sink loss."
        )
        group.add_argument(
            '--token-visual-embedding-norm-coefficient',
            type=float,
            default=5e-2,
            help="Weight for the token visual embedding norm sink loss."
        )
        group.add_argument(
            '--token-visual-embedding-disable-aux-loss',
            action='store_true',
            default=False,
            help="Disable auxiliary losses for token visual embedding tasks."
        )
        group.add_argument(
            '--token-visual-embedding-mlp-hidden-dim',
            type=int,
            default=1024,
            help="Hidden dimension for the token visual embedding MLP head."
        )
        group.add_argument(
            '--token-visual-embedding-mlp-dropout',
            type=float,
            default=0.0,
            help="Dropout probability for the token visual embedding MLP head."
        )

        # -> token embedding related parameters
        group = self.add_argument_group('Token Embedding')
        group.add_argument(
            '--token-embedding-do-not-compute-mean-embedding',
            action='store_true',
            default=False,
            help="Disables computing the per class mean embedding vectors "
                 "which are used for computing the visual mean-based metrics."
        )

        # -> token scene related parameters
        group = self.add_argument_group('Token Scene')
        group.add_argument(
            '--token-scene-loss-label-smoothing',
            type=float,
            default=0.1,
            help="Label smoothing factor for the token scene loss."
        )

        # dataset and augmentation --------------------------------------------
        group = self.add_argument_group('Dataset and Augmentation')
        group.add_argument(
            '--dataset',
            type=str,
            default='nyuv2',
            help="Dataset(s) to train/validate on. "
                 "Use ':' to combine multiple datasets, e.g., 'nyuv2:sunrgbd'. "
                 "Note that the first dataset is used in this case  for "
                 "determining relevant dataset/network/training parameters. "
                 "Moreover, use 'dataset[camera,camera4]' to select specific "
                 "cameras of an dataset. "
                 "To select a specific depth estimator, use: "
                 "'dataset^depth_estimator', e.g., "
                 "'ade20k^depthanything_v2__indoor_large'. Note that the depth "
                 "images must be precomputed. "
                 f"Available datasets: {', '.join(KNOWN_DATASETS)}."
        )
        group.add_argument(
            '--dataset-path',
            type=str,
            default=None,
            help="Path(s) to dataset root(s). If not given, the path is "
                 "determined automatically using the distributed training "
                 "package. If no path(s) can be determined, data loading is "
                 "disabled. Use ':' to combine the paths for combined datasets."
        )
        group.add_argument(
            '--split',
            type=str,
            default='train',
            help="Dataset split(s) to use for training. Use ':' to combine "
                 "the splits for combined datasets."
        )
        group.add_argument(
            '--raw-depth',
            action='store_true',
            default=False,
            help="Whether to use the raw depth values instead of refined "
                 "depth values."
        )
        group.add_argument(
            '--scale-depth',
            action='store_true',
            default=False,
            help="Whether to use scaled depth values (each sample scaled to"
                 "[0, 1] independently) instead of standardized depth "
                 "values (mean: 0.0, std: 1.0 across all samples)."
        )
        group.add_argument(
            '--use-original-scene-labels',
            action='store_true',
            default=False,
            help="Do not use unified scene class labels for domestic indoor "
                 "environments (Hypersim, NYUv2, ScanNet, and SUNRGB-D only)."
        )
        group.add_argument(
            '--aug-scale-min',
            type=float,
            default=1.0,
            help="Minimum scale for random rescaling during training."
        )
        group.add_argument(
            '--aug-scale-max',
            type=float,
            default=1.5,
            help="Maximum scale for random rescaling during training."
        )
        group.add_argument(
            '--cache-dataset',
            action='store_true',
            default=False,
            help="Cache dataset in RAM to speed up training."
        )
        group.add_argument(
            '--n-workers',
            type=int,
            default=8,
            help="Number of workers for data loading and preprocessing"
        )
        group.add_argument(
            '--no-persistent-worker',
            action='store_true',
            default=False,
            help="Disable persistent worker processes for the dataloaders."
        )
        group.add_argument(
            '--validation-resize-keep-aspect-ratio',
            action='store_true',
            default=False,
            help="Whether to keep the aspect ratio when resizing inputs during "
                 "validation."
        )
        group.add_argument(
            '--validation-resize-padding-mode',
            type=str,
            default='zero',
            choices=('zero', 'reflect'),
            help="Padding mode to use when resizing inputs during validation "
                 "and enabled `--validation-resize-keep-aspect-ratio`."
        )
        group.add_argument(
            '--subset-train',
            type=str,
            default='1.0',
            help="Limit train data to a subset, e.g., '0.1' will limit the "
                 "data for training to only 10 percent of the total number of "
                 "samples, '1.0' will use all data. Use ':' to combine"
                 "subset parameters for concatenated datasets."
                 "Note, the subset is chosen randomly each epoch, except if "
                 "`subset-deterministic` is passed."
        )
        group.add_argument(
            '--subset-deterministic',
            action='store_true',
            default=False,
            help="Use the same subset in each epoch and across different "
                 "training runs. Requires `subset-train` to be set."
        )

        group = self.add_argument_group('Dataset and Augmentation -> ScanNet')
        # -> NYUv2 related parameters
        group.add_argument(
            '--nyuv2-semantic-n-classes',
            type=int,
            default=40,
            choices=(13, 40, 894),
            help="Number of semantic classes to use for NYUv2 dataset."
        )
        # -> ScanNet related parameters
        group.add_argument(
            '--scannet-subsample',
            type=int,
            default=50,
            choices=(50, 100, 200, 500),
            help="Subsample to use for ScanNet dataset for training."
        )
        group.add_argument(
            '--scannet-semantic-n-classes',
            type=int,
            default=40,
            choices=(20, 40, 200, 549),
            help="Number of semantic classes to use for ScanNet dataset."
        )
        # -> SUNRGB-D related parameters
        group = self.add_argument_group('Dataset and Augmentation -> SUNRGB-D')
        group.add_argument(
            '--sunrgbd-depth-do-not-force-mm',
            action='store_true',
            default=False,
            help="Do not force mm for SUNRGB-D depth values. Use this option "
                 "to evaluate weights of the EMSANet paper on SUNRGB-D."
        )
        group.add_argument(
            '--sunrgbd-instances-version',
            type=str,
            default='panopticndt',
            choices=('emsanet', 'panopticndt', 'anyold'),
            help="We have created two versions of SUNRGB-D with instance "
                 "annotations extracted from existing 3d-box annotations: "
                 "'emsanet': this initial version was created for training the "
                 "original EMSANet - see IJCNN 2022 paper; "
                 "'panopticndt': referes to a revised version that was created "
                 "along with the work for PanopticNDT - see IROS 2023 paper, "
                 "it refines large parts of the instance extraction (see "
                 "changelog for v0.6.0 of the nicr-scene-analysis-datasets "
                 "package for details). "
                 "Additionally, 'anyold' can be used can be used to force "
                 "loading any SUNRGB-D dataset prepared with a dataset package "
                 "version < v0.7.0; however, use this value only if you know "
                 "what you are doing!"
        )
        # -> Hypersim related parameters
        # TODO: can be removed from codebase later
        group = self.add_argument_group('Dataset and Augmentation -> Hypersim')
        group.add_argument(
            '--hypersim-use-old-depth-stats',
            action='store_true',
            default=False,
            help="Use old (v030) depth stats for Hypersim dataset. Enable "
                 "this argument if you load weights created earlier than "
                 "Apr. 28, 2022."
        )
        group.add_argument(
            '--hypersim-subsample',
            type=int,
            default=1,
            choices=(1, 2, 5, 10, 20),
            help="Subsample to use for Hypersim dataset for training."
        )

        # validation/evaluation ------------------------------------------------
        group = self.add_argument_group('Validation/Evaluation')
        group.add_argument(
            '--validation-only',
            action='store_true',
            default=False,
            help="No training, validation only. Requires `weights-filepath`."
        )
        group.add_argument(
            '--visualize-validation',
            default=False,
            action='store_true',
            help="Whether the validation should be visualized. Consider adding "
                 "`--debug` to visalize the ground-truth side outputs as well."
        )
        group.add_argument(
            '--visualization-output-path',
            type=str,
            default=None,
            help="Path where to save visualized predictions. By default, a "
                 "new directory is created in the directory where the weights "
                 "come from. The filename of the weights is included in the "
                 "name of the visualization directory, so that it is evident "
                 "which weights have led to these visualizations."
        )
        group.add_argument(    # useful for appm context module
            '--validation-input-height',
            type=int,
            default=None,
            help="Network input height for validation. Images will be resized "
                 "to this height. If not given, `input-height` is used (same "
                 "height for training and validation). "
        )
        group.add_argument(    # useful for appm context module
            '--validation-input-width',
            type=int,
            default=None,
            help="Network input width for validation. Images will be resized "
                 "to this width. If not given, `input-width` is used (same "
                 "width for training and validation)."
        )
        group.add_argument(
            '--validation-batch-size',
            type=int,
            default=1,
            help="Batch size to use for validation."
        )
        group.add_argument(
            '--validation-split',
            type=str,
            default='valid',
            help="Dataset split(s) to use for validation. Use ':' to combine "
                 "the splits for combined datasets."
        )
        group.add_argument(
            '--validation-skip',
            type=float,
            default=0.8,
            help="Skip validation (metric calculation, example creation, and "
                 "checkpointing) in early epochs. For example, passing a"
                 "value of '0.2' and `n_epochs` of '500', skips validation "
                 "for the first 0.2*500 = 100 epochs. A value of '1.0' "
                 "disables validation at all."
        )
        group.add_argument(
            '--validation-force-interval',
            type=int,
            default=20,
            help="Force validation after every X epochs even when using "
                 "`validation-skip`. This allows to still see progress and "
                 "save checkpoints during training."
        )
        group.add_argument(
            '--validation-full-resolution',
            action='store_true',
            default=False,
            help="Whether to validate on full-resolution inputs (do not apply "
                 "any resizing to the inputs, for Cityscapes or "
                 "Hypersim dataset)."
        )
        # -> ScanNet related parameters
        group = self.add_argument_group('Validation/Evaluation -> ScanNet')
        group.add_argument(
            '--validation-scannet-subsample',
            type=int,
            default=100,
            choices=(50, 100, 200, 500),
            help="Subsample to use for ScanNet dataset for validation."
        )
        group.add_argument(
            '--validation-scannet-benchmark-mode',
            action='store_true',
            default=False,
            help="Enable benchmark mode for validation on ScanNet dataset, "
                 "i.e., mapping ignored classes to void "
                 "(`scannet-semantic-n-classes`=40/549 only)."
        )
        # -> checkpointing
        group = self.add_argument_group(
            'Validation/Evaluation -> Checkpointing'
        )
        group.add_argument(
            '--checkpointing-metrics',
            nargs='+',
            type=str,
            default=None,
            help="Metric(s) to use for checkpointing. For example "
                 "'miou bacc miou+bacc' leads to checkpointing when either "
                 "miou, bacc, or the sum of miou and bacc reaches its highest "
                 "value. Note that current implemention only supports "
                 "combining metrics using '+'. Omitted this parameter "
                 "disables checkpointing."
        )
        group.add_argument(
            '--checkpointing-best-only',
            action='store_true',
            default=False,
            help="Store only the best checkpoint."
        )
        group.add_argument(
            '--checkpointing-skip',
            type=float,
            default=0.0,
            help="Skip checkpointing in early epochs. For example, passing a"
                 "value of '0.2' and `n_epochs` of '500', skips checkpointing "
                 "for the first 0.2*500 = 100 epochs. A value of '1.0' "
                 "disables checkpointing at all."
        )

        # resuming ------------------------------------------------------------
        subparsers = self.add_subparsers(
            parser_class=ap.ArgumentParser,     # important to avoid recursion
            dest='action'
        )
        subparser = subparsers.add_parser(
            'resume',
            help="Resume previous training run with auto argument and "
                 "checkpoint detection, see `path` argument for details."
        )
        subparser.add_argument(
            'resume_path',
            type=str,
            default=None,
            help="Path to previous training run, e.g., './runs_xy'. All args "
                 "are automatically replaced with the args given in "
                 "'./runs_xy/argsv.txt'. Furthermore, `--resume-ckpt-filepath` "
                 "is set to './runs_xy/checkpoints/ckpt_resume.pth'. For "
                 "safety, a backup of the given run folder is created."
        )

        self.add_argument(
            '--resume-ckpt-filepath',
            type=str,
            default=None,
            help="Path to checkpoint file to resume training from. "
        )

        self.add_argument(
            '--resume-ckpt-interval',
            type=int,
            default=20,
            help="Write resume checkpoint containing state dicts for model, "
                 "optimizer, and lr scheduler every X epochs. "
                 "This allows resuming a previous training."
        )

        # debugging -----------------------------------------------------------
        self.add_argument(
            '--debug',
            action='store_true',
            default=False,
            help="Enables debug outputs (and exporting the model to ONNX)."
        )
        self.add_argument(
            '--skip-sanity-check',
            action='store_true',
            default=False,
            help="Disables the simple sanity check before training that "
                 "ensures that crucial parts (data, forward, metrics, ...) are "
                 "working as expected. The check is done by forwarding a "
                 "single batch of all dataloadera WITHOUT backpropagation."
        )
        self.add_argument(
            '--overfit-n-batches',
            type=int,
            default=-1,
            help="Forces to overfit on specified number of batches. Note that "
                 "for both training and validation samples are drawn from the "
                 "(first) validation loader without shuffling."
        )
        # Weights & Biases ----------------------------------------------------
        self.add_argument(
            '--wandb-mode',
            type=str,
            choices=('online', 'offline', 'disabled'),     # see wandb
            default='online',
            help="Mode for Weights & Biases"
        )
        self.add_argument(
            '--wandb-project',
            type=str,
            default='IRSAFormer',
            help="Project name for Weights & Biases"
        )

        # other parameters ----------------------------------------------------
        self.add_argument(
            '--device',
            type=str,
            default='cuda',
            choices=('cuda', 'cpu', 'mps'),
            help="Device to use for training and validation."
        )
        self.add_argument(
            '--disable-progress-bars',
            action='store_true',
            default=False,
            help="Disables all tqdm progress bars (currently only in main.py)."
        )
        self.add_argument(
            '--seed',
            type=int,
            default=None,
            help="Optional random seed applied to Python, NumPy, and Torch "
                 "RNGs."
        )

    def parse_args(self, args=None, namespace=None, verbose=True):
        # parse args
        pa = super().parse_args(args=args, namespace=namespace)

        def _warn(text):
            if verbose:
                print(f"[Warning] {text}")

        # check for resumed training ------------------------------------------
        if 'resume' == pa.action:
            is_resumed_training = True
            resume_path = pa.resume_path

            # load args from file
            args_fp = os.path.join(pa.resume_path, 'argsv.txt')
            print(f"Resuming training with args from: '{args_fp}'.")
            with open(args_fp, 'r') as f:
                args_str = f.read().strip()
            args_run = shlex.split(args_str)[1:]     # remove script name

            # create a backup of the given run folder
            backup_number = 1
            while 0 != backup_number:    # was not successful
                backup_path = os.path.normpath(pa.resume_path)  # trailing /
                backup_path += f'_before_resume{backup_number}'

                if os.path.isdir(backup_path):
                    print(f"Found already existing backup: '{backup_path}'")
                    backup_number += 1
                    continue

                print(f"Creating backup at: '{backup_path}'.")
                shutil.copytree(src=pa.resume_path, dst=backup_path)
                break

            # set resume checkpoint filepath
            ckpt_fp = os.path.join(pa.resume_path, 'checkpoints',
                                   'ckpt_resume.pth')
            assert os.path.isfile(ckpt_fp)
            args_run.extend(
                shlex.split(f'--resume-ckpt-filepath {shlex.quote(ckpt_fp)}')
            )
            # parse args again
            pa = super().parse_args(args=args_run, namespace=namespace)
        else:
            is_resumed_training = False
            resume_path = None

        # store additional information
        pa.resume_path = resume_path
        pa.is_resumed_training = is_resumed_training

        # convert nargs+ arguments from lists to tuples -----------------------
        for k, v in dict(vars(pa)).items():
            if isinstance(v, list):
                setattr(pa, k, tuple(v))

        # perform some initial argument checks
        token_tasks_active = any(t.startswith('token-') for t in pa.tasks)
        if token_tasks_active and pa.token_modality not in pa.input_modalities:
            raise ValueError(
                f"`--token-modality {pa.token_modality}` requires the "
                "selected modality to be present in `--input-modalities`."
            )

        # weights filepaths ---------------------------------------------------
        if pa.encoder_backbone_pretrained_weights_filepath is not None:
            # check if filepaths for rgb and depth are not set
            if any((
                pa.rgb_encoder_backbone_pretrained_weights_filepath is not None,
                (
                    pa.depth_encoder_backbone_pretrained_weights_filepath
                    is not None
                ),
                pa.rgbd_encoder_backbone_pretrained_weights_filepath is not None
            )):
                raise ValueError(
                    "Only use `encoder-backbone-pretrained-weights-filepath` "
                    "if you want to initialize all used encoder backbones with "
                    "the same weights! "
                    "`rgb-encoder-backbone-pretrained-weights-filepath` and "
                    "`depth-encoder-backbone-pretrained-weights-filepath` and "
                    "`rgbd-encoder-backbone-pretrained-weights-filepath` must"
                    "not be set."
                )
            pa.rgb_encoder_backbone_pretrained_weights_filepath = \
                pa.encoder_backbone_pretrained_weights_filepath
            pa.depth_encoder_backbone_pretrained_weights_filepath = \
                pa.encoder_backbone_pretrained_weights_filepath
            pa.rgbd_encoder_backbone_pretrained_weights_filepath = \
                pa.encoder_backbone_pretrained_weights_filepath
        # this argument is not needed anymore
        del pa.encoder_backbone_pretrained_weights_filepath

        # disable encoder fusion if only one input modality is used
        if 1 == len(pa.input_modalities):
            pa.encoder_fusion = 'none'
            _warn("Set `encoder-fusion` to 'none' as there is only one input "
                  "modality.")

        # multi-task parameters -----------------------------------------------
        # check if any token- tasks are used
        if token_tasks_active:

            # check if additional token tasks are required
            required = set()
            for t in pa.tasks:
                if t in MASK_DEPENDENT_TASKS:
                    required.add('token-mask')
                if t == 'token-orientation':
                    required.add('token-panoptic')

            missing = required.difference(pa.tasks)
            if missing:
                missing_str = "', '".join(sorted(missing))
                raise ValueError(
                    "The selected token task requires the following tasks as "
                    f"well: '{missing_str}'."
                )
            if 'token-mask' not in pa.tasks:
                if pa.token_attention_mask_mode != 'none':
                    pa.token_attention_mask_mode = 'none'
                    _warn(
                        "Set `token-attention-mask-mode` to 'none' as no "
                        "token mask decoder is used."
                    )
                pa.token_attention_mask_block_schedule = None
                custom_schedule = None
            else:
                custom_schedule = pa.token_attention_mask_block_schedule
            if custom_schedule is None:
                pa.token_attention_mask_block_schedule = None
            else:
                parsed_schedule = []
                for window in custom_schedule:
                    start_str, end_str = window.split(':', 1)
                    start_val = float(start_str)
                    end_val = float(end_str)
                    assert end_val >= start_val, \
                        "Each block-specific schedule must have end >= start."
                    if not (0.0 <= start_val <= 1.0 and 0.0 <= end_val <= 1.0):
                        raise ValueError(
                            "Attention mask schedule values must be in [0, 1]."
                        )
                    parsed_schedule.append((start_val, end_val))
                pa.token_attention_mask_block_schedule = tuple(parsed_schedule)

        # training ------------------------------------------------------------
        if pa.gradient_accumulation_steps < 1:
            raise ValueError("gradient-accumulation-steps must be >= 1.")

        if (
            pa.encoder_fusion_stage_indices is not None
            and pa.encoder_fusion_stage_interval is not None
        ):
            raise ValueError(
                "Use only one of '--encoder-fusion-stage-indices' or "
                "'--encoder-fusion-stage-interval'."
            )
        if pa.encoder_fusion_stage_interval is not None:
            if pa.encoder_fusion_stage_interval < 1:
                raise ValueError(
                    "'--encoder-fusion-stage-interval' must be >= 1."
                )

        reference_batch_size = 8
        effective_batch_size = (
            pa.batch_size * pa.gradient_accumulation_steps
        )
        if effective_batch_size != reference_batch_size:
            # the provided learning rate refers to the reference effective
            # batch size. Scale when deviating.
            pa.learning_rate = (
                pa.learning_rate * effective_batch_size / reference_batch_size
            )
            _warn(
                "Adapted learning rate to "
                f"'{pa.learning_rate}' for effective batch size "
                f"{effective_batch_size} (batch_size={pa.batch_size}, "
                f"grad_accum={pa.gradient_accumulation_steps})."
            )

        if pa.tasks_weighting is None:
            pa.tasks_weighting = (1,)*len(pa.tasks)

        if len(pa.tasks_weighting) != len(pa.tasks):
            raise ValueError("Length for given task weighting does not match "
                             f"number of tasks: {len(pa.tasks_weighting)} vs. "
                             f"{len(pa.tasks)}.")

        # scheduler specific checks
        def _validate_stage_frac(name, start_frac, warmup_frac):
            if start_frac < 0 or start_frac >= 1:
                raise ValueError(
                    f"{name}: start fraction must be within [0, 1)."
                )
            if warmup_frac < 0 or warmup_frac > 1:
                raise ValueError(
                    f"{name}: warmup fraction must be within [0, 1]."
                )
            if start_frac + warmup_frac >= 1:
                raise ValueError(
                    f"{name}: start fraction plus warmup fraction must stay "
                    "below 1."
                )

        stage_params = (
            (
                "Non-backbone",
                pa.non_backbone_lr_start_frac,
                pa.non_backbone_lr_warmup_frac
            ),
            (
                "RGB backbone",
                pa.rgb_encoder_backbone_lr_start_frac,
                pa.rgb_encoder_backbone_lr_warmup_frac
            ),
            (
                "Depth backbone",
                pa.depth_encoder_backbone_lr_start_frac,
                pa.depth_encoder_backbone_lr_warmup_frac
            ),
            (
                "RGBD backbone",
                pa.rgbd_encoder_backbone_lr_start_frac,
                pa.rgbd_encoder_backbone_lr_warmup_frac
            ),
        )
        for name, start_frac, warmup_frac in stage_params:
            _validate_stage_frac(name, start_frac, warmup_frac)

        # common failures for ScanNet
        if 'scannet' in pa.dataset and pa.validation_scannet_benchmark_mode:
            if pa.scannet_semantic_n_classes not in (40, 549):
                raise ValueError(
                    "`validation-scannet-benchmark-mode` requires "
                    "`scannet-semantic-n-classes` to be 40 or 549."
                )

        # common failures for COCO
        if 'coco' in pa.dataset:
            if 'depth' in pa.input_modalities:
                raise ValueError("COCO dataset does not feature depth data.")
            if 'scene' in pa.tasks:
                raise ValueError("Scene classification is not supported for "
                                 "COCO dataset.")

        if any(d in pa.dataset for d in ('cityscapes', 'hypersim', 'scannet',
                                         'ade20k')):
            # -> Cityscapes: depth values are clipped to 300m, as values above
            # are most likely invalid, they are set them to 0
            # -> Hypersim: depth values are clipped to the limit of
            # png16 (65.535m) during dataset preparation, invalid values are
            # set to 0 (note, the actual amount of clipped pixels is quite
            # small)
            # -> ScanNet: depth is captured with a depth sensor, so the values
            # contain invalid values anyway
            # -> ADE20K: depth values are predicted using a depth estimator,
            # there might be some invalid values

            # to account for this and to ignore these pixels, '--raw-depth'
            # should be forced
            pa.raw_depth = True
            _warn(f"Forced `raw-depth` as `dataset` is '{pa.dataset}'.")

        # handle new/old format for '--subset-train'
        if ':' in pa.subset_train:
            # subset given for concatenated datasets, e.g., 0.5:1.0
            pa.subset_train = tuple(map(float, pa.subset_train.split(':')))
        else:
            pa.subset_train = float(pa.subset_train)

        # evaluation ----------------------------------------------------------
        if pa.validation_full_resolution:
            if not any(d in pa.dataset for d in ('cityscapes', 'hypersim')):
                # height/width in cityscapes and hypersim are multiple 32
                raise ValueError(
                    "Validation with full resolution inputs is only supported"
                    "for 'cityscapes' or 'hypersim'."
                )
        # ensure that validation input size is set (None -> input size)
        if pa.validation_input_width is None:
            pa.validation_input_width = pa.input_width
        if pa.validation_input_height is None:
            pa.validation_input_height = pa.input_height

        if pa.validation_batch_size is None:
            pa.validation_batch_size = 3*pa.batch_size
            _warn(f"`validation-batch-size` not given, using default: "
                  f"{pa.validation_batch_size}.")

        # handle some common misconfigurations
        # when checking the dataset name
        # and
        # additionally datasets can make use of synthetic depth images
        # generated by an depth estimator.
        # E.g. nyuv2^depthanything_v2__indoor_large would use nyuv2 with depth
        # images predicted by depthanything_v2__indoor_large.
        # to still trigger the warning, we split the string by '^' and
        # just take the dataset name without the depth estimator.
        if 'valid' == pa.validation_split and \
           pa.dataset.split('^')[0] in ('nyuv2', 'sunrgbd'):
            pa.validation_split = 'test'
            _warn(f"Dataset '{pa.dataset}' does not have a 'valid' split, "
                  "using 'test' split instead.")

        if pa.validation_skip > pa.checkpointing_skip:
            _warn(f"Setting `checkpointing_skip` to '{pa.validation_skip}' as "
                  f"`validation_skip` is larger '{pa.validation_skip}'.")

        # set default for pa.visualization_output_path and check that it does
        # not already exist
        if pa.visualize_validation and pa.validation_only:
            if pa.visualization_output_path is None:
                weights_dirpath, weights_filename = os.path.split(
                    pa.weights_filepath
                )
                pa.visualization_output_path = os.path.join(
                    weights_dirpath,
                    f'visualization_{os.path.splitext(weights_filename)[0]}'
                )
            if os.path.exists(pa.visualization_output_path):
                raise ValueError(
                    "The path provided by `visualization-output-path` "
                    f"'{pa.visualization_output_path}' already exists. Please "
                    "provide a different path."
                )

        # TODO: can be removed from codebase later
        if all((pa.validation_only,
                'hypersim' in pa.dataset,
                pa.weights_filepath is not None,
                not pa.hypersim_use_old_depth_stats)):

            from datetime import datetime

            t_weights_created = os.path.getctime(pa.weights_filepath)
            t_fix = datetime.strptime('2022.04.28', '%Y.%m.%d').timestamp()
            if t_weights_created < t_fix:
                _warn("Detected Hypersim checkpoint created earlier than "
                      "Apr. 28, 2022. Consider adding "
                      "`hypersim-use-old-depth-stats` argument to ensure "
                      "correct depth stats.")

        # other parameters ----------------------------------------------------
        if pa.debug:
            _warn("`debug` is set, enabling debug outputs and ONNX export "
                  "(use EXPORT_ONNX_MODELS=true python ...)")

        # print args
        if verbose:
            args_str = json.dumps(vars(pa), indent=4, sort_keys=True)
            print(f"Running with args:\n {args_str}")

        # return parsed (and subsequently modified) args
        return pa
