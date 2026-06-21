# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

Notes:
- matching inputs/outputs of the onnx model to pass them to the
  postprocessors is not quite stable (just a fast proof-of-concept
  implementation)
- postprocessing is always done using PyTorch (on GPU if available) and not
  much optimized so far (many operations could be done using ONNX) and, thus,
  should not be part of a timing comparison
"""
import os
import re
import subprocess
import time
import warnings
import math
import types
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from nicr_mt_scene_analysis.data import CollateIgnoredDict
from nicr_mt_scene_analysis.data import move_batch_to_device
from nicr_scene_analysis_datasets.dataset_base import OrientationDict
from nicr_scene_analysis_datasets.dataset_base import SampleIdentifier

from irsaformer.args import ArgParserIRSAFormer
from irsaformer.data import get_datahelper
from irsaformer.data import get_dataset
from irsaformer.data import mt_collate
from irsaformer.model import IRSAFormer
from irsaformer.preprocessing import get_preprocessor
from irsaformer.visualization import visualize
from irsaformer.weights import load_weights


class NCHWToNHWCWrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.decoders = self.model.decoders

    def forward(self, batch, do_postprocessing=False):
        output = self.model(batch, do_postprocessing=do_postprocessing)
        side_output = output[0][1]
        output = output[0][0]
        output = output.permute(0, 2, 3, 1)

        return [[output, side_output]]


def _replace_modules_with_wrappers(module, module_wrappers):
    for name, child in module.named_children():
        child_type = type(child)
        if child_type in module_wrappers:
            wrapper_cls = module_wrappers[child_type]
            setattr(module, name, wrapper_cls(child))
        else:
            _replace_modules_with_wrappers(child, module_wrappers)


class AdaptiveAvgPool2dWrapper(torch.nn.Module):
    def __init__(self, avgpool):
        super().__init__()
        assert isinstance(avgpool, torch.nn.AdaptiveAvgPool2d)
        self.avgpool = avgpool
        self._replaced = False

    def forward(self, x):
        y = self.avgpool(x)

        if not self._replaced:
            stride_hw = [x.shape[i] // y.shape[i] for i in (-2, -1)]
            kernel_size = [
                x.shape[i] - (y.shape[i] - 1) * stride_hw[k]
                for k, i in enumerate((-2, -1))
            ]
            self.avgpool = torch.nn.AvgPool2d(
                kernel_size, stride=stride_hw
            )
            self._replaced = True

        return y


def _torch_dtype_from_trt(trt_dtype):
    import tensorrt as trt  # local import to keep optional dependency

    np_dtype = trt.nptype(trt_dtype)
    return torch.from_numpy(np.empty((), dtype=np_dtype)).dtype


def _patch_rope_for_onnx(model: torch.nn.Module):
    # for ONNX export only, we monkey-patch the RoPE module used by the DINOv3
    # backbone. the replacement recomputes the same axial rotary position
    # embedding from the standard formula (rotary position embeddings, Su et
    # al. 2021. DINOv3 axial variant), reading the module's own dim / periods /
    # rotate_half. the only difference from the original is using
    # `torch.cat([angles, angles], dim=-1)` instead of `angles.tile(2)`. this
    # is numerically identical but avoids the dynamic Tile/If subgraph that
    # TensorRT 10.x rejects (observed on Jetson Orin: `/encoder/If_output_0`
    # with shapes [2] vs [1], causing trtexec to fail). the original method is
    # restored right after export so training/inference behavior is unchanged.
    from timm.layers.pos_embed_sincos import RotaryEmbeddingDinoV3

    patched = []
    for module in model.modules():
        if not isinstance(module, RotaryEmbeddingDinoV3):
            continue
        def _get_pos_embed_from_coords_concat(self, coords: torch.Tensor):
            dim = self.dim // 4
            device = self.periods.device
            dtype = self.periods.dtype
            assert self.periods.numel() == dim
            coords = coords[:, :, None].to(device=device, dtype=dtype)
            angles = 2 * math.pi * coords / self.periods[None, None, :]
            angles = angles.flatten(1)
            if self.rotate_half:
                angles = torch.cat([angles, angles], dim=-1)
            else:
                angles = torch.stack([angles, angles], dim=-1).reshape(
                    angles.shape[0], -1
                )
            sin = torch.sin(angles)
            cos = torch.cos(angles)
            return sin, cos

        original = module._get_pos_embed_from_coords
        module._get_pos_embed_from_coords = types.MethodType(
            _get_pos_embed_from_coords_concat, module
        )
        patched.append((module, original))
    return patched


def _restore_rope_patches(patched):
    for module, original in patched:
        module._get_pos_embed_from_coords = original


class TRTModel:
    """Minimal TensorRT engine runner relying solely on TensorRT Python API."""

    def __init__(self, engine_path: str, profile_idx: Optional[int] = None):
        import tensorrt as trt  # local import

        self._trt = trt
        self._engine_path = engine_path
        self._profile_idx = profile_idx

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize TensorRT engine '{engine_path}'."
            )
        self.context = self.engine.create_execution_context()

        self._input_bindings = []
        self._output_bindings = []
        if (
            hasattr(self.engine, 'num_io_tensors')
            and self.engine.num_io_tensors
        ):
            for tensor_idx in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(tensor_idx)
                dtype = self.engine.get_tensor_dtype(name)
                index = self.engine.get_tensor_index(name)
                mode = self.engine.get_tensor_mode(name)
                binding = {'index': index, 'name': name, 'dtype': dtype}
                if mode == trt.TensorIOMode.INPUT:
                    self._input_bindings.append(binding)
                else:
                    self._output_bindings.append(binding)
        else:
            for index in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(index)
                dtype = self.engine.get_binding_dtype(index)
                binding = {'index': index, 'name': name, 'dtype': dtype}
                if self.engine.binding_is_input(index):
                    self._input_bindings.append(binding)
                else:
                    self._output_bindings.append(binding)

        self._num_bindings = self.engine.num_bindings

    @property
    def input_names(self) -> List[str]:
        return [binding['name'] for binding in self._input_bindings]

    @property
    def output_names(self) -> List[str]:
        return [binding['name'] for binding in self._output_bindings]

    def __call__(
        self,
        x_input: Union[
            torch.Tensor,
            List[torch.Tensor],
            Dict[str, torch.Tensor]
        ]
    ):
        import tensorrt as trt  # noqa: F401 local import

        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT execution requires CUDA.")

        if isinstance(x_input, torch.Tensor):
            input_dict: Dict[str, torch.Tensor] = {
                self._input_bindings[0]['name']: x_input
            }
        elif isinstance(x_input, list):
            input_dict = {
                binding['name']: tensor
                for binding, tensor in zip(self._input_bindings, x_input)
            }
        elif isinstance(x_input, dict):
            input_dict = x_input
        else:
            raise TypeError(f"Unsupported input type '{type(x_input)}'.")

        # prepare bindings and ensure tensors are on GPU
        bindings = [0] * self._num_bindings
        input_gpu_tensors: Dict[str, torch.Tensor] = {}
        for binding in self._input_bindings:
            name = binding['name']
            tensor = input_dict[name]
            if not torch.is_tensor(tensor):
                raise TypeError(f"Expected tensor for binding '{name}'.")
            tensor_gpu = tensor.to('cuda', non_blocking=True)
            if not tensor_gpu.is_contiguous():
                tensor_gpu = tensor_gpu.contiguous()

            # set binding shape
            shape = tuple(tensor_gpu.shape)
            if hasattr(self.context, 'set_input_shape'):
                self.context.set_input_shape(name, shape)
            else:
                self.context.set_binding_shape(binding['index'], shape)

            input_gpu_tensors[name] = tensor_gpu
            bindings[binding['index']] = int(tensor_gpu.data_ptr())

        # prepare output tensors (allocated on GPU)
        output_gpu_tensors: Dict[str, torch.Tensor] = {}
        for binding in self._output_bindings:
            name = binding['name']
            if hasattr(self.context, 'get_tensor_shape'):
                shape = tuple(self.context.get_tensor_shape(name))
            else:
                shape = tuple(self.context.get_binding_shape(binding['index']))
            torch_dtype = _torch_dtype_from_trt(binding['dtype'])
            output_gpu = torch.empty(shape, dtype=torch_dtype, device='cuda')
            output_gpu_tensors[name] = output_gpu
            bindings[binding['index']] = int(output_gpu.data_ptr())

        stream = torch.cuda.current_stream()
        if hasattr(self.context, 'execute_async_v3'):
            self.context.execute_async_v3(
                stream_handle=stream.cuda_stream,
                bindings=bindings
            )
        else:
            self.context.execute_async_v2(bindings, stream.cuda_stream)
        stream.synchronize()

        # collect outputs on CPU
        outputs_cpu = {
            name: tensor.cpu()
            for name, tensor in output_gpu_tensors.items()
        }

        if isinstance(x_input, list):
            return [
                outputs_cpu[binding['name']]
                for binding in self._output_bindings
            ]
        if isinstance(x_input, torch.Tensor):
            sole_output = outputs_cpu[self._output_bindings[0]['name']]
            return sole_output
        return outputs_cpu


def _parse_args():
    parser = ArgParserIRSAFormer()
    group = parser.add_argument_group('Inference Timing')
    # add arguments
    # general
    group.add_argument(
        '--model-onnx-filepath',
        type=str,
        default=None,
        help="Path to ONNX model file when `model` is 'onnx'."
    )

    # input
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
        '--inference-batch-size',
        type=int,
        default=1,
        help="Batch size to use for inference."
    )

    # runs
    group.add_argument(
        '--n-runs',
        type=int,
        default=100,
        help="Number of runs the inference time will be measured."
    )
    group.add_argument(
        '--n-runs-warmup',
        type=int,
        default=10,
        help="Number of forward passes through the model before the inference "
             "time measurements starts. This is necessary as the first runs "
             "are slower."
    )

    # timings
    group.add_argument(
        '--no-time-pytorch',
        action='store_true',
        default=False,
        help="Do not measure inference time using PyTorch."
    )
    group.add_argument(
        '--no-time-tensorrt',
        action='store_true',
        default=False,
        help="Do not measure inference time using TensorRT."
    )
    group.add_argument(
        '--with-postprocessing',
        action='store_true',
        default=False,
        help="Include postprocessing in timing."
    )

    # export
    group.add_argument(
        '--export-outputs',
        action='store_true',
        default=False,
        help="Whether to export the outputs of the model."
    )

    # tensorrt
    group.add_argument(
        '--trt-floatx',
        type=int,
        choices=(8, 16, 32),
        default=32,
        help="Whether to measure with float8, float16, or float32."
    )
    group.add_argument(
        '--trt-onnx-opset-version',
        type=int,
        default=20,
        help="Opset version to use for export."
    )
    group.add_argument(
        '--trt-do-not-force-rebuild',
        dest='trt_force_rebuild',
        action='store_false',
        default=True,
        help="Reuse existing TensorRT engine."
    )
    group.add_argument(
        '--trt-enable-dynamic-batch-axis',
        action='store_true',
        default=False,
        help="Enable dynamic axes."
    )
    group.add_argument(
        '--trt-onnx-export-only',
        action='store_true',
        default=False,
        help="Export ONNX model for TensorRT only. To measure inference time, "
             "use '--model-onnx-filepath ./model_tensorrt.onnx' in a second "
             "run."
    )
    group.add_argument(
        '--trt-use-python',
        action='store_true',
        default=False,
        help="Use python bindings instead of trtexec to use the engine, which "
             "might be slightly slower but is required to do inference with "
             "real samples."
    )
    args = parser.parse_args()
    return args


def create_batch(data, start_idx, batch_size):
    batch = [data[i % len(data)]
             for i in range(start_idx, start_idx + batch_size)]

    return mt_collate(batch, type_blacklist=(np.ndarray,
                                             CollateIgnoredDict,
                                             OrientationDict,
                                             SampleIdentifier))


def sample_batches(data, batch_size, n_batches):
    for i in range(n_batches):
        yield create_batch(data, i*batch_size, batch_size)


def create_engine(onnx_filepath,
                  engine_filepath,
                  floatx=16,
                  batch_size=1,
                  inputs=None,
                  input_names=None,
                  force_rebuild=True,
                  dynamic_shapes=False):

    if os.path.exists(engine_filepath) and not force_rebuild:
        # engine already exists
        return

    # note, we use trtexec to convert ONNX files to TensorRT engines
    print("Building engine using trtexec ...")
    if floatx == 32:
        print("\t... this may take a while")
    else:
        print("\t... this may take -> AGES <-")

    cmd = (
        f'trtexec'
        f' --onnx={onnx_filepath}'
        f' --saveEngine={engine_filepath}'
    )
    if floatx == 16:
        cmd += ' --fp16'
    elif floatx == 8:
        cmd += ' --fp8'

    if dynamic_shapes:
        shape_str = ''
        input_names_for_shape_str = input_names

        # if an RGB-D encoder is used, we still need to get the shapes for both
        # rgb and depth separately
        if (
            len(input_names_for_shape_str) == 1
            and input_names_for_shape_str[0] == 'rgbd'
        ):
            input_names_for_shape_str = ['rgb', 'depth']

        for name in input_names_for_shape_str:
            shape = inputs[0][name].shape
            if len(shape) == 4:
                _, c, h, w = shape
            else:
                c, h, w = shape
            shape_str += f'{name}:{batch_size}x{c}x{h}x{w},'
        shape_format = (
            f' --minShapes={shape_str[:-1]}'
            f' --optShapes={shape_str[:-1]}'
            f' --maxShapes={shape_str[:-1]}'
        )
        cmd += shape_format

    # Execute command
    print('Building engine ...')
    out = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    if out.returncode != 0:
        print(out.stdout.decode('utf-8'))
    assert out.returncode == 0


def time_inference_tensorrt_trtexec(onnx_filepath,
                                    inputs,
                                    input_names,
                                    floatx=16,
                                    batch_size=1,
                                    n_runs=100,
                                    n_runs_warmup=10,
                                    force_engine_rebuild=True,
                                    postprocessors=None,
                                    postprocessors_device='cpu',
                                    store_data=False,
                                    dynamic_shapes=False):
    # create engine
    trt_filepath = os.path.splitext(onnx_filepath)[0] + '.trt'
    create_engine(onnx_filepath, trt_filepath,
                  floatx=floatx, batch_size=batch_size,
                  inputs=inputs, input_names=input_names,
                  force_rebuild=force_engine_rebuild,
                  dynamic_shapes=dynamic_shapes)

    # build execution command for TensorRT
    N_WARMUP_TIME = 10000    # = 10 seconds (we do not use n_runs_warmup here)
    cmd = (
        'trtexec'
        f' --loadEngine={trt_filepath}'
        ' --noDataTransfers'
        ' --useCudaGraph'
        ' --useSpinWait'
        ' --separateProfileRun'
        f' --warmUp={N_WARMUP_TIME}'
        f' --iterations={n_runs}'
    )
    if floatx == 16:
        cmd += ' --fp16'
    elif floatx == 8:
        cmd += ' --fp8'
    print('Running inference ...')
    print(cmd)

    # execute command and parse output
    outs = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
    output = outs.stdout.decode('utf-8')
    # get qps from output: Throughput: ([0-9.]+) qps
    res = re.findall(r'Throughput: ([0-9.]+) qps', output)
    assert len(res) == 1

    # return outputs that match the output of the remaining functions, i.e.,
    # convert qps to a single timing, and return an empty list for inputs &
    # outputs (we do not have them)
    return np.array([1/float(res[0])]), []


def time_inference_tensorrt_python(onnx_filepath,
                                   inputs,
                                   input_names,
                                   floatx=16,
                                   batch_size=1,
                                   n_runs=100,
                                   n_runs_warmup=10,
                                   force_engine_rebuild=True,
                                   postprocessors=None,
                                   postprocessors_device='cpu',
                                   store_data=False):
    # create engine
    trt_filepath = os.path.splitext(onnx_filepath)[0] + '.trt'
    create_engine(onnx_filepath, trt_filepath,
                  floatx=floatx, batch_size=batch_size,
                  inputs=inputs, input_names=input_names,
                  force_rebuild=force_engine_rebuild)

    # load engine
    trt_model = TRTModel(trt_filepath)

    # time inference
    timings = []
    outs = []
    for i, input_ in enumerate(sample_batches(inputs,
                                              batch_size,
                                              n_runs+n_runs_warmup)):
        start_time = time.time()

        # get model output
        output = trt_model(input_)

        if postprocessors is None:
            out_trt = output
        else:
            out_trt = {}
            for name, post in postprocessors.items():
                # create input
                # bit hacky, this works as the keys are ordered
                in_post = [
                    output.cpu()
                    for k, output in output.items()
                    if name in k
                ]

                if 'cpu' != postprocessors_device:
                    # copy back to GPU (not smart)
                    in_post = [t.to(postprocessors_device) for t in in_post]

                    # we also need some inputs on gpu for the postprocessing
                    # keep rgb/depth inputs, including fullres keys
                    input_post = {
                        k: v.to(postprocessors_device)
                        for k, v in input_.items()
                        if (
                            ('rgb' in k or 'depth' in k)
                            and torch.is_tensor(v)
                        )
                    }
                else:
                    # simply we use the whole input batch for postprocessing
                    input_post = input_

                in_post_side = None
                if 1 == len(in_post):
                    # single input to postprocessor
                    in_post = in_post[0]
                else:
                    # multiple inputs to postprocessor (instance / panoptic)
                    in_post = tuple(in_post)

                if 'panoptic_helper' == name:
                    # this is not quite smart but works for now
                    # first element is semantic, the remaining instance
                    in_post = (in_post[0], in_post[1:])
                    in_post_side = (None, None)

                out_trt.update(
                    post.postprocess(
                        data=(in_post, in_post_side),
                        batch=input_post,
                        is_training=False
                    )
                )

            if postprocessors_device != 'cpu':
                out_trt = move_batch_to_device(out_trt, 'cpu')

        if i >= n_runs_warmup:
            timings.append(time.time() - start_time)

        if store_data:
            outs.append((input_, out_trt))

    return np.array(timings), outs


def time_inference_pytorch(model,
                           inputs,
                           device,
                           n_runs=100,
                           n_runs_warmup=5,
                           batch_size=1,
                           with_postprocessing=False,
                           store_data=False):
    timings = []
    with torch.no_grad():
        outs = []
        for i, input_ in enumerate(sample_batches(inputs,
                                                  batch_size,
                                                  n_runs+n_runs_warmup)):
            # use PyTorch to time events
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            # copy to gpu
            # keep rgb/depth inputs, including fullres keys
            inputs_gpu = {
                k: v.to(device)
                for k, v in input_.items()
                if (
                    ('rgb' in k or 'depth' in k)
                    and torch.is_tensor(v)
                )
            }

            # model forward pass
            out_pytorch = model(inputs_gpu,
                                do_postprocessing=with_postprocessing)

            # copy back to cpu
            if not with_postprocessing:
                out_pytorch_cpu = []
                # output is tuple (outputs, side_output)
                for outputs, _ in out_pytorch:    # ignore side outputs
                    for output in outputs:
                        if isinstance(output, tuple):
                            # panoptic helper is again a tuple
                            out_pytorch_cpu.extend([o.cpu() for o in output])
                        else:
                            out_pytorch_cpu.append(output.cpu())
            else:
                # output is a dict
                out_pytorch_cpu = move_batch_to_device(out_pytorch, 'cpu')

            end.record()
            torch.cuda.synchronize()

            if i >= n_runs_warmup:
                timings.append(start.elapsed_time(end) / 1e3)

            if store_data:
                outs.append((input_, out_pytorch_cpu))

    return np.array(timings), outs


def get_fps_from_timings(timings, batch_size):
    return np.mean(1 / timings) * batch_size


def main(args):
    # prepare inputs -----------------------------------------------------------
    n_samples = 49
    args.batch_size = 1    # force bs 1 for collecting samples
    args.validation_batch_size = 1    # force bs 1 for collecting samples
    args.n_workers = 0    # no threads in torch dataloaders, use main thread

    data_helper = get_datahelper(args)

    inputs = []
    if args.dataset_path is not None:
        if not args.trt_use_python:
            raise ValueError("Please set '--trt-use-python' when providing "
                             "'--dataset-path' for TensorRT timing.")

        # simply use first dataset (they all share the same properties)
        dataset = data_helper.datasets_valid[0]

        # get preprocessed samples of the given dataset
        data_helper.set_valid_preprocessor(
            get_preprocessor(
                args,
                dataset=dataset,
                phase='test',
                multiscale_downscales=None
            )
        )

        # disable memory pinning as it currently (pytorch 2.3.1) handles types
        # derived from tuple (e.g., our SampleIdentifier, see mt_collate usage)
        # in a wrong way, see:
        # https://github.com/pytorch/pytorch/blob/v2.3.1/torch/utils/data/_utils/pin_memory.py#L79
        data_helper.valid_dataloaders[0].pin_memory = False

        for sample in data_helper.valid_dataloaders[0]:
            inputs.append(
                {k: v[0] for k, v in sample.items()}    # remove batch axis
            )

            if (n_samples) == len(inputs):
                # enough samples collected
                break
    else:
        if args.with_postprocessing:
            # postprocessing random inputs does not really make sense
            # moreover, we need more fullres keys
            raise ValueError("Please set `--dataset-path` to enable "
                             "inference with meaningful inputs.")

        # the dataset's config is used later on for model building
        dataset = get_dataset(args, split=args.validation_split)

        # we do not have access to the data of dataset, simply collect random
        # inputs
        rgb_images = []
        depth_images = []
        for _ in range(n_samples):
            img_rgb = np.random.randint(
                low=0,
                high=255,
                size=(args.input_height, args.input_width, 3),
                dtype='uint8'
            )
            img_depth = np.random.randint(
                low=0,
                high=40000,
                size=(args.input_height, args.input_width),
                dtype='uint16'
            )
            # preprocess
            img_rgb = (img_rgb / 255).astype('float32').transpose(2, 0, 1)
            img_depth = (img_depth.astype('float32') / 20000)[None]
            img_rgb = np.ascontiguousarray(img_rgb)
            img_depth = np.ascontiguousarray(img_depth)
            rgb_images.append(torch.tensor(img_rgb))
            depth_images.append(torch.tensor(img_depth))

        # convert to input format (see BatchType)
        if 2 == len(args.input_modalities):
            inputs = [{'rgb': rgb_images[i], 'depth': depth_images[i]}
                      for i in range(len(rgb_images))]
        elif 'rgb' in args.input_modalities:
            inputs = [{'rgb': rgb_images[i]}
                      for i in range(len(rgb_images))]
        elif 'depth' in args.input_modalities:
            inputs = [{'depth': depth_images[i]}
                      for i in range(len(rgb_images))]
        elif 'rgbd' in args.input_modalities:
            inputs = [{'rgb': rgb_images[i], 'depth': depth_images[i]}
                      for i in range(len(rgb_images))]

    # create model ------------------------------------------------------------
    if args.model_onnx_filepath is not None:
        warnings.warn(
            "PyTorch inference timing disabled since onnx model is given."
        )
        args.no_time_pytorch = True
    if args.trt_onnx_export_only:
        args.no_time_pytorch = True

    # create model
    args.no_pretrained_backbone = True
    model = IRSAFormer(args=args,
                       dataset_configs={args.dataset: dataset.config})

    # load weights
    if args.weights_filepath is not None:
        checkpoint = torch.load(args.weights_filepath,
                                map_location=lambda storage, loc: storage)
        print(f"Loading checkpoint: '{args.weights_filepath}'.")
        if 'epoch' in checkpoint:
            print(f"-> Epoch: {checkpoint['epoch']}")
        delta_meta = checkpoint.get('_delta')
        load_weights(args, model, checkpoint['state_dict'],
                     delta_meta=delta_meta)
    else:
        # Make all parameters (weights and biases) completely random
        # because else TensorRT can fail to build the engine.
        for _, param in model.named_parameters():
            if param.requires_grad:
                param.data = torch.randn(param.size())

    device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
    model.eval()

    # define dummy input for export
    dummy_input = (create_batch(inputs, start_idx=0, batch_size=1),
                   {'do_postprocessing': False})

    # When using real data there will be many more keys in the input dict
    # which are not required for the model. For onnx export we filter them.
    if args.dataset_path is not None:
        keys_to_keep = ['rgb', 'depth']
        dummy_input_dict = {
            k: v for k, v in dummy_input[0].items() if k in keys_to_keep
        }
        dummy_input = (dummy_input_dict, dummy_input[1])

    # define names for input and output graph nodes
    # note, meaningful names are required to match postprocessors and
    # to set up dynamic_axes dict correctly
    input_names = [k for k in dummy_input[0].keys()]

    # time inference using PyTorch --------------------------------------------
    if not args.no_time_pytorch:
        # move model to gpu
        model.to(device)

        timings_pytorch, ios_pytorch = time_inference_pytorch(
            model,
            inputs,
            device,
            n_runs=args.n_runs,
            n_runs_warmup=args.n_runs_warmup,
            batch_size=args.inference_batch_size,
            with_postprocessing=args.with_postprocessing,
            store_data=args.export_outputs
        )
        mean_fps = get_fps_from_timings(
            timings_pytorch,
            batch_size=args.inference_batch_size
        )
        print(f'fps pytorch: {mean_fps:0.4f}')

        # move model back to cpu (required for further steps)
        model.to('cpu')

    # time inference using TensorRT -------------------------------------------
    if not args.no_time_tensorrt:
        if args.model_onnx_filepath is None:
            # we have to export the model to onnx

            # determine output structure in order to derive names
            outputs = model(dummy_input[0], **dummy_input[1])
            assert len(outputs) == len(model.decoders)
            # encode output structure to output names (note, this is parsed
            # later to assign the outputs to the postprocessors if the model
            # is loaded from pure onnx)
            output_names = []
            for (outs, _), decoder_name in zip(outputs, model.decoders):
                if not isinstance(outs, tuple):
                    # semantic (single tensor)
                    outs = tuple(outs)

                if 'panoptic_helper' == decoder_name:
                    # this is not quite smart but works for now
                    # join semantic (single tensor) and instance outputs
                    outs = (outs[0], ) + outs[1]

                for j, _ in enumerate(outs):
                    # format output name
                    output_names.append(f'{decoder_name}_{j}')

            onnx_filepath = './model_tensorrt.onnx'

            # determine dynamic axes if enabled
            dynamic_axes = None
            if args.trt_enable_dynamic_batch_axis:
                dynamic_axes = {}
                for input_name in input_names:
                    dynamic_axes[input_name] = {0: 'batch_size'}

                for output_name in output_names:
                    dynamic_axes[output_name] = {0: 'batch_size'}

            _replace_modules_with_wrappers(
                model, {torch.nn.AdaptiveAvgPool2d: AdaptiveAvgPool2dWrapper}
            )
            with torch.no_grad():
                _ = model(dummy_input[0], **dummy_input[1])
            patched = _patch_rope_for_onnx(model)
            try:
                torch.onnx.export(
                    model,
                    (dummy_input[0],),
                    onnx_filepath,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    dynamo=False,
                    opset_version=args.trt_onnx_opset_version,
                    do_constant_folding=True,
                    kwargs=dummy_input[1]
                )
            finally:
                _restore_rope_patches(patched)

            print(f"ONNX file (opset {args.trt_onnx_opset_version}) written "
                  f"to '{onnx_filepath}'.")

            if args.trt_onnx_export_only:
                # stop here
                exit(0)
        else:
            onnx_filepath = args.model_onnx_filepath

        # extract postprocessors
        if args.with_postprocessing:
            postprocessors = {
                k: v.postprocessing for k, v in model.decoders.items()
            }
        else:
            postprocessors = None

        if args.trt_use_python:
            timings_tensorrt, ios_tensorrt = time_inference_tensorrt_python(
                onnx_filepath,
                inputs,
                input_names,
                floatx=args.trt_floatx,
                batch_size=args.inference_batch_size,
                n_runs=args.n_runs,
                n_runs_warmup=args.n_runs_warmup,
                force_engine_rebuild=args.trt_force_rebuild,
                postprocessors=postprocessors,
                postprocessors_device=device,
                store_data=args.export_outputs
            )
            label = 'python'
        else:
            timings_tensorrt, ios_tensorrt = time_inference_tensorrt_trtexec(
                onnx_filepath,
                inputs,
                input_names,
                floatx=args.trt_floatx,
                batch_size=args.inference_batch_size,
                n_runs=args.n_runs,
                n_runs_warmup=args.n_runs_warmup,
                force_engine_rebuild=args.trt_force_rebuild,
                postprocessors=postprocessors,
                postprocessors_device=device,
                store_data=args.export_outputs,
                dynamic_shapes=args.trt_enable_dynamic_batch_axis,
            )
            label = 'trtexec'
        mean_fps = get_fps_from_timings(
            timings_tensorrt,
            batch_size=args.inference_batch_size
        )
        print(f'fps tensorrt ({label}): {mean_fps:0.4f}')

    if args.export_outputs:
        assert args.with_postprocessing, "Re-run with `--with-postprocessing`"

        results_path = os.path.join(os.path.dirname(__file__),
                                    'inference_results',
                                    args.dataset)

        os.makedirs(results_path, exist_ok=True)

        if 'ios_pytorch' in locals():
            for inp, out in ios_pytorch:
                visualize(
                    output_path=os.path.join(results_path, 'pytorch'),
                    batch=inp,
                    predictions=out,
                    dataset_config=dataset.config
                )

        if 'ios_tensorrt' in locals():
            for inp, out in ios_tensorrt:
                visualize(
                    output_path=os.path.join(results_path,
                                             f'tensorrt_{args.trt_floatx}'),
                    batch=inp,
                    predictions=out,
                    dataset_config=dataset.config
                )


if __name__ == '__main__':
    # parse args
    args = _parse_args()

    print('PyTorch version:', torch.__version__)

    if not args.no_time_tensorrt:
        # to enable execution without TensorRT, we import relevant modules here
        import tensorrt as trt

        print('TensorRT version:', trt.__version__)

    main(args)
