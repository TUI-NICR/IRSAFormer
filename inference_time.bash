#!/bin/bash
# Reproduce the TensorRT throughput reported in the paper. Run this on the target
# device (e.g. NVIDIA Jetson AGX Orin) with the IRSAFormer dependencies and
# TensorRT installed. Throughput is backbone- and modality-bound, so we time the
# panoptic checkpoints as representatives of the six DINOv3 configurations. The
# reported FPS use FP16. Set DATASET_PATH to time on real samples instead of
# random input. Results are written to results.csv.
set -o xtrace

MODELS=(
    panoptic_small_plus_rgb
    panoptic_small_plus_rgbd
    panoptic_base_rgb
    panoptic_base_rgbd
    panoptic_large_rgb
    panoptic_large_rgbd
)

ARGS_COMMON=(--dataset nyuv2 --inference-input-height 480 --inference-input-width 640)
if [[ -n "${DATASET_PATH}" ]]; then
    ARGS_COMMON+=(--dataset-path "${DATASET_PATH}")
fi

ARGS_EXPORT=(--no-time-pytorch --trt-onnx-export-only)
ARGS_TIME_TRT32=(--no-time-pytorch --model-onnx-filepath ./model_tensorrt.onnx --n-runs-warmup 20 --n-runs 80)
ARGS_TIME_TRT16=(--no-time-pytorch --model-onnx-filepath ./model_tensorrt.onnx --n-runs-warmup 20 --n-runs 80 --trt-floatx 16)

SED_TRT='s/.*fps tensorrt ([^)]*): \([0-9.]*\).*/\1/p'

RESULTS_FILE='./results.csv'
echo "Model,TensorRT FP32 FPS,TensorRT FP16 FPS" > "${RESULTS_FILE}"

for name in "${MODELS[@]}"; do
    ckpt_dir="./trained_models/irsaformer_${name}"
    read -r -a model_args < "${ckpt_dir}/model_args.txt"
    weights=(--weights-filepath "${ckpt_dir}/irsaformer_${name}.pth")
    common=("${ARGS_COMMON[@]}" "${model_args[@]}" "${weights[@]}")

    # export the ONNX model once, then reuse it for the FP32 and FP16 timings
    rm -f ./model_tensorrt.onnx
    python3 inference_time_whole_model.py "${ARGS_EXPORT[@]}" "${common[@]}"

    fps_fp32=$(python3 inference_time_whole_model.py "${ARGS_TIME_TRT32[@]}" "${common[@]}" | sed -n "${SED_TRT}")
    [[ -z "${fps_fp32}" ]] && fps_fp32="NA"

    fps_fp16=$(python3 inference_time_whole_model.py "${ARGS_TIME_TRT16[@]}" "${common[@]}" | sed -n "${SED_TRT}")
    [[ -z "${fps_fp16}" ]] && fps_fp16="NA"

    echo "${name},${fps_fp32},${fps_fp16}" >> "${RESULTS_FILE}"
done
