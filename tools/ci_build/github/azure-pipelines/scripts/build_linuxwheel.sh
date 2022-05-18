#!/ bin/bash
set -ex

onnx_dir=$1
source_dir=$2
bin_dir=$3
python_dir=$4
#used for special EP
build_py_parameters=$5

mkdir -p "$onnx_dir"
docker run --rm \
    --volume /data/onnx:/data/onnx:ro \
    --volume "$source_dir":/onnxruntime_src \
    --volume "$bin_dir":/build \
    --volume /data/models:/build/models:ro \
    --volume "$onnx_dir":/home/onnxruntimedev/.onnx \
    -e NIGHTLY_BUILD \
    -e BUILD_BUILDNUMBER \
    onnxruntimecpubuild \
    "$python_dir"/bin/python3 /onnxruntime_src/tools/ci_build/build.py \
        --build_dir /build --cmake_generator Ninja \
        --config Release --update --build \
        --skip_submodule_sync \
        --parallel \
        --enable_lto \
        --build_wheel \
        --enable_onnx_tests \
        "$build_py_parameters"
