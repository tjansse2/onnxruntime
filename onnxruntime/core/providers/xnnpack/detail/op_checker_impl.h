// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "node_support_checker.h"


namespace onnxruntime {
class NodeUnit;
class GraphViewer;

namespace xnnpack {
//from xnnpack subgraph.h
enum xnn_compute_type {
  xnn_compute_type_invalid = 0,
  xnn_compute_type_fp32,
  xnn_compute_type_fp16,
  xnn_compute_type_qc8,
  xnn_compute_type_qs8,
  xnn_compute_type_qu8,
  /*
  xnn_compute_type_fp32_to_fp16,
  xnn_compute_type_fp32_to_qs8,
  xnn_compute_type_fp32_to_qu8,
  xnn_compute_type_fp16_to_fp32,
  xnn_compute_type_qs8_to_fp32,
  xnn_compute_type_qu8_to_fp32,*/
};
// check to see if an ONNX NCHW Conv node is supported by this implementation. the first input and output will be
// converted to NHWC by ORT.
bool IsConvOnnxNodeSupported(const onnxruntime::NodeUnit& nchw_nodeunit, const onnxruntime::GraphViewer& graph);
// check to see if an ONNX NCHW Conv node is supported by this implementation. the first input and output will be
// converted to NHWC by ORT.
bool IsMaxPoolOnnxNodeSupported(const onnxruntime::NodeUnit& nchw_nodeunit, const onnxruntime::GraphViewer& graph);


}  // namespace xnnpack
}  // namespace onnxruntime
