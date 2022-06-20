#include <unordered_map>

#include "op_checker_impl.h"
#include "core/providers/shared/node_unit/node_unit.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/common/safeint.h"
#include "core/providers/xnnpack/detail/utils.h"
#include "core/framework/tensorprotoutils.h"

// each operator provides a helper to check if supported
#include "core/providers/xnnpack/nn/conv.h"
#include "core/providers/xnnpack/nn/max_pool.h"

namespace onnxruntime {
namespace xnnpack {

bool IsQuantizedConv(QuantizedOpType quant_op_type) {
  return (quant_op_type == QuantizedOpType::QLinearConv) ||
         (quant_op_type == QuantizedOpType::QDQConv);
}

bool IsQuantizedMaxPool(QuantizedOpType quant_op_type) {
  return (quant_op_type == QuantizedOpType::QLinearMaxPool) ||
         (quant_op_type == QuantizedOpType::QDQMaxPool);
}

bool GetType(const NodeArg& node_arg, int32_t& type) {
  type = ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
  const auto* type_proto = node_arg.TypeAsProto();
  if (!type_proto || !type_proto->has_tensor_type() || !type_proto->tensor_type().has_elem_type()) {
    LOGS_DEFAULT(WARNING) << "NodeArg [" << node_arg.Name() << "] has no input type";
    return false;
  }

  type = type_proto->tensor_type().elem_type();
  return true;
}

bool GetShape(const NodeArg& node_arg, Shape& shape) {
  shape.clear();
  const auto* shape_proto = node_arg.Shape();

  if (!shape_proto) {
    LOGS_DEFAULT(WARNING) << "NodeArg [" << node_arg.Name() << "] has no shape info";
    return false;
  }

  // uses 0 for dynamic dimension, which is the default value for dim.dim_value()
  for (const auto& dim : shape_proto->dim())
    shape.push_back(SafeInt<uint32_t>(dim.dim_value()));

  return true;
}

static const onnx::TensorProto* GetQuantizationScale(const InitializedTensorSet& initializers,
                                         const NodeUnitIODef& io_def) {
  onnx::TensorProto tensor_proto_ret;
  const auto scale_name = io_def.quant_param->scale.Name();
  auto it = initializers.find(scale_name);
  if (it == initializers.cend()) {
    return nullptr;
  }
  return it->second;
}

static const onnx::TensorProto* GetQuantizationZeroPoint(const InitializedTensorSet& initializers,
                                             const NodeUnitIODef& io_def) {
  if (!io_def.quant_param->zero_point)
    return nullptr;

  const auto& zero_point_name = io_def.quant_param->zero_point->Name();
  if (!Contains(initializers, zero_point_name)) {
    return nullptr;
  }

  return initializers.at(zero_point_name);
}

//Xnnpack predefined a few dtypes for quantized tensor, hence we can easily check if xnnpack support it 
xnn_datatype GetDtypeInXnnpack(const onnxruntime::NodeUnit& node_unit, int32_t io_index,
    bool is_output, const onnxruntime::GraphViewer& graph_viewer) {
  //we are not check the legality of io_index here
  const NodeUnitIODef& iodef = is_output ? node_unit.Outputs()[io_index] : node_unit.Inputs()[io_index];
  xnn_datatype datatype = xnn_datatype_invalid;
  int32_t input_type = 0;
  if (!GetType(iodef.node_arg, input_type)) {
    return datatype;
  }
  const InitializedTensorSet& initializers = graph_viewer.GetAllInitializedTensors();
  auto* zero_tensor = GetQuantizationZeroPoint(initializers, iodef);
  auto* scale_tensor = GetQuantizationScale(initializers, iodef);
  int64_t scales_dim = !scale_tensor ? 0: (scale_tensor->dims().empty() ? 1 : scale_tensor->dims()[0]);
  int64_t zero_dim = !zero_tensor ? 0: (zero_tensor->dims().empty() ? 1 : zero_tensor->dims()[0]);
  const auto quantization_params = iodef.quant_param.value();
  Shape tensor_shape;
  if (!GetShape(iodef.node_arg, tensor_shape))
    return datatype;
  std::vector<uint8_t> unpacked_tensor;
  //we have process float-type in the begining
  switch (input_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      if (quantization_params.zero_point == nullptr) {
        LOGS_DEFAULT(VERBOSE) << "missing zero point quantization parameters for "
            "UINT8 tensor";
        break;
      }
      if (scales_dim != 1 || zero_dim != 1) {
        LOGS_DEFAULT(VERBOSE) << "unsupported number " << scales_dim 
            << " of scale quantization parameters for UINT8 tensor"
            "per-channel uint8 quantization isn't supported";
        break;
      }
      datatype = xnn_datatype_quint8;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      if (!iodef.quant_param.has_value() 
          || scale_tensor == nullptr || scale_tensor->data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT8
          || zero_tensor == nullptr || zero_tensor->data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT8) {
        break;
      }
      
      if (quantization_params.zero_point == nullptr) {
        LOGS_DEFAULT(VERBOSE) << "missing zero point quantization parameters for int8 tensor";
        break;
      }
      if (scales_dim != zero_dim) {
        LOGS_DEFAULT(VERBOSE) << "mismatching number of scale " << scales_dim
                              << " and zeropoint" << zero_dim << " quantization parameters for INT8 ";
        break;
      }
      if (scales_dim == 1) {
        datatype = xnn_datatype_qint8;
        //nchw right now, check channel dim
      } else if (scales_dim == tensor_shape[1]) {
        auto status = onnxruntime::utils::UnpackInitializerData(*zero_tensor, node_unit.ModelPath(), unpacked_tensor);
        if (!status.IsOK()) {
          LOGS_DEFAULT(ERROR) << "error when unpack zero tensor: " 
                              << ", error msg: " << status.ErrorMessage();
          break;
        }
        const int8_t* zero_points = reinterpret_cast<const int8_t*>(unpacked_tensor.data());
        for (size_t i = 0; i < unpacked_tensor.size(); i++) {
          if (zero_points[i] != 0) {
            LOGS_DEFAULT(VERBOSE) << "only support 0 as zero point, "
                                  << "zero_points[" << i << "] has value: " << zero_points[i];
            break;
          }
        }
      datatype = xnn_datatype_qcint8;
      } else {
        LOGS_DEFAULT(VERBOSE) << "mismatching number of quantization parameters  " << scales_dim
                              << " and outer dimension " << tensor_shape[1];              
      }
      break;
    //now it only support Bias as Int32
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      if (quantization_params.zero_point == nullptr) {
        LOGS_DEFAULT(VERBOSE) << "missing zero point quantization parameters for "
                                 "UINT8 tensor";
        break;
      }
      if (zero_tensor) {
        auto status = onnxruntime::utils::UnpackInitializerData(*zero_tensor, node_unit.ModelPath(), unpacked_tensor);
        if (!status.IsOK()) {
          LOGS_DEFAULT(ERROR) << "error when unpack zero tensor: "
                              << ", error msg: " << status.ErrorMessage();
          break;
        }
      } else {
        break;
      }
      if (scales_dim == 1) {
        if (unpacked_tensor.size() < 1 || unpacked_tensor[0] != 0) {
          LOGS_DEFAULT(ERROR) << "unsupported zero-point value for INT32 ";
          break;
        }
        datatype = xnn_datatype_qint32;
        // nchw right now, check channel dim
      } else if (scales_dim == tensor_shape[1]) {
        const int32_t* zero_points = reinterpret_cast<const int32_t*>(unpacked_tensor.data());
        for (size_t i = 0; i < unpacked_tensor.size(); i++) {
          if (zero_points[i] != 0) {
            LOGS_DEFAULT(VERBOSE) << "unsupported zero-point value in channel "
                                  << "zero_points[" << i << "] has value: " << zero_points[i];
            break;
          }
        }
        datatype = xnn_datatype_qcint32;
      } else {
        LOGS_DEFAULT(VERBOSE) << "mismatching number of quantization parameters %d and outer ";
        break;
      }
      break;
     //TODO
    /* case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      break;
      */
    default:
      break;
  }
  return datatype;
}


#pragma region op_conv
//this function is refered to xnnpack_conv
static xnn_compute_type ValidateXnnpackConvtype(
    xnn_datatype input_datatype,
    xnn_datatype filter_datatype,
    xnn_datatype* bias_datatype,
    xnn_datatype output_datatype) {
  switch (filter_datatype) {
    case xnn_datatype_fp32:
      if (input_datatype == xnn_datatype_fp32 &&
          (!bias_datatype || *bias_datatype == xnn_datatype_fp32) &&
          output_datatype == xnn_datatype_fp32) {
        return xnn_compute_type_fp32;
      }
      break;
#ifndef XNN_NO_QS8_OPERATORS
    case xnn_datatype_qint8:
      if (input_datatype == xnn_datatype_qint8 &&
          (!bias_datatype || *bias_datatype == xnn_datatype_qint32) &&
          output_datatype == xnn_datatype_qint8) {
        return xnn_compute_type_qs8;
      }
      break;
    case xnn_datatype_qcint8:
      if (input_datatype == xnn_datatype_qint8 &&
          (!bias_datatype || *bias_datatype == xnn_datatype_qcint32) &&
          output_datatype == xnn_datatype_qint8) {
        return xnn_compute_type_qc8;
      }
      break;
#endif  // !defined(XNN_NO_QS8_OPERATORS)
#ifndef XNN_NO_QU8_OPERATORS
    case xnn_datatype_quint8:
      if (input_datatype == xnn_datatype_quint8 &&
          (!bias_datatype || *bias_datatype == xnn_datatype_qint32) &&
          output_datatype == xnn_datatype_quint8) {
        return xnn_compute_type_qu8;
      }
      break;
#endif  // !defined(XNN_NO_QU8_OPERATORS)
    default:
      break;
  }
  return xnn_compute_type_invalid;
}
// xnnpack support qc8|qs8|qu8
/*
 * | conv type| input dtype|weight dtype| per channel|zero point handle|
 * | qc8      |  i8        | i8         |  yes       |zero
 * | qcu8     |  xx        | xx         |  yes       | not surpported yet
 * | qs8      |  i8        | i8         |  no        |orig_zp
 * | qu8      |  u8        | u8         |  no        |orig_zp + 128
 */
//
static bool isValidQuantConv(const onnxruntime::NodeUnit& node_unit, const onnxruntime::GraphViewer& graph) {
  bool supported = false;
  do {
    xnn_datatype x_input_type, w_input_type, bias_input_type, output_type;
    //quant conv has at least two inputs, x_tensor and weight
    const auto& inputs = node_unit.Inputs();
    x_input_type = GetDtypeInXnnpack(node_unit, 0, false, graph);
    w_input_type = GetDtypeInXnnpack(node_unit, 1, false, graph);
    bias_input_type = inputs.size() > 2 ? GetDtypeInXnnpack(node_unit, 2, false, graph) : xnn_datatype_invalid;
    output_type = GetDtypeInXnnpack(node_unit, 0, true, graph);
    xnn_compute_type conv_type = ValidateXnnpackConvtype(x_input_type, w_input_type, &bias_input_type, output_type);
    if (xnn_compute_type_invalid == conv_type) {
      break;
    }
    supported = true;
  } while (false);
  return supported;
}

// helper to check whether an ONNX Conv node is supported by the NHWC version
// if this returns true, the layout transformer will be run by GraphPartitioner to convert the first input/output to
// NHWC format, and move the node to the internal NHWC domain.
bool IsConvOnnxNodeSupported(const onnxruntime::NodeUnit& nodeunit, const onnxruntime::GraphViewer& graph) {
  bool supported = false;
  //we have a seperate function to handle quantizedOp, should we check quantized weight and conv attributes here?
  if (IsQuantizedConv(GetQuantizedOpType(nodeunit))) {
    return isValidQuantConv(nodeunit, graph);
  }
  
  const onnxruntime::Node& node = nodeunit.GetNode();
  // use do {} while(false) so it's easier to set a breakpoint on the return
  do {
    // Conv has at least 2 inputs.
    auto input_defs = node.InputDefs();
    const auto& x_arg = *input_defs[0];
    const auto& weight_arg = *input_defs[1];

    int32_t a_input_type;
    // we only support float currently
    if (!GetType(x_arg, a_input_type)
        || a_input_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      break;
    }

    // we only support 2D (4 dims with batch and channel)
    const auto* x_shape = x_arg.Shape();
    if (!x_shape || x_shape->dim_size() != 4) {
      break;
    }

    // require C, H, W to be known so we can construct the xnnpack kernel prior to Compute
    if (!x_shape->dim(1).has_dim_value() ||
        !x_shape->dim(2).has_dim_value() ||
        !x_shape->dim(3).has_dim_value()) {
      break;
    }

    // weight must be constant and also rank 4
    const auto* weight = graph.GetConstantInitializer(weight_arg.Name(), true);
    if (weight == nullptr || weight->dims_size() != 4) {
      break;
    }

    // if there's a bias input it must be constant
    if (input_defs.size() == 3) {
      const auto& bias_arg = *input_defs[2];
      if (bias_arg.Exists() && !graph.IsConstantInitializer(bias_arg.Name(), true)) {
        break;
      }
    }

    onnxruntime::ProtoHelperNodeContext nc(node);
    onnxruntime::OpNodeProtoHelper info(&nc);

    // 'group' value needs to be 1 or C.
    // the second dim of weight is C/group, so if that == 1, group == C
    int64_t group = 0;
    info.GetAttrOrDefault<int64_t>("group", &group, 1);
    if (group != 1 && weight->dims(1) != 1) {
      break;
    }

    // if 'pads' is not specified we use 'auto_pad'
    if (graph_utils::GetNodeAttribute(node, "pads") == nullptr) {
      AutoPadType auto_pad = AutoPadType::NOTSET;

      std::string auto_pad_str;
      if (info.GetAttr<std::string>("auto_pad", &auto_pad_str).IsOK()) {
        // auto_pad was set
        //
        // The "auto_pad_str" string must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID
        // tf2onnx converter doesn't use SAME_LOWER.
        // SAME_UPPER maps to TF SAME padding.
        // TODO: What does PT converter use? We need to support models from PT in mobile.
        auto_pad = StringToAutoPadType(auto_pad_str);
        if (!IsPaddingTypeSupported(auto_pad)) {
          break;
        }
      }
    }

    supported = true;
  } while (false);

  return supported;
}
#pragma endregion op_conv
#pragma region op_maxpool
bool IsMaxPoolOnnxNodeSupported(const onnxruntime::NodeUnit& nodeunit, const onnxruntime::GraphViewer& /*graph*/) {
  bool supported = false;
  //TODO, support quantized pool
  if (IsQuantizedMaxPool(GetQuantizedOpType(nodeunit))) {
    return supported;
  }
  const onnxruntime::Node& node = nodeunit.GetNode();
  // use do {} while(false) so it's easier to set a breakpoint on the return
  do {
    // MaxPool has 1 input.
    auto input_defs = node.InputDefs();
    const auto& x_arg = *input_defs[0];

    // we only support float currently
    const auto* x_type = x_arg.TypeAsProto();
    if (x_type == nullptr ||
        x_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      break;
    }

    // we only support 2D (4 dims with batch and channel)
    const auto* x_shape = x_arg.Shape();
    if (!x_shape || x_shape->dim_size() != 4) {
      break;
    }

    // require C, H, W to be known so we can construct the xnnpack kernel prior to Compute
    if (!x_shape->dim(1).has_dim_value() ||
        !x_shape->dim(2).has_dim_value() ||
        !x_shape->dim(3).has_dim_value()) {
      break;
    }

    // we don't support creating the optional 'I' output
    const auto& output_defs = node.OutputDefs();
    if (output_defs.size() == 2 && output_defs[1]->Exists()) {
      break;
    }

    onnxruntime::ProtoHelperNodeContext nc(node);
    onnxruntime::OpNodeProtoHelper info(&nc);
    onnxruntime::PoolAttributes pool_attrs(info, "MaxPool", node.SinceVersion());

    // xnnpack doesn't appear to support using 'ceil' to calculate the output shape
    // https://github.com/google/XNNPACK/blob/3caa8b9de973839afa1e2a1462ff356e6927a66b/src/operators/max-pooling-nhwc.c#L256
    // calls compute_output_dimension but there's no ability to specify rounding that value up.
    if (pool_attrs.ceil_mode != 0) {
      break;
    }

    if (!IsPaddingTypeSupported(pool_attrs.auto_pad)) {
      break;
    }

    if ((pool_attrs.kernel_shape.size() != 2) ||
        (pool_attrs.kernel_shape[0] == 1 && pool_attrs.kernel_shape[1] == 1)) {
      // XNNPack doesn't support 1x1 maxpool.
      break;
    }

    supported = true;
  } while (false);

  return supported;
}
#pragma endregion op_maxpool
}  // namespace xnnpack
}  // namespace onnxruntime
