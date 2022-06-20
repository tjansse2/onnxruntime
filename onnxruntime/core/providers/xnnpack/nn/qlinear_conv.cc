// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/framework/transpose_helper.h"
#include "core/providers/xnnpack/detail/utils.h"

#include <xnnpack.h>

namespace onnxruntime {
namespace xnnpack {
struct QuantParam {
  uint8_t X_zero_point_value;
  uint8_t W_zero_point_value;
  uint8_t Y_zero_point_value;

  float X_scale_value;
  float W_scale_value;
  float Y_scale_value;
};

namespace {
Status CreateXnnpackKernel(const ConvAttributes& conv_attrs,
                           int64_t C, int64_t M,
                           const TensorShapeVector& kernel_shape,
                           const std::optional<std::pair<uint8_t, uint8_t>>& clip_min_max,
                           const Tensor& W, const int32_t* B_data,
                           struct xnn_operator*& p,
                           QuantParam& quant_param
#ifdef XNN_CACHE_ENABLE
                           ,
                           xnn_caches_t caches_t
#endif
) {

  const auto* W_data = W.Data<uint8_t>();

  int64_t group_count = conv_attrs.group;
  int64_t group_input_channels = C / group_count;
  int64_t group_output_channels = M / group_count;
  uint8_t output_min = clip_min_max ? clip_min_max->first : 0;
  uint8_t output_max = clip_min_max ? clip_min_max->second : 255;

  auto status = xnn_create_convolution2d_nhwc_qu8(
      static_cast<uint32_t>(conv_attrs.pads[0]),
      static_cast<uint32_t>(conv_attrs.pads[3]),
      static_cast<uint32_t>(conv_attrs.pads[2]),
      static_cast<uint32_t>(conv_attrs.pads[1]),
      static_cast<uint32_t>(kernel_shape[0]),
      static_cast<uint32_t>(kernel_shape[1]),
      static_cast<uint32_t>(conv_attrs.strides[0]),
      static_cast<uint32_t>(conv_attrs.strides[1]),
      static_cast<uint32_t>(conv_attrs.dilations[0]),
      static_cast<uint32_t>(conv_attrs.dilations[1]),
      static_cast<uint32_t>(group_count),
      static_cast<size_t>(group_input_channels),
      static_cast<size_t>(group_output_channels),
      static_cast<size_t>(C),
      static_cast<size_t>(M),
      quant_param.X_zero_point_value, quant_param.X_scale_value,
      quant_param.W_zero_point_value, quant_param.W_scale_value,
      W_data, B_data,
      quant_param.Y_zero_point_value, quant_param.Y_scale_value,
      output_min, output_max,
      0,  // flags
#ifdef XNN_CACHE_ENABLE
      caches_t,
#endif
      &p);
  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to create xnnpack kernel. xnn_create_convolution2d_nhwc_qu8 returned ", status);
  }
  return Status::OK();
}
}  // namespace

class QLinearConv : public OpKernel {
 public:
  explicit QLinearConv(const OpKernelInfo& info) : OpKernel(info),
                                                   conv_attrs_(info) {

    const auto& node{Node()};

    const auto& input_defs = node.InputDefs();
    const NodeArg& X = *input_defs[0];
    C_ = X.Shape()->dim(3).dim_value();  // input is NHWC. op support checker made sure C dim was known

    // as the weight input is a constant initializer we can calculate all the sizes here instead of in Compute
    const Tensor* W = nullptr;
    ORT_ENFORCE(info.TryGetConstantInput(InputTensors::IN_W, &W),
                "Weight input was not constant initializer. XNNPACK EP should not have asked for the node. Node name:",
                node.Name());
    //quant param, which used in create xnnpack_conv_kernel
    const Tensor* X_zero_point = nullptr;
    const Tensor* W_zero_point = nullptr;
    const Tensor* Y_zero_point = nullptr;
    ORT_ENFORCE(info.TryGetConstantInput(InputTensors::IN_X_ZERO_POINT,&X_zero_point),
        "X_zero_point input was not constant initializer. XNNPACK EP should not have asked for the node. Node name:",
                node.Name());
    ORT_ENFORCE(info.TryGetConstantInput(InputTensors::IN_W_ZERO_POINT, &W_zero_point),
                "W_zero_point input was not constant initializer. XNNPACK EP should not have asked for the node. Node name:",
                node.Name());
    ORT_ENFORCE(info.TryGetConstantInput(InputTensors::IN_Y_ZERO_POINT, &Y_zero_point),
                "Y_zero_point input was not constant initializer. XNNPACK EP should not have asked for the node. Node name:",
                node.Name());

    quant_param_.X_zero_point_value = *(X_zero_point->template Data<uint8_t>());
    quant_param_.W_zero_point_value = *(W_zero_point->template Data<uint8_t>());
    quant_param_.Y_zero_point_value = *(Y_zero_point->template Data<uint8_t>());

    const Tensor* X_scale = nullptr;
    const Tensor* W_scale = nullptr;
    const Tensor* Y_scale = nullptr;
    ORT_ENFORCE(info.TryGetConstantInput(InputTensors::IN_X_SCALE, &X_scale),
                "X_scale input was not constant initializer. XNNPACK EP should not have asked for the node. Node name:",
                node.Name());
    ORT_ENFORCE(info.TryGetConstantInput(InputTensors::IN_W_SCALE, &W_scale),
                "W_scale input was not constant initializer. XNNPACK EP should not have asked for the node. Node name:",
                node.Name());
    ORT_ENFORCE(info.TryGetConstantInput(InputTensors::IN_Y_SCALE, &Y_scale),
                "Y_scale input was not constant initializer. XNNPACK EP should not have asked for the node. Node name:",
                node.Name());

    quant_param_.X_scale_value = *(X_scale->template Data<float>());
    quant_param_.W_scale_value = *(W_scale->template Data<float>());
    quant_param_.Y_scale_value = *(Y_scale->template Data<float>());

    // 'M' is first dim of weight. Prepacking will alter the layout of W later
    M_ = W->Shape()[0];

    // this happens before PrePack, so the W input is still in the ONNX spec format
    ORT_THROW_IF_ERROR(conv_attrs_.ComputeKernelShape(W->Shape(), kernel_shape_));

    if (conv_attrs_.pads.empty()) {
      conv_attrs_.pads.resize(kernel_shape_.size() * 2, 0);
    }

    if (conv_attrs_.dilations.empty()) {
      conv_attrs_.dilations.resize(kernel_shape_.size(), 1);
    }

    if (conv_attrs_.strides.empty()) {
      conv_attrs_.strides.resize(kernel_shape_.size(), 1);
    }

    // we only take nodes with no bias, or a constant bias.
    bool has_bias = input_defs.size() == (InputTensors::IN_BIAS + 1) && input_defs[InputTensors::IN_BIAS]->Exists();

    ORT_ENFORCE(has_bias == false || info.TryGetConstantInput(InputTensors::IN_BIAS, &B_),
                "Invalid Node with non-constant Bias input. XNNPACK EP should not have asked for the node. Node name:",
                node.Name());
#ifdef XNN_CACHE_ENABLE
#if XNN_PLATFORM_JIT
    xnn_init_code_cache(&code_cache_);
#endif
    caches_.code_cache = &code_cache_;
#endif
  }

  Status Compute(OpKernelContext* context) const override;
  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 bool& is_packed,              // out
                 PrePackedWeights*) override;  // out
 private:
  enum InputTensors : int {
    IN_X = 0,
    IN_X_SCALE = 1,
    IN_X_ZERO_POINT = 2,
    IN_W = 3,
    IN_W_SCALE = 4,
    IN_W_ZERO_POINT = 5,
    IN_Y_SCALE = 6,
    IN_Y_ZERO_POINT = 7,
    IN_BIAS = 8
  };

  enum OutputTensors : int {
    OUT_Y = 0
  };
  QuantParam quant_param_;
  ConvAttributes conv_attrs_;
  std::unique_ptr<Tensor> packed_w_;
  const Tensor* B_{nullptr};
  TensorShapeVector kernel_shape_;
  int64_t C_;
  int64_t M_;
  XnnpackOperator op0_ = nullptr;
  std::optional<std::pair<uint8_t, uint8_t>> clip_min_max_;
#ifdef XNN_CACHE_ENABLE
  xnn_code_cache code_cache_;
  xnn_caches caches_;
#endif
  };

// use PrePack to handle the weight layout change as that's not a simple NCHW -> NHWC transpose
Status QLinearConv::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                            bool& is_packed,      // out
                            PrePackedWeights*) {  // out
  is_packed = false;

  // only layout of weight input is adjusted via PrePack
  if (input_idx != InputTensors::IN_W) {
    return Status::OK();
  }
  // Transpose from {M, C/group, kH, kW} to {M, kH, kW, C/group}
  auto orig_shape = tensor.Shape();

  std::vector<size_t> perm{0, 2, 3, 1};
  std::vector<int64_t> new_dims{orig_shape[0],
                                orig_shape[2],
                                orig_shape[3],
                                orig_shape[1]};

  packed_w_ = Tensor::Create(tensor.DataType(), TensorShape(new_dims), alloc);
  //                                           from to
  SingleAxisTranspose(perm, tensor, *packed_w_, 1, 3);
  is_packed = true;

  // we can create the kernel now
  struct xnn_operator* p = nullptr;
#ifdef XNN_CACHE_ENABLE
  ORT_RETURN_IF_ERROR(CreateXnnpackKernel(conv_attrs_, C_, M_, kernel_shape_, clip_min_max_,
                                          *packed_w_, B_ ? B_->Data<int32_t>() : nullptr, p,
      quant_param_ ,&xnn_caches));
#else
  ORT_RETURN_IF_ERROR(CreateXnnpackKernel(conv_attrs_, C_, M_, kernel_shape_, clip_min_max_,
                                          *packed_w_, B_ ? B_->Data<int32_t>() : nullptr, p,
                                          quant_param_));
#endif

  op0_.reset(p);
  return Status::OK();
}

Status QLinearConv::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);  // input is NHWC
  const auto& X_shape = X->Shape();
  const int64_t N = X_shape[0];

  const int64_t H = X_shape[1];
  const int64_t W = X_shape[2];

  const size_t kernel_rank = kernel_shape_.size();

  TensorShapeVector Y_dims({N});
  TensorShape input_shape = {H, W};
  ConvAttributes::ConvPadVector pads(conv_attrs_.pads);
  if (pads.empty()) {
    pads.resize(kernel_rank * 2, 0);
  }
  ORT_RETURN_IF_ERROR(conv_attrs_.InferPadsAndOutputShape(input_shape, kernel_shape_,
                                                          conv_attrs_.strides, conv_attrs_.dilations, pads,
                                                          Y_dims));
  Y_dims.push_back(M_);
  Tensor* Y = context->Output(0, TensorShape(Y_dims));
  // Bail out early if one of the dimensions is zero.
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  const auto* Xdata = X->Data<uint8_t>();
  auto* Ydata = Y->MutableData<uint8_t>();

  ORT_ENFORCE(xnn_status_success == xnn_setup_convolution2d_nhwc_qu8(op0_.get(), N, input_shape[0], input_shape[1], Xdata, Ydata, nullptr));
  ORT_ENFORCE(xnn_status_success == xnn_run_operator(op0_.get(), nullptr));

  return Status::OK();
}

// Register an alternate version of this kernel that supports the channels_last
// attribute in order to consume and produce NHWC tensors.
/*
ONNX_OPERATOR_KERNEL_EX(
    QLinearConv,
    kMSInternalNHWCDomain,
    10,
    kXnnpackExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    QLinearConv);
*/

ONNX_OPERATOR_TYPED_KERNEL_EX(
    QLinearConv,
    kMSInternalNHWCDomain,
    10,
    uint8_t,
    kXnnpackExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint8_t>()),
    QLinearConv);

}  // namespace xnnpack
}  // namespace onnxruntime
