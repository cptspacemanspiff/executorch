/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <c10/util/irange.h>

#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

Tensor& ones_out(KernelRuntimeContext& ctx, IntArrayRef size, Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, size) == Error::Ok, InvalidArgument, out);

  ScalarType out_type = out.scalar_type();
  ET_SWITCH_REALHBBF16_TYPES(out_type, ctx, __func__, CTYPE, [&] {
    auto out_data = out.mutable_data_ptr<CTYPE>();
    for (const auto i : c10::irange(out.numel())) {
      out_data[i] = static_cast<CTYPE>(1);
    }
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
