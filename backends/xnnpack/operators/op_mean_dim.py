# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict

import torch
from executorch.backends.xnnpack.operators.node_visitor import (
    check_or_raise,
    get_tensor_value,
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    XNNGlobalAvgPooling2d,
    XNNGraph,
    XNNStaticMean,
    XNode,
)
from executorch.backends.xnnpack.utils.utils import normalize_mean_dims
from executorch.backends.xnnpack.utils.xnnpack_constants import XNN_FLAG_KEEP_DIMS


@register_node_visitor
class MeanDim(NodeVisitor):
    """
    Lowers aten.mean.dim to XNNPACK.

    Two paths:
      * The 4D / dims=[2,3] / keepdim special case maps to Global Average Pooling
        (the historical, well-trodden path; tensors are converted to NHWC).
      * Everything else (e.g. RMSNorm's mean over the last dim of a 3D tensor) maps
        to a generic static reduce (xnn_define_static_reduce), keeping the natural
        tensor layout.

    XNNPACK reduce has no dtype-override field, so mean.dim with an explicit dtype
    is rejected (and left non-delegated) in both paths.
    """

    target = "aten.mean.dim"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:
        check_or_raise(
            node.kwargs.get("dtype") is None,
            "XNNPACK does not support mean.dim with dtype",
        )

        input_node = cast(torch.fx.Node, node.args[0])
        input_rank = input_node.meta["val"].dim()
        mean_dims = normalize_mean_dims(node.args[1], input_rank)
        keepdim = len(node.args) == 3 and bool(node.args[2])

        use_gap = input_rank == 4 and sorted(mean_dims) == [2, 3] and keepdim

        self.define_nodes_tensor_inputs_outputs(
            node, xnn_graph, vals_to_ids, convert_to_nhwc=use_gap
        )
        input_id = vals_to_ids[input_node]
        output_id = vals_to_ids[node]

        if use_gap:
            input_shape = get_tensor_value(xnn_graph.xvalues[input_id]).dims
            check_or_raise(
                len(input_shape) == 4, "Require input to mean.dim be 4 dimensional"
            )
            ser_node = XNode(
                xnode_union=XNNGlobalAvgPooling2d(
                    input_id=input_id, output_id=output_id, flags=XNN_FLAG_KEEP_DIMS
                ),
                debug_handle=debug_handle,
            )
            xnn_graph.xnodes.append(ser_node)
            return

        # General reduce path (natural layout, arbitrary axes / rank).
        axes = sorted(mean_dims)
        ser_node = XNode(
            xnode_union=XNNStaticMean(
                num_reduction_axes=len(axes),
                reduction_axes=axes,
                input_id=input_id,
                output_id=output_id,
                flags=XNN_FLAG_KEEP_DIMS if keepdim else 0,
            ),
            debug_handle=debug_handle,
        )
        xnn_graph.xnodes.append(ser_node)
