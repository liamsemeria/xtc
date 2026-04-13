# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_mlir

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

# Small conv2d
N, H, W, F, R, S, C, SH, SW, dtype = 1, 8, 8, 16, 5, 5, 3, 2, 2, "float32"
a = O.tensor((N, H, W, C), dtype, name="I")
b = O.tensor((R, S, C, F), dtype, name="W")

with O.graph(name="pad_conv2d_nhwc_mini") as gb:
    p = O.pad2d(a, padding=2, axes=(1, 2), name="pad")
    O.conv2d(p, b, stride=(SH, SW), name="conv")

graph = gb.graph
print(graph)

impl = Backend(graph, use_tensor_dialect=True)

sch = impl.get_scheduler()
sch.fuse_producer_at("w",0)
sch.tile("w", {"w1": 4})
sch.tile("f", {"f1": 16})
sch.interchange(["b", "h", "w", "f", "r", "s", "c", "w1", "f1"])
sch.vectorize(["f1"])
sch.unroll({"w1": 4, "c": 3})
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="pad_conv2d_nhwc_mini_mlir_tensor",
    print_source_ir=True,
    print_transformed_ir=True,
    print_bufferization_ir=True,
    #print_lowered_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK: // -----// IR Dump Before transform //----- //
# CHECK-NEXT: #map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
# CHECK-NEXT: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
# CHECK-NEXT: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT: #map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @pad_conv2d_nhwc_mini(%arg0: tensor<1x8x8x3xf32> {llvm.noalias}, %arg1: tensor<5x5x3x16xf32> {llvm.noalias}, %arg2: memref<1x4x4x16xf32> {llvm.noalias}) {
# CHECK-NEXT:     %0 = tensor.empty() : tensor<1x12x12x3xf32>
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%0 : tensor<1x12x12x3xf32>) attrs =  {__xtc_id_pad_} {
# CHECK-NEXT:     ^bb0(%out: f32):
# CHECK-NEXT:       %5 = linalg.index 0 : index
# CHECK-NEXT:       %6 = linalg.index 1 : index
# CHECK-NEXT:       %7 = linalg.index 2 : index
# CHECK-NEXT:       %8 = linalg.index 3 : index
# CHECK-NEXT:       %c0 = arith.constant 0 : index
# CHECK-NEXT:       %c0_1 = arith.constant 0 : index
# CHECK-NEXT:       %9 = arith.subi %5, %c0_1 : index
# CHECK-NEXT:       %c1 = arith.constant 1 : index
# CHECK-NEXT:       %10 = arith.cmpi sge, %9, %c0 : index
# CHECK-NEXT:       %11 = arith.cmpi slt, %9, %c1 : index
# CHECK-NEXT:       %c2 = arith.constant 2 : index
# CHECK-NEXT:       %12 = arith.subi %6, %c2 : index
# CHECK-NEXT:       %c8 = arith.constant 8 : index
# CHECK-NEXT:       %13 = arith.cmpi sge, %12, %c0 : index
# CHECK-NEXT:       %14 = arith.cmpi slt, %12, %c8 : index
# CHECK-NEXT:       %c2_2 = arith.constant 2 : index
# CHECK-NEXT:       %15 = arith.subi %7, %c2_2 : index
# CHECK-NEXT:       %c8_3 = arith.constant 8 : index
# CHECK-NEXT:       %16 = arith.cmpi sge, %15, %c0 : index
# CHECK-NEXT:       %17 = arith.cmpi slt, %15, %c8_3 : index
# CHECK-NEXT:       %c0_4 = arith.constant 0 : index
# CHECK-NEXT:       %18 = arith.subi %8, %c0_4 : index
# CHECK-NEXT:       %c3 = arith.constant 3 : index
# CHECK-NEXT:       %19 = arith.cmpi sge, %18, %c0 : index
# CHECK-NEXT:       %20 = arith.cmpi slt, %18, %c3 : index
# CHECK-NEXT:       %21 = arith.andi %10, %11 : i1
# CHECK-NEXT:       %22 = arith.andi %21, %13 : i1
# CHECK-NEXT:       %23 = arith.andi %22, %14 : i1
# CHECK-NEXT:       %24 = arith.andi %23, %16 : i1
# CHECK-NEXT:       %25 = arith.andi %24, %17 : i1
# CHECK-NEXT:       %26 = arith.andi %25, %19 : i1
# CHECK-NEXT:       %27 = arith.andi %26, %20 : i1
# CHECK-NEXT:       %28 = scf.if %27 -> (f32) {
# CHECK-NEXT:         %extracted = tensor.extract %arg0[%9, %12, %15, %18] : tensor<1x8x8x3xf32>
# CHECK-NEXT:         scf.yield %extracted : f32
# CHECK-NEXT:       } else {
# CHECK-NEXT:         scf.yield %cst : f32
# CHECK-NEXT:       }
# CHECK-NEXT:       linalg.yield %28 : f32
# CHECK-NEXT:     } -> tensor<1x12x12x3xf32>
# CHECK-NEXT:     %2 = tensor.empty() : tensor<1x4x4x16xf32>
# CHECK-NEXT:     %cst_0 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %3 = linalg.fill {__xtc_id_conv_0_} ins(%cst_0 : f32) outs(%2 : tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
# CHECK-NEXT:     %4 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%1, %arg1 : tensor<1x12x12x3xf32>, tensor<5x5x3x16xf32>) outs(%3 : tensor<1x4x4x16xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:     ^bb0(%in: f32, %in_1: f32, %out: f32):
# CHECK-NEXT:       %5 = arith.mulf %in, %in_1 : f32
# CHECK-NEXT:       %6 = arith.addf %out, %5 : f32
# CHECK-NEXT:       linalg.yield %6 : f32
# CHECK-NEXT:     } -> tensor<1x4x4x16xf32>
# CHECK-NEXT:     bufferization.materialize_in_destination %4 in restrict writable %arg2 : (tensor<1x4x4x16xf32>, memref<1x4x4x16xf32>) -> ()
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT:   transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:     transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:     transform.yield 
# CHECK-NEXT:   }
# CHECK-NEXT:   transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:     %0 = transform.structured.match attributes {__xtc_id_conv_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0, 0, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops "./b" : !transform.any_op
# CHECK-NEXT:     %1 = transform.structured.match attributes {__xtc_id_pad_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %1 into %loops : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     %2 = transform.structured.match attributes {__xtc_id_conv_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %2 tile_sizes [0, 1, 0, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_1 "./h" : !transform.any_op
# CHECK-NEXT:     %3 = transform.structured.match attributes {__xtc_id_pad_} in %new_containing_op : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     %fused_op_2, %new_containing_op_3 = transform.structured.fuse_into_containing_op %3 into %loops_1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     %4 = transform.structured.match attributes {__xtc_id_conv_} in %new_containing_op : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %4 tile_sizes [0, 0, 4, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_5 "./w" : !transform.any_op
# CHECK-NEXT:     %5 = transform.structured.match attributes {__xtc_id_pad_} in %new_containing_op_3 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     %fused_op_6, %new_containing_op_7 = transform.structured.fuse_into_containing_op %5 into %loops_5 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     %6 = transform.structured.match attributes {__xtc_id_conv_} in %new_containing_op_3 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %6 tile_sizes [0, 0, 0, 16, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_9 "./f" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %tiled_linalg_op_8 tile_sizes [0, 0, 0, 0, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_11 "./r" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_12, %loops_13 = transform.structured.tile_using_for %tiled_linalg_op_10 tile_sizes [0, 0, 0, 0, 0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_13 "./s" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_14, %loops_15 = transform.structured.tile_using_for %tiled_linalg_op_12 tile_sizes [0, 0, 0, 0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_15 "./c" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_16, %loops_17 = transform.structured.tile_using_for %tiled_linalg_op_14 tile_sizes [0, 0, 1, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_17 "./w1" : !transform.any_op
# CHECK-NEXT:     transform.include @_vecto failures(suppress) (%tiled_linalg_op_16) : (!transform.any_op) -> ()
# CHECK-NEXT:     transform.loop.unroll %loops_17 {factor = 4 : i64} : !transform.any_op
# CHECK-NEXT:     transform.loop.unroll %loops_15 {factor = 3 : i64} : !transform.any_op
# CHECK-NEXT:     %7 = transform.get_parent_op %new_containing_op {isolated_from_above} : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     transform.apply_patterns to %7 {
# CHECK-NEXT:       transform.apply_patterns.vector.reduction_to_contract
# CHECK-NEXT:       transform.apply_patterns.vector.transfer_permutation_patterns
# CHECK-NEXT:     } : !transform.any_op
# CHECK-NEXT:     transform.apply_patterns to %7 {
# CHECK-NEXT:       transform.apply_patterns.vector.lower_outerproduct
# CHECK-NEXT:       transform.apply_patterns.vector.lower_contraction
# CHECK-NEXT:     } : !transform.any_op
# CHECK-NEXT:     transform.yield 
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: // -----// IR Dump After transform //----- //
# CHECK-NEXT: #map = affine_map<(d0) -> (d0 * 2)>
# CHECK-NEXT: #map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
# CHECK-NEXT: #map2 = affine_map<(d0)[s0] -> (d0 + s0)>
# CHECK-NEXT: #map3 = affine_map<(d0)[s0] -> (d0 * 2 + s0)>
# CHECK-NEXT: #map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
# CHECK-NEXT: #map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT: #map6 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @pad_conv2d_nhwc_mini(%arg0: tensor<1x8x8x3xf32> {llvm.noalias}, %arg1: tensor<5x5x3x16xf32> {llvm.noalias}, %arg2: memref<1x4x4x16xf32> {llvm.noalias}) {
# CHECK-NEXT:     %c6 = arith.constant 6 : index
# CHECK-NEXT:     %c5 = arith.constant 5 : index
# CHECK-NEXT:     %c16 = arith.constant 16 : index
# CHECK-NEXT:     %c4 = arith.constant 4 : index
# CHECK-NEXT:     %c3 = arith.constant 3 : index
# CHECK-NEXT:     %c8 = arith.constant 8 : index
# CHECK-NEXT:     %c2 = arith.constant 2 : index
# CHECK-NEXT:     %c1 = arith.constant 1 : index
# CHECK-NEXT:     %c0 = arith.constant 0 : index
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %0 = tensor.empty() : tensor<1x12x12x3xf32>
# CHECK-NEXT:     %1 = tensor.empty() : tensor<1x4x4x16xf32>
# CHECK-NEXT:     %2 = linalg.fill {__xtc_id_conv_0_} ins(%cst : f32) outs(%1 : tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
# CHECK-NEXT:     %3 = scf.for %arg3 = %c0 to %c1 step %c1 iter_args(%arg4 = %2) -> (tensor<1x4x4x16xf32>) {
# CHECK-NEXT:       %extracted_slice = tensor.extract_slice %0[%arg3, 0, 0, 0] [1, 11, 11, 3] [1, 1, 1, 1] : tensor<1x12x12x3xf32> to tensor<1x11x11x3xf32>
# CHECK-NEXT:       %extracted_slice_0 = tensor.extract_slice %arg4[%arg3, 0, 0, 0] [1, 4, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> to tensor<1x4x4x16xf32>
# CHECK-NEXT:       %4 = scf.for %arg5 = %c0 to %c4 step %c1 iter_args(%arg6 = %extracted_slice_0) -> (tensor<1x4x4x16xf32>) {
# CHECK-NEXT:         %5 = affine.apply #map(%arg5)
# CHECK-NEXT:         %extracted_slice_1 = tensor.extract_slice %extracted_slice[0, %5, 0, 0] [1, 5, 11, 3] [1, 1, 1, 1] : tensor<1x11x11x3xf32> to tensor<1x5x11x3xf32>
# CHECK-NEXT:         %extracted_slice_2 = tensor.extract_slice %arg6[0, %arg5, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> to tensor<1x1x4x16xf32>
# CHECK-NEXT:         %6 = scf.for %arg7 = %c0 to %c4 step %c4 iter_args(%arg8 = %extracted_slice_2) -> (tensor<1x1x4x16xf32>) {
# CHECK-NEXT:           %7 = affine.apply #map(%arg7)
# CHECK-NEXT:           %extracted_slice_4 = tensor.extract_slice %extracted_slice_1[0, 0, %7, 0] [1, 5, 11, 3] [1, 1, 1, 1] : tensor<1x5x11x3xf32> to tensor<1x5x11x3xf32>
# CHECK-NEXT:           %8 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%extracted_slice_4 : tensor<1x5x11x3xf32>) attrs =  {__xtc_id_pad_} {
# CHECK-NEXT:           ^bb0(%out: f32):
# CHECK-NEXT:             %10 = affine.apply #map2(%arg3)[%c0]
# CHECK-NEXT:             %11 = linalg.index 1 : index
# CHECK-NEXT:             %12 = affine.apply #map3(%arg5)[%11]
# CHECK-NEXT:             %13 = linalg.index 2 : index
# CHECK-NEXT:             %14 = affine.apply #map3(%arg7)[%13]
# CHECK-NEXT:             %15 = linalg.index 3 : index
# CHECK-NEXT:             %16 = arith.cmpi sge, %10, %c0 : index
# CHECK-NEXT:             %17 = arith.cmpi slt, %10, %c1 : index
# CHECK-NEXT:             %18 = arith.subi %12, %c2 : index
# CHECK-NEXT:             %19 = arith.cmpi sge, %18, %c0 : index
# CHECK-NEXT:             %20 = arith.cmpi slt, %18, %c8 : index
# CHECK-NEXT:             %21 = arith.subi %14, %c2 : index
# CHECK-NEXT:             %22 = arith.cmpi sge, %21, %c0 : index
# CHECK-NEXT:             %23 = arith.cmpi slt, %21, %c8 : index
# CHECK-NEXT:             %24 = arith.cmpi sge, %15, %c0 : index
# CHECK-NEXT:             %25 = arith.cmpi slt, %15, %c3 : index
# CHECK-NEXT:             %26 = arith.andi %16, %17 : i1
# CHECK-NEXT:             %27 = arith.andi %26, %19 : i1
# CHECK-NEXT:             %28 = arith.andi %27, %20 : i1
# CHECK-NEXT:             %29 = arith.andi %28, %22 : i1
# CHECK-NEXT:             %30 = arith.andi %29, %23 : i1
# CHECK-NEXT:             %31 = arith.andi %30, %24 : i1
# CHECK-NEXT:             %32 = arith.andi %31, %25 : i1
# CHECK-NEXT:             %33 = scf.if %32 -> (f32) {
# CHECK-NEXT:               %extracted = tensor.extract %arg0[%10, %18, %21, %15] : tensor<1x8x8x3xf32>
# CHECK-NEXT:               scf.yield %extracted : f32
# CHECK-NEXT:             } else {
# CHECK-NEXT:               scf.yield %cst : f32
# CHECK-NEXT:             }
# CHECK-NEXT:             linalg.yield %33 : f32
# CHECK-NEXT:           } -> tensor<1x5x11x3xf32>
# CHECK-NEXT:           %extracted_slice_5 = tensor.extract_slice %arg8[0, 0, %arg7, 0] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x4x16xf32>
# CHECK-NEXT:           %9 = scf.for %arg9 = %c0 to %c16 step %c16 iter_args(%arg10 = %extracted_slice_5) -> (tensor<1x1x4x16xf32>) {
# CHECK-NEXT:             %extracted_slice_7 = tensor.extract_slice %arg1[0, 0, 0, %arg9] [5, 5, 3, 16] [1, 1, 1, 1] : tensor<5x5x3x16xf32> to tensor<5x5x3x16xf32>
# CHECK-NEXT:             %extracted_slice_8 = tensor.extract_slice %arg10[0, 0, 0, %arg9] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x4x16xf32>
# CHECK-NEXT:             %10 = scf.for %arg11 = %c0 to %c5 step %c1 iter_args(%arg12 = %extracted_slice_8) -> (tensor<1x1x4x16xf32>) {
# CHECK-NEXT:               %extracted_slice_10 = tensor.extract_slice %8[0, %arg11, 0, 0] [1, 1, 11, 3] [1, 1, 1, 1] : tensor<1x5x11x3xf32> to tensor<1x1x11x3xf32>
# CHECK-NEXT:               %extracted_slice_11 = tensor.extract_slice %extracted_slice_7[%arg11, 0, 0, 0] [1, 5, 3, 16] [1, 1, 1, 1] : tensor<5x5x3x16xf32> to tensor<1x5x3x16xf32>
# CHECK-NEXT:               %11 = scf.for %arg13 = %c0 to %c5 step %c1 iter_args(%arg14 = %arg12) -> (tensor<1x1x4x16xf32>) {
# CHECK-NEXT:                 %extracted_slice_12 = tensor.extract_slice %extracted_slice_10[0, 0, %arg13, 0] [1, 1, 7, 3] [1, 1, 1, 1] : tensor<1x1x11x3xf32> to tensor<1x1x7x3xf32>
# CHECK-NEXT:                 %extracted_slice_13 = tensor.extract_slice %extracted_slice_11[0, %arg13, 0, 0] [1, 1, 3, 16] [1, 1, 1, 1] : tensor<1x5x3x16xf32> to tensor<1x1x3x16xf32>
# CHECK-NEXT:                 %extracted_slice_14 = tensor.extract_slice %extracted_slice_12[0, 0, 0, %c0] [1, 1, 7, 1] [1, 1, 1, 1] : tensor<1x1x7x3xf32> to tensor<1x1x7x1xf32>
# CHECK-NEXT:                 %extracted_slice_15 = tensor.extract_slice %extracted_slice_13[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x3x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %extracted_slice_16 = tensor.extract_slice %extracted_slice_14[0, 0, %c0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_17 = tensor.extract_slice %arg14[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %12 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_16, %extracted_slice_15 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_17 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %24 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %25 = arith.addf %out, %24 : f32
# CHECK-NEXT:                   linalg.yield %25 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_18 = tensor.insert_slice %12 into %arg14[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_19 = tensor.extract_slice %extracted_slice_14[0, 0, %c2, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_20 = tensor.extract_slice %inserted_slice_18[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %13 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_19, %extracted_slice_15 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_20 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %24 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %25 = arith.addf %out, %24 : f32
# CHECK-NEXT:                   linalg.yield %25 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_21 = tensor.insert_slice %13 into %inserted_slice_18[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_22 = tensor.extract_slice %extracted_slice_14[0, 0, %c4, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_23 = tensor.extract_slice %inserted_slice_21[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %14 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_22, %extracted_slice_15 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_23 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %24 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %25 = arith.addf %out, %24 : f32
# CHECK-NEXT:                   linalg.yield %25 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_24 = tensor.insert_slice %14 into %inserted_slice_21[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_25 = tensor.extract_slice %extracted_slice_14[0, 0, %c6, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_26 = tensor.extract_slice %inserted_slice_24[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %15 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_25, %extracted_slice_15 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_26 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %24 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %25 = arith.addf %out, %24 : f32
# CHECK-NEXT:                   linalg.yield %25 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_27 = tensor.insert_slice %15 into %inserted_slice_24[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_28 = tensor.extract_slice %extracted_slice_12[0, 0, 0, %c1] [1, 1, 7, 1] [1, 1, 1, 1] : tensor<1x1x7x3xf32> to tensor<1x1x7x1xf32>
# CHECK-NEXT:                 %extracted_slice_29 = tensor.extract_slice %extracted_slice_13[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x3x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %extracted_slice_30 = tensor.extract_slice %extracted_slice_28[0, 0, %c0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_31 = tensor.extract_slice %inserted_slice_27[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %16 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_30, %extracted_slice_29 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_31 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %24 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %25 = arith.addf %out, %24 : f32
# CHECK-NEXT:                   linalg.yield %25 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_32 = tensor.insert_slice %16 into %inserted_slice_27[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_33 = tensor.extract_slice %extracted_slice_28[0, 0, %c2, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_34 = tensor.extract_slice %inserted_slice_32[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %17 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_33, %extracted_slice_29 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_34 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %24 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %25 = arith.addf %out, %24 : f32
# CHECK-NEXT:                   linalg.yield %25 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_35 = tensor.insert_slice %17 into %inserted_slice_32[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_36 = tensor.extract_slice %extracted_slice_28[0, 0, %c4, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_37 = tensor.extract_slice %inserted_slice_35[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %18 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_36, %extracted_slice_29 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_37 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %24 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %25 = arith.addf %out, %24 : f32
# CHECK-NEXT:                   linalg.yield %25 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_38 = tensor.insert_slice %18 into %inserted_slice_35[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_39 = tensor.extract_slice %extracted_slice_28[0, 0, %c6, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_40 = tensor.extract_slice %inserted_slice_38[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %19 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_39, %extracted_slice_29 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_40 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %24 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %25 = arith.addf %out, %24 : f32
# CHECK-NEXT:                   linalg.yield %25 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_41 = tensor.insert_slice %19 into %inserted_slice_38[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_42 = tensor.extract_slice %extracted_slice_12[0, 0, 0, %c2] [1, 1, 7, 1] [1, 1, 1, 1] : tensor<1x1x7x3xf32> to tensor<1x1x7x1xf32>
# CHECK-NEXT:                 %extracted_slice_43 = tensor.extract_slice %extracted_slice_13[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x3x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %extracted_slice_44 = tensor.extract_slice %extracted_slice_42[0, 0, %c0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_45 = tensor.extract_slice %inserted_slice_41[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %20 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_44, %extracted_slice_43 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_45 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %24 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %25 = arith.addf %out, %24 : f32
# CHECK-NEXT:                   linalg.yield %25 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_46 = tensor.insert_slice %20 into %inserted_slice_41[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_47 = tensor.extract_slice %extracted_slice_42[0, 0, %c2, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_48 = tensor.extract_slice %inserted_slice_46[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %21 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_47, %extracted_slice_43 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_48 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %24 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %25 = arith.addf %out, %24 : f32
# CHECK-NEXT:                   linalg.yield %25 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_49 = tensor.insert_slice %21 into %inserted_slice_46[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_50 = tensor.extract_slice %extracted_slice_42[0, 0, %c4, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_51 = tensor.extract_slice %inserted_slice_49[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %22 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_50, %extracted_slice_43 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_51 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %24 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %25 = arith.addf %out, %24 : f32
# CHECK-NEXT:                   linalg.yield %25 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_52 = tensor.insert_slice %22 into %inserted_slice_49[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_53 = tensor.extract_slice %extracted_slice_42[0, 0, %c6, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_54 = tensor.extract_slice %inserted_slice_52[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %23 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_53, %extracted_slice_43 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_54 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %24 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %25 = arith.addf %out, %24 : f32
# CHECK-NEXT:                   linalg.yield %25 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_55 = tensor.insert_slice %23 into %inserted_slice_52[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 scf.yield %inserted_slice_55 : tensor<1x1x4x16xf32>
# CHECK-NEXT:               } {"./s"}
# CHECK-NEXT:               scf.yield %11 : tensor<1x1x4x16xf32>
# CHECK-NEXT:             } {"./r"}
# CHECK-NEXT:             %inserted_slice_9 = tensor.insert_slice %10 into %arg10[0, 0, 0, %arg9] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:             scf.yield %inserted_slice_9 : tensor<1x1x4x16xf32>
# CHECK-NEXT:           } {"./f"}
# CHECK-NEXT:           %inserted_slice_6 = tensor.insert_slice %9 into %arg8[0, 0, %arg7, 0] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:           scf.yield %inserted_slice_6 : tensor<1x1x4x16xf32>
# CHECK-NEXT:         } {"./w"}
# CHECK-NEXT:         %inserted_slice_3 = tensor.insert_slice %6 into %arg6[0, %arg5, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> into tensor<1x4x4x16xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_3 : tensor<1x4x4x16xf32>
# CHECK-NEXT:       } {"./h"}
# CHECK-NEXT:       %inserted_slice = tensor.insert_slice %4 into %arg4[%arg3, 0, 0, 0] [1, 4, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> into tensor<1x4x4x16xf32>
# CHECK-NEXT:       scf.yield %inserted_slice : tensor<1x4x4x16xf32>
# CHECK-NEXT:     } {"./b"}
# CHECK-NEXT:     bufferization.materialize_in_destination %3 in restrict writable %arg2 : (tensor<1x4x4x16xf32>, memref<1x4x4x16xf32>) -> ()
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: // -----// IR Dump Before Tensor Lowering //----- //
# CHECK-NEXT: #map = affine_map<(d0) -> (d0 * 2)>
# CHECK-NEXT: #map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
# CHECK-NEXT: #map2 = affine_map<(d0)[s0] -> (d0 + s0)>
# CHECK-NEXT: #map3 = affine_map<(d0)[s0] -> (d0 * 2 + s0)>
# CHECK-NEXT: #map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
# CHECK-NEXT: #map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT: #map6 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @pad_conv2d_nhwc_mini(%arg0: tensor<1x8x8x3xf32> {llvm.noalias}, %arg1: tensor<5x5x3x16xf32> {llvm.noalias}, %arg2: memref<1x4x4x16xf32> {llvm.noalias}) {
# CHECK-NEXT:     %c6 = arith.constant 6 : index
# CHECK-NEXT:     %c5 = arith.constant 5 : index
# CHECK-NEXT:     %c16 = arith.constant 16 : index
# CHECK-NEXT:     %c4 = arith.constant 4 : index
# CHECK-NEXT:     %c3 = arith.constant 3 : index
# CHECK-NEXT:     %c8 = arith.constant 8 : index
# CHECK-NEXT:     %c2 = arith.constant 2 : index
# CHECK-NEXT:     %c1 = arith.constant 1 : index
# CHECK-NEXT:     %c0 = arith.constant 0 : index
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %0 = tensor.empty() : tensor<1x12x12x3xf32>
# CHECK-NEXT:     %1 = tensor.empty() : tensor<1x4x4x16xf32>
# CHECK-NEXT:     %2 = linalg.fill {__xtc_id_conv_0_} ins(%cst : f32) outs(%1 : tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
# CHECK-NEXT:     %3 = scf.for %arg3 = %c0 to %c1 step %c1 iter_args(%arg4 = %2) -> (tensor<1x4x4x16xf32>) {
# CHECK-NEXT:       %extracted_slice = tensor.extract_slice %0[%arg3, 0, 0, 0] [1, 11, 11, 3] [1, 1, 1, 1] : tensor<1x12x12x3xf32> to tensor<1x11x11x3xf32>
# CHECK-NEXT:       %extracted_slice_0 = tensor.extract_slice %arg4[%arg3, 0, 0, 0] [1, 4, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> to tensor<1x4x4x16xf32>
# CHECK-NEXT:       %4 = scf.for %arg5 = %c0 to %c4 step %c1 iter_args(%arg6 = %extracted_slice_0) -> (tensor<1x4x4x16xf32>) {
# CHECK-NEXT:         %5 = affine.apply #map(%arg5)
# CHECK-NEXT:         %extracted_slice_1 = tensor.extract_slice %extracted_slice[0, %5, 0, 0] [1, 5, 11, 3] [1, 1, 1, 1] : tensor<1x11x11x3xf32> to tensor<1x5x11x3xf32>
# CHECK-NEXT:         %extracted_slice_2 = tensor.extract_slice %arg6[0, %arg5, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> to tensor<1x1x4x16xf32>
# CHECK-NEXT:         %6 = scf.for %arg7 = %c0 to %c4 step %c4 iter_args(%arg8 = %extracted_slice_2) -> (tensor<1x1x4x16xf32>) {
# CHECK-NEXT:           %7 = affine.apply #map(%arg7)
# CHECK-NEXT:           %extracted_slice_4 = tensor.extract_slice %extracted_slice_1[0, 0, %7, 0] [1, 5, 11, 3] [1, 1, 1, 1] : tensor<1x5x11x3xf32> to tensor<1x5x11x3xf32>
# CHECK-NEXT:           %8 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%extracted_slice_4 : tensor<1x5x11x3xf32>) attrs =  {__xtc_id_pad_} {
# CHECK-NEXT:           ^bb0(%out: f32):
# CHECK-NEXT:             %10 = affine.apply #map2(%arg3)[%c0]
# CHECK-NEXT:             %11 = linalg.index 1 : index
# CHECK-NEXT:             %12 = affine.apply #map3(%arg5)[%11]
# CHECK-NEXT:             %13 = linalg.index 2 : index
# CHECK-NEXT:             %14 = affine.apply #map3(%arg7)[%13]
# CHECK-NEXT:             %15 = linalg.index 3 : index
# CHECK-NEXT:             %16 = arith.cmpi sge, %10, %c0 : index
# CHECK-NEXT:             %17 = arith.cmpi slt, %10, %c1 : index
# CHECK-NEXT:             %18 = arith.subi %12, %c2 : index
# CHECK-NEXT:             %19 = arith.cmpi sge, %18, %c0 : index
# CHECK-NEXT:             %20 = arith.cmpi slt, %18, %c8 : index
# CHECK-NEXT:             %21 = arith.subi %14, %c2 : index
# CHECK-NEXT:             %22 = arith.cmpi sge, %21, %c0 : index
# CHECK-NEXT:             %23 = arith.cmpi slt, %21, %c8 : index
# CHECK-NEXT:             %24 = arith.cmpi sge, %15, %c0 : index
# CHECK-NEXT:             %25 = arith.cmpi slt, %15, %c3 : index
# CHECK-NEXT:             %26 = arith.andi %16, %17 : i1
# CHECK-NEXT:             %27 = arith.andi %26, %19 : i1
# CHECK-NEXT:             %28 = arith.andi %27, %20 : i1
# CHECK-NEXT:             %29 = arith.andi %28, %22 : i1
# CHECK-NEXT:             %30 = arith.andi %29, %23 : i1
# CHECK-NEXT:             %31 = arith.andi %30, %24 : i1
# CHECK-NEXT:             %32 = arith.andi %31, %25 : i1
# CHECK-NEXT:             %33 = scf.if %32 -> (f32) {
# CHECK-NEXT:               %extracted = tensor.extract %arg0[%10, %18, %21, %15] : tensor<1x8x8x3xf32>
# CHECK-NEXT:               scf.yield %extracted : f32
# CHECK-NEXT:             } else {
# CHECK-NEXT:               scf.yield %cst : f32
# CHECK-NEXT:             }
# CHECK-NEXT:             linalg.yield %33 : f32
# CHECK-NEXT:           } -> tensor<1x5x11x3xf32>
# CHECK-NEXT:           %extracted_slice_5 = tensor.extract_slice %arg8[0, 0, %arg7, 0] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x4x16xf32>
# CHECK-NEXT:           %9 = scf.for %arg9 = %c0 to %c16 step %c16 iter_args(%arg10 = %extracted_slice_5) -> (tensor<1x1x4x16xf32>) {
# CHECK-NEXT:             %extracted_slice_7 = tensor.extract_slice %arg1[0, 0, 0, %arg9] [5, 5, 3, 16] [1, 1, 1, 1] : tensor<5x5x3x16xf32> to tensor<5x5x3x16xf32>
# CHECK-NEXT:             %extracted_slice_8 = tensor.extract_slice %arg10[0, 0, 0, %arg9] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x4x16xf32>
# CHECK-NEXT:             %10 = scf.for %arg11 = %c0 to %c5 step %c1 iter_args(%arg12 = %extracted_slice_8) -> (tensor<1x1x4x16xf32>) {
# CHECK-NEXT:               %extracted_slice_10 = tensor.extract_slice %8[0, %arg11, 0, 0] [1, 1, 11, 3] [1, 1, 1, 1] : tensor<1x5x11x3xf32> to tensor<1x1x11x3xf32>
# CHECK-NEXT:               %extracted_slice_11 = tensor.extract_slice %extracted_slice_7[%arg11, 0, 0, 0] [1, 5, 3, 16] [1, 1, 1, 1] : tensor<5x5x3x16xf32> to tensor<1x5x3x16xf32>
# CHECK-NEXT:               %11 = scf.for %arg13 = %c0 to %c5 step %c1 iter_args(%arg14 = %arg12) -> (tensor<1x1x4x16xf32>) {
# CHECK-NEXT:                 %extracted_slice_12 = tensor.extract_slice %extracted_slice_10[0, 0, %arg13, 0] [1, 1, 7, 3] [1, 1, 1, 1] : tensor<1x1x11x3xf32> to tensor<1x1x7x3xf32>
# CHECK-NEXT:                 %extracted_slice_13 = tensor.extract_slice %extracted_slice_11[0, %arg13, 0, 0] [1, 1, 3, 16] [1, 1, 1, 1] : tensor<1x5x3x16xf32> to tensor<1x1x3x16xf32>
# CHECK-NEXT:                 %extracted_slice_14 = tensor.extract_slice %extracted_slice_12[0, 0, 0, %c0] [1, 1, 7, 1] [1, 1, 1, 1] : tensor<1x1x7x3xf32> to tensor<1x1x7x1xf32>
# CHECK-NEXT:                 %extracted_slice_15 = tensor.extract_slice %extracted_slice_13[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x3x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %extracted_slice_16 = tensor.extract_slice %extracted_slice_14[0, 0, %c0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_17 = tensor.extract_slice %arg14[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %12 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_16, %extracted_slice_15 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_17 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %24 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %25 = arith.addf %out, %24 : f32
# CHECK-NEXT:                   linalg.yield %25 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_18 = tensor.insert_slice %12 into %arg14[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_19 = tensor.extract_slice %extracted_slice_14[0, 0, %c2, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_20 = tensor.extract_slice %inserted_slice_18[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %13 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_19, %extracted_slice_15 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_20 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %24 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %25 = arith.addf %out, %24 : f32
# CHECK-NEXT:                   linalg.yield %25 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_21 = tensor.insert_slice %13 into %inserted_slice_18[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_22 = tensor.extract_slice %extracted_slice_14[0, 0, %c4, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_23 = tensor.extract_slice %inserted_slice_21[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %14 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_22, %extracted_slice_15 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_23 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %24 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %25 = arith.addf %out, %24 : f32
# CHECK-NEXT:                   linalg.yield %25 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_24 = tensor.insert_slice %14 into %inserted_slice_21[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_25 = tensor.extract_slice %extracted_slice_14[0, 0, %c6, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_26 = tensor.extract_slice %inserted_slice_24[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %15 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_25, %extracted_slice_15 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_26 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %24 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %25 = arith.addf %out, %24 : f32
# CHECK-NEXT:                   linalg.yield %25 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_27 = tensor.insert_slice %15 into %inserted_slice_24[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_28 = tensor.extract_slice %extracted_slice_12[0, 0, 0, %c1] [1, 1, 7, 1] [1, 1, 1, 1] : tensor<1x1x7x3xf32> to tensor<1x1x7x1xf32>
# CHECK-NEXT:                 %extracted_slice_29 = tensor.extract_slice %extracted_slice_13[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x3x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %extracted_slice_30 = tensor.extract_slice %extracted_slice_28[0, 0, %c0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_31 = tensor.extract_slice %inserted_slice_27[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %16 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_30, %extracted_slice_29 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_31 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %24 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %25 = arith.addf %out, %24 : f32
# CHECK-NEXT:                   linalg.yield %25 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_32 = tensor.insert_slice %16 into %inserted_slice_27[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_33 = tensor.extract_slice %extracted_slice_28[0, 0, %c2, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_34 = tensor.extract_slice %inserted_slice_32[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %17 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_33, %extracted_slice_29 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_34 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %24 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %25 = arith.addf %out, %24 : f32
# CHECK-NEXT:                   linalg.yield %25 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_35 = tensor.insert_slice %17 into %inserted_slice_32[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_36 = tensor.extract_slice %extracted_slice_28[0, 0, %c4, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_37 = tensor.extract_slice %inserted_slice_35[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %18 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_36, %extracted_slice_29 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_37 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %24 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %25 = arith.addf %out, %24 : f32
# CHECK-NEXT:                   linalg.yield %25 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_38 = tensor.insert_slice %18 into %inserted_slice_35[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_39 = tensor.extract_slice %extracted_slice_28[0, 0, %c6, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_40 = tensor.extract_slice %inserted_slice_38[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %19 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_39, %extracted_slice_29 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_40 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %24 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %25 = arith.addf %out, %24 : f32
# CHECK-NEXT:                   linalg.yield %25 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_41 = tensor.insert_slice %19 into %inserted_slice_38[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_42 = tensor.extract_slice %extracted_slice_12[0, 0, 0, %c2] [1, 1, 7, 1] [1, 1, 1, 1] : tensor<1x1x7x3xf32> to tensor<1x1x7x1xf32>
# CHECK-NEXT:                 %extracted_slice_43 = tensor.extract_slice %extracted_slice_13[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x3x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %extracted_slice_44 = tensor.extract_slice %extracted_slice_42[0, 0, %c0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_45 = tensor.extract_slice %inserted_slice_41[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %20 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_44, %extracted_slice_43 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_45 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %24 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %25 = arith.addf %out, %24 : f32
# CHECK-NEXT:                   linalg.yield %25 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_46 = tensor.insert_slice %20 into %inserted_slice_41[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_47 = tensor.extract_slice %extracted_slice_42[0, 0, %c2, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_48 = tensor.extract_slice %inserted_slice_46[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %21 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_47, %extracted_slice_43 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_48 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %24 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %25 = arith.addf %out, %24 : f32
# CHECK-NEXT:                   linalg.yield %25 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_49 = tensor.insert_slice %21 into %inserted_slice_46[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_50 = tensor.extract_slice %extracted_slice_42[0, 0, %c4, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_51 = tensor.extract_slice %inserted_slice_49[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %22 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_50, %extracted_slice_43 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_51 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %24 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %25 = arith.addf %out, %24 : f32
# CHECK-NEXT:                   linalg.yield %25 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_52 = tensor.insert_slice %22 into %inserted_slice_49[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_53 = tensor.extract_slice %extracted_slice_42[0, 0, %c6, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_54 = tensor.extract_slice %inserted_slice_52[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %23 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_53, %extracted_slice_43 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_54 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %24 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %25 = arith.addf %out, %24 : f32
# CHECK-NEXT:                   linalg.yield %25 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_55 = tensor.insert_slice %23 into %inserted_slice_52[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 scf.yield %inserted_slice_55 : tensor<1x1x4x16xf32>
# CHECK-NEXT:               } {"./s"}
# CHECK-NEXT:               scf.yield %11 : tensor<1x1x4x16xf32>
# CHECK-NEXT:             } {"./r"}
# CHECK-NEXT:             %inserted_slice_9 = tensor.insert_slice %10 into %arg10[0, 0, 0, %arg9] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:             scf.yield %inserted_slice_9 : tensor<1x1x4x16xf32>
# CHECK-NEXT:           } {"./f"}
# CHECK-NEXT:           %inserted_slice_6 = tensor.insert_slice %9 into %arg8[0, 0, %arg7, 0] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:           scf.yield %inserted_slice_6 : tensor<1x1x4x16xf32>
# CHECK-NEXT:         } {"./w"}
# CHECK-NEXT:         %inserted_slice_3 = tensor.insert_slice %6 into %arg6[0, %arg5, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> into tensor<1x4x4x16xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_3 : tensor<1x4x4x16xf32>
# CHECK-NEXT:       } {"./h"}
# CHECK-NEXT:       %inserted_slice = tensor.insert_slice %4 into %arg4[%arg3, 0, 0, 0] [1, 4, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> into tensor<1x4x4x16xf32>
# CHECK-NEXT:       scf.yield %inserted_slice : tensor<1x4x4x16xf32>
# CHECK-NEXT:     } {"./b"}
# CHECK-NEXT:     bufferization.materialize_in_destination %3 in restrict writable %arg2 : (tensor<1x4x4x16xf32>, memref<1x4x4x16xf32>) -> ()
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: // -----// IR Dump After Tensor Lowering //----- //
# CHECK-NEXT: #map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
# CHECK-NEXT: #map1 = affine_map<(d0)[s0] -> (d0 * 2 + s0)>
# CHECK-NEXT: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
# CHECK-NEXT: #map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT: #map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @pad_conv2d_nhwc_mini(%arg0: memref<1x8x8x3xf32> {llvm.noalias}, %arg1: memref<5x5x3x16xf32> {llvm.noalias}, %arg2: memref<1x4x4x16xf32> {llvm.noalias}) {
# CHECK-NEXT:     %c5 = arith.constant 5 : index
# CHECK-NEXT:     %c4 = arith.constant 4 : index
# CHECK-NEXT:     %c3 = arith.constant 3 : index
# CHECK-NEXT:     %c8 = arith.constant 8 : index
# CHECK-NEXT:     %c2 = arith.constant 2 : index
# CHECK-NEXT:     %c1 = arith.constant 1 : index
# CHECK-NEXT:     %c0 = arith.constant 0 : index
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %alloca = memref.alloca() {alignment = 256 : i64} : memref<1x5x11x3xf32>
# CHECK-NEXT:     linalg.fill {__xtc_id_conv_0_} ins(%cst : f32) outs(%arg2 : memref<1x4x4x16xf32>)
# CHECK-NEXT:     %0 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %arg2) -> (memref<1x4x4x16xf32>) {
# CHECK-NEXT:       %subview = memref.subview %arg4[0, %arg3, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : memref<1x4x4x16xf32> to memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:       linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca : memref<1x5x11x3xf32>) attrs =  {__xtc_id_pad_} {
# CHECK-NEXT:       ^bb0(%out: f32):
# CHECK-NEXT:         %2 = linalg.index 1 : index
# CHECK-NEXT:         %3 = affine.apply #map1(%arg3)[%2]
# CHECK-NEXT:         %4 = linalg.index 2 : index
# CHECK-NEXT:         %5 = linalg.index 3 : index
# CHECK-NEXT:         %6 = arith.subi %3, %c2 : index
# CHECK-NEXT:         %7 = arith.cmpi sge, %6, %c0 : index
# CHECK-NEXT:         %8 = arith.cmpi slt, %6, %c8 : index
# CHECK-NEXT:         %9 = arith.subi %4, %c2 : index
# CHECK-NEXT:         %10 = arith.cmpi sge, %9, %c0 : index
# CHECK-NEXT:         %11 = arith.cmpi slt, %9, %c8 : index
# CHECK-NEXT:         %12 = arith.cmpi sge, %5, %c0 : index
# CHECK-NEXT:         %13 = arith.cmpi slt, %5, %c3 : index
# CHECK-NEXT:         %14 = arith.andi %7, %8 : i1
# CHECK-NEXT:         %15 = arith.andi %14, %10 : i1
# CHECK-NEXT:         %16 = arith.andi %15, %11 : i1
# CHECK-NEXT:         %17 = arith.andi %16, %12 : i1
# CHECK-NEXT:         %18 = arith.andi %17, %13 : i1
# CHECK-NEXT:         %19 = scf.if %18 -> (f32) {
# CHECK-NEXT:           %20 = memref.load %arg0[%c0, %6, %9, %5] : memref<1x8x8x3xf32>
# CHECK-NEXT:           scf.yield %20 : f32
# CHECK-NEXT:         } else {
# CHECK-NEXT:           scf.yield %cst : f32
# CHECK-NEXT:         }
# CHECK-NEXT:         linalg.yield %19 : f32
# CHECK-NEXT:       }
# CHECK-NEXT:       %1 = scf.for %arg5 = %c0 to %c5 step %c1 iter_args(%arg6 = %subview) -> (memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:         %subview_1 = memref.subview %alloca[0, %arg5, 0, 0] [1, 1, 11, 3] [1, 1, 1, 1] : memref<1x5x11x3xf32> to memref<1x1x11x3xf32, strided<[165, 33, 3, 1], offset: ?>>
# CHECK-NEXT:         %subview_2 = memref.subview %arg1[%arg5, 0, 0, 0] [1, 5, 3, 16] [1, 1, 1, 1] : memref<5x5x3x16xf32> to memref<1x5x3x16xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:         %2 = scf.for %arg7 = %c0 to %c5 step %c1 iter_args(%arg8 = %arg6) -> (memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:           %subview_3 = memref.subview %subview_1[0, 0, %arg7, 0] [1, 1, 7, 3] [1, 1, 1, 1] : memref<1x1x11x3xf32, strided<[165, 33, 3, 1], offset: ?>> to memref<1x1x7x3xf32, strided<[165, 33, 3, 1], offset: ?>>
# CHECK-NEXT:           %subview_4 = memref.subview %subview_2[0, %arg7, 0, 0] [1, 1, 3, 16] [1, 1, 1, 1] : memref<1x5x3x16xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<1x1x3x16xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:           %subview_5 = memref.subview %subview_3[0, 0, 0, 0] [1, 1, 7, 1] [1, 1, 1, 1] : memref<1x1x7x3xf32, strided<[165, 33, 3, 1], offset: ?>> to memref<1x1x7x1xf32, strided<[165, 33, 3, 1], offset: ?>>
# CHECK-NEXT:           %subview_6 = memref.subview %subview_4[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x3x16xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:           %subview_7 = memref.subview %subview_5[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[165, 33, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[165, 33, 3, 1], offset: ?>>
# CHECK-NEXT:           %subview_8 = memref.subview %arg8[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_7, %subview_6 : memref<1x1x1x1xf32, strided<[165, 33, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>>) outs(%subview_8 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:           ^bb0(%in: f32, %in_47: f32, %out: f32):
# CHECK-NEXT:             %3 = arith.mulf %in, %in_47 : f32
# CHECK-NEXT:             %4 = arith.addf %out, %3 : f32
# CHECK-NEXT:             linalg.yield %4 : f32
# CHECK-NEXT:           }
# CHECK-NEXT:           %subview_9 = memref.subview %arg8[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           memref.copy %subview_8, %subview_9 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           %subview_10 = memref.subview %subview_5[0, 0, 2, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[165, 33, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[165, 33, 3, 1], offset: ?>>
# CHECK-NEXT:           %subview_11 = memref.subview %arg8[0, 0, 1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_10, %subview_6 : memref<1x1x1x1xf32, strided<[165, 33, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>>) outs(%subview_11 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:           ^bb0(%in: f32, %in_47: f32, %out: f32):
# CHECK-NEXT:             %3 = arith.mulf %in, %in_47 : f32
# CHECK-NEXT:             %4 = arith.addf %out, %3 : f32
# CHECK-NEXT:             linalg.yield %4 : f32
# CHECK-NEXT:           }
# CHECK-NEXT:           %subview_12 = memref.subview %arg8[0, 0, 1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           memref.copy %subview_11, %subview_12 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           %subview_13 = memref.subview %subview_5[0, 0, 4, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[165, 33, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[165, 33, 3, 1], offset: ?>>
# CHECK-NEXT:           %subview_14 = memref.subview %arg8[0, 0, 2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_13, %subview_6 : memref<1x1x1x1xf32, strided<[165, 33, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>>) outs(%subview_14 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:           ^bb0(%in: f32, %in_47: f32, %out: f32):
# CHECK-NEXT:             %3 = arith.mulf %in, %in_47 : f32
# CHECK-NEXT:             %4 = arith.addf %out, %3 : f32
# CHECK-NEXT:             linalg.yield %4 : f32
# CHECK-NEXT:           }
# CHECK-NEXT:           %subview_15 = memref.subview %arg8[0, 0, 2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           memref.copy %subview_14, %subview_15 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           %subview_16 = memref.subview %subview_5[0, 0, 6, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[165, 33, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[165, 33, 3, 1], offset: ?>>
# CHECK-NEXT:           %subview_17 = memref.subview %arg8[0, 0, 3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_16, %subview_6 : memref<1x1x1x1xf32, strided<[165, 33, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>>) outs(%subview_17 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:           ^bb0(%in: f32, %in_47: f32, %out: f32):
# CHECK-NEXT:             %3 = arith.mulf %in, %in_47 : f32
# CHECK-NEXT:             %4 = arith.addf %out, %3 : f32
# CHECK-NEXT:             linalg.yield %4 : f32
# CHECK-NEXT:           }
# CHECK-NEXT:           %subview_18 = memref.subview %arg8[0, 0, 3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           memref.copy %subview_17, %subview_18 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           %subview_19 = memref.subview %subview_3[0, 0, 0, 1] [1, 1, 7, 1] [1, 1, 1, 1] : memref<1x1x7x3xf32, strided<[165, 33, 3, 1], offset: ?>> to memref<1x1x7x1xf32, strided<[165, 33, 3, 1], offset: ?>>
# CHECK-NEXT:           %subview_20 = memref.subview %subview_4[0, 0, 1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x3x16xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:           %subview_21 = memref.subview %subview_19[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[165, 33, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[165, 33, 3, 1], offset: ?>>
# CHECK-NEXT:           %subview_22 = memref.subview %arg8[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_21, %subview_20 : memref<1x1x1x1xf32, strided<[165, 33, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>>) outs(%subview_22 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:           ^bb0(%in: f32, %in_47: f32, %out: f32):
# CHECK-NEXT:             %3 = arith.mulf %in, %in_47 : f32
# CHECK-NEXT:             %4 = arith.addf %out, %3 : f32
# CHECK-NEXT:             linalg.yield %4 : f32
# CHECK-NEXT:           }
# CHECK-NEXT:           %subview_23 = memref.subview %arg8[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           memref.copy %subview_22, %subview_23 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           %subview_24 = memref.subview %subview_19[0, 0, 2, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[165, 33, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[165, 33, 3, 1], offset: ?>>
# CHECK-NEXT:           %subview_25 = memref.subview %arg8[0, 0, 1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_24, %subview_20 : memref<1x1x1x1xf32, strided<[165, 33, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>>) outs(%subview_25 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:           ^bb0(%in: f32, %in_47: f32, %out: f32):
# CHECK-NEXT:             %3 = arith.mulf %in, %in_47 : f32
# CHECK-NEXT:             %4 = arith.addf %out, %3 : f32
# CHECK-NEXT:             linalg.yield %4 : f32
# CHECK-NEXT:           }
# CHECK-NEXT:           %subview_26 = memref.subview %arg8[0, 0, 1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           memref.copy %subview_25, %subview_26 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           %subview_27 = memref.subview %subview_19[0, 0, 4, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[165, 33, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[165, 33, 3, 1], offset: ?>>
# CHECK-NEXT:           %subview_28 = memref.subview %arg8[0, 0, 2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_27, %subview_20 : memref<1x1x1x1xf32, strided<[165, 33, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>>) outs(%subview_28 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:           ^bb0(%in: f32, %in_47: f32, %out: f32):
# CHECK-NEXT:             %3 = arith.mulf %in, %in_47 : f32
# CHECK-NEXT:             %4 = arith.addf %out, %3 : f32
# CHECK-NEXT:             linalg.yield %4 : f32
# CHECK-NEXT:           }
# CHECK-NEXT:           %subview_29 = memref.subview %arg8[0, 0, 2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           memref.copy %subview_28, %subview_29 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           %subview_30 = memref.subview %subview_19[0, 0, 6, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[165, 33, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[165, 33, 3, 1], offset: ?>>
# CHECK-NEXT:           %subview_31 = memref.subview %arg8[0, 0, 3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_30, %subview_20 : memref<1x1x1x1xf32, strided<[165, 33, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>>) outs(%subview_31 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:           ^bb0(%in: f32, %in_47: f32, %out: f32):
# CHECK-NEXT:             %3 = arith.mulf %in, %in_47 : f32
# CHECK-NEXT:             %4 = arith.addf %out, %3 : f32
# CHECK-NEXT:             linalg.yield %4 : f32
# CHECK-NEXT:           }
# CHECK-NEXT:           %subview_32 = memref.subview %arg8[0, 0, 3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           memref.copy %subview_31, %subview_32 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           %subview_33 = memref.subview %subview_3[0, 0, 0, 2] [1, 1, 7, 1] [1, 1, 1, 1] : memref<1x1x7x3xf32, strided<[165, 33, 3, 1], offset: ?>> to memref<1x1x7x1xf32, strided<[165, 33, 3, 1], offset: ?>>
# CHECK-NEXT:           %subview_34 = memref.subview %subview_4[0, 0, 2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x3x16xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:           %subview_35 = memref.subview %subview_33[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[165, 33, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[165, 33, 3, 1], offset: ?>>
# CHECK-NEXT:           %subview_36 = memref.subview %arg8[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_35, %subview_34 : memref<1x1x1x1xf32, strided<[165, 33, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>>) outs(%subview_36 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:           ^bb0(%in: f32, %in_47: f32, %out: f32):
# CHECK-NEXT:             %3 = arith.mulf %in, %in_47 : f32
# CHECK-NEXT:             %4 = arith.addf %out, %3 : f32
# CHECK-NEXT:             linalg.yield %4 : f32
# CHECK-NEXT:           }
# CHECK-NEXT:           %subview_37 = memref.subview %arg8[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           memref.copy %subview_36, %subview_37 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           %subview_38 = memref.subview %subview_33[0, 0, 2, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[165, 33, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[165, 33, 3, 1], offset: ?>>
# CHECK-NEXT:           %subview_39 = memref.subview %arg8[0, 0, 1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_38, %subview_34 : memref<1x1x1x1xf32, strided<[165, 33, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>>) outs(%subview_39 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:           ^bb0(%in: f32, %in_47: f32, %out: f32):
# CHECK-NEXT:             %3 = arith.mulf %in, %in_47 : f32
# CHECK-NEXT:             %4 = arith.addf %out, %3 : f32
# CHECK-NEXT:             linalg.yield %4 : f32
# CHECK-NEXT:           }
# CHECK-NEXT:           %subview_40 = memref.subview %arg8[0, 0, 1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           memref.copy %subview_39, %subview_40 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           %subview_41 = memref.subview %subview_33[0, 0, 4, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[165, 33, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[165, 33, 3, 1], offset: ?>>
# CHECK-NEXT:           %subview_42 = memref.subview %arg8[0, 0, 2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_41, %subview_34 : memref<1x1x1x1xf32, strided<[165, 33, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>>) outs(%subview_42 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:           ^bb0(%in: f32, %in_47: f32, %out: f32):
# CHECK-NEXT:             %3 = arith.mulf %in, %in_47 : f32
# CHECK-NEXT:             %4 = arith.addf %out, %3 : f32
# CHECK-NEXT:             linalg.yield %4 : f32
# CHECK-NEXT:           }
# CHECK-NEXT:           %subview_43 = memref.subview %arg8[0, 0, 2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           memref.copy %subview_42, %subview_43 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           %subview_44 = memref.subview %subview_33[0, 0, 6, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[165, 33, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[165, 33, 3, 1], offset: ?>>
# CHECK-NEXT:           %subview_45 = memref.subview %arg8[0, 0, 3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_44, %subview_34 : memref<1x1x1x1xf32, strided<[165, 33, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>>) outs(%subview_45 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:           ^bb0(%in: f32, %in_47: f32, %out: f32):
# CHECK-NEXT:             %3 = arith.mulf %in, %in_47 : f32
# CHECK-NEXT:             %4 = arith.addf %out, %3 : f32
# CHECK-NEXT:             linalg.yield %4 : f32
# CHECK-NEXT:           }
# CHECK-NEXT:           %subview_46 = memref.subview %arg8[0, 0, 3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           memref.copy %subview_45, %subview_46 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           scf.yield %arg8 : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:         } {"./s"}
# CHECK-NEXT:         scf.yield %2 : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:       } {"./r"}
# CHECK-NEXT:       %subview_0 = memref.subview %arg4[0, %arg3, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : memref<1x4x4x16xf32> to memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:       memref.copy %1, %subview_0 : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:       scf.yield %arg4 : memref<1x4x4x16xf32>
# CHECK-NEXT:     } {"./h"}
# CHECK-NEXT:     memref.copy %0, %arg2 : memref<1x4x4x16xf32> to memref<1x4x4x16xf32>
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: graph:
# CHECK-NEXT:   name: pad_conv2d_nhwc_mini
# CHECK-NEXT:   inputs:
# CHECK-NEXT:   - %0 : 1x8x8x3xfloat32
# CHECK-NEXT:   - %1 : 5x5x3x16xfloat32
# CHECK-NEXT:   outputs:
# CHECK-NEXT:   - %3 : 1x4x4x16xfloat32
# CHECK-NEXT:   nodes:
# CHECK-NEXT:   - %2: pad2d(%0, padding={1: (2, 2), 2: (2, 2)}, constant_value=0) {name = 'pad'} : [1x8x8x3xfloat32] -> [1x12x12x3xfloat32]
# CHECK-NEXT:   - %3: conv2d(%2, %1, stride=(2, 2)) {name = 'conv'} : [1x12x12x3xfloat32, 5x5x3x16xfloat32] -> [1x4x4x16xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT: CODE: 0
