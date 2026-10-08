# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: mlir-target=nvgpu

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend
from xtc.runtimes.accelerator.gpu import GPUDevice
from xtc.artifacts import get_operation

gpu = GPUDevice()
op = get_operation("conv2d", "ResNet18_01")
N, H, W, F, R, S, C = [op["dims"][k] for k in ["n", "h", "w", "f", "r", "s", "c"]]
SH, SW = [op["params"][k] for k in ["SH", "SW"]]
dtype = "float32"

a = O.tensor((N, H + R - 1, W + S - 1, C), dtype, name="I", device=gpu)
b = O.tensor((R, S, C, F), dtype, name="W",device=gpu)

with O.graph(name="pad_conv2d_nhwc_mini") as gb:
    p = O.pad2d(a, padding=2, axes=(1, 2), name="pad", device=gpu)
    c = O.conv2d(p, b, stride=(SH, SW), name="conv", device=gpu)
    O.relu(c, name="relu", device=gpu)

graph = gb.graph
print(graph)

impl = Backend(graph, use_tensor_dialect=True)

sch = impl.get_scheduler(default_node="conv")
sch.interchange(["b", "h", "w", "r", "s", "c", "f"])
sch.gpu_block(["b", "h", "w"])
sch.fuse_producer_at("w", 0)
sch.fuse_consumer_at("w")
# sch.vectorize(["f"])
sched = sch.schedule()

comp = impl.get_compiler(
    target=gpu,
    shared_lib=True,
    dump_file="pad_conv2d_relu_gpu_mlir_tensor_fused",
    print_source_ir=True,
    print_transformed_ir=True,
    print_bufferization_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
# CHECK-NEXT:  #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT:  #map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT:  #map4 = affine_map<(d0, d1, d2, d3) -> ()>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @pad_conv2d_nhwc_mini(%arg0: tensor<1x230x230x3xf32> {llvm.noalias, memref.on_device}, %arg1: tensor<7x7x3x64xf32> {llvm.noalias, memref.on_device}, %arg2: memref<1x114x114x64xf32> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %0 = tensor.empty() : tensor<1x234x234x3xf32>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%0 : tensor<1x234x234x3xf32>) attrs =  {__xtc_id_pad_} {
# CHECK-NEXT:      ^bb0(%out: f32):
# CHECK-NEXT:        %7 = linalg.index 0 : index
# CHECK-NEXT:        %8 = linalg.index 1 : index
# CHECK-NEXT:        %9 = linalg.index 2 : index
# CHECK-NEXT:        %10 = linalg.index 3 : index
# CHECK-NEXT:        %c0 = arith.constant 0 : index
# CHECK-NEXT:        %c0_2 = arith.constant 0 : index
# CHECK-NEXT:        %11 = arith.subi %7, %c0_2 : index
# CHECK-NEXT:        %c1 = arith.constant 1 : index
# CHECK-NEXT:        %12 = arith.cmpi sge, %11, %c0 : index
# CHECK-NEXT:        %13 = arith.cmpi slt, %11, %c1 : index
# CHECK-NEXT:        %c2 = arith.constant 2 : index
# CHECK-NEXT:        %14 = arith.subi %8, %c2 : index
# CHECK-NEXT:        %c230 = arith.constant 230 : index
# CHECK-NEXT:        %15 = arith.cmpi sge, %14, %c0 : index
# CHECK-NEXT:        %16 = arith.cmpi slt, %14, %c230 : index
# CHECK-NEXT:        %c2_3 = arith.constant 2 : index
# CHECK-NEXT:        %17 = arith.subi %9, %c2_3 : index
# CHECK-NEXT:        %c230_4 = arith.constant 230 : index
# CHECK-NEXT:        %18 = arith.cmpi sge, %17, %c0 : index
# CHECK-NEXT:        %19 = arith.cmpi slt, %17, %c230_4 : index
# CHECK-NEXT:        %c0_5 = arith.constant 0 : index
# CHECK-NEXT:        %20 = arith.subi %10, %c0_5 : index
# CHECK-NEXT:        %c3 = arith.constant 3 : index
# CHECK-NEXT:        %21 = arith.cmpi sge, %20, %c0 : index
# CHECK-NEXT:        %22 = arith.cmpi slt, %20, %c3 : index
# CHECK-NEXT:        %23 = arith.andi %12, %13 : i1
# CHECK-NEXT:        %24 = arith.andi %23, %15 : i1
# CHECK-NEXT:        %25 = arith.andi %24, %16 : i1
# CHECK-NEXT:        %26 = arith.andi %25, %18 : i1
# CHECK-NEXT:        %27 = arith.andi %26, %19 : i1
# CHECK-NEXT:        %28 = arith.andi %27, %21 : i1
# CHECK-NEXT:        %29 = arith.andi %28, %22 : i1
# CHECK-NEXT:        %30 = scf.if %29 -> (f32) {
# CHECK-NEXT:          %extracted = tensor.extract %arg0[%11, %14, %17, %20] : tensor<1x230x230x3xf32>
# CHECK-NEXT:          scf.yield %extracted : f32
# CHECK-NEXT:        } else {
# CHECK-NEXT:          scf.yield %cst : f32
# CHECK-NEXT:        }
# CHECK-NEXT:        linalg.yield %30 : f32
# CHECK-NEXT:      } -> tensor<1x234x234x3xf32>
# CHECK-NEXT:      %2 = tensor.empty() : tensor<1x114x114x64xf32>
# CHECK-NEXT:      %cst_0 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %3 = linalg.fill {__xtc_id_conv_0_} ins(%cst_0 : f32) outs(%2 : tensor<1x114x114x64xf32>) -> tensor<1x114x114x64xf32>
# CHECK-NEXT:      %4 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%1, %arg1 : tensor<1x234x234x3xf32>, tensor<7x7x3x64xf32>) outs(%3 : tensor<1x114x114x64xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_2: f32, %out: f32):
# CHECK-NEXT:        %7 = arith.mulf %in, %in_2 fastmath<fast> : f32
# CHECK-NEXT:        %8 = arith.addf %out, %7 fastmath<fast> : f32
# CHECK-NEXT:        linalg.yield %8 : f32
# CHECK-NEXT:      } -> tensor<1x114x114x64xf32>
# CHECK-NEXT:      %5 = tensor.empty() : tensor<1x114x114x64xf32>
# CHECK-NEXT:      %cst_1 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %6 = linalg.generic {indexing_maps = [#map, #map4, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4, %cst_1 : tensor<1x114x114x64xf32>, f32) outs(%5 : tensor<1x114x114x64xf32>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_2: f32, %out: f32):
# CHECK-NEXT:        %7 = arith.maximumf %in, %in_2 : f32
# CHECK-NEXT:        linalg.yield %7 : f32
# CHECK-NEXT:      } -> tensor<1x114x114x64xf32>
# CHECK-NEXT:      bufferization.materialize_in_destination %6 in restrict writable %arg2 : (tensor<1x114x114x64xf32>, memref<1x114x114x64xf32>) -> ()
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_post_bufferize(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match attributes {"./w", mapping = [#gpu.block<x>, #gpu.block<y>, #gpu.block<z>]} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %1 = transform.gpu.map_forall_to_blocks %0 generate_gpu_launch : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match attributes {__xtc_id_conv_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_op, %forall_op = transform.structured.tile_using_forall %0 tile_sizes [1, 1, 1, 0, 0, 0, 0](mapping = [#gpu.block<x>, #gpu.block<y>, #gpu.block<z>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op "./w" : !transform.any_op
# CHECK-NEXT:      %1 = transform.structured.match attributes {__xtc_id_pad_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %1 into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      %2 = transform.structured.match attributes {__xtc_id_conv_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %2 tile_sizes [0, 0, 0, 0, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops "./r" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 0, 0, 0, 0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_1 "./s" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %tiled_linalg_op_0 tile_sizes [0, 0, 0, 0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_3 "./c" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [0, 0, 0, 1, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "./f" : !transform.any_op
# CHECK-NEXT:      %3 = transform.structured.match attributes {__xtc_id_relu_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_consumer, %new_loops = transform.xtc.fuse_consumer %3 into %new_containing_op {in_place} : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %new_loops "./w" : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
# CHECK-NEXT:  #map1 = affine_map<(d0) -> (d0 * 2)>
# CHECK-NEXT:  #map2 = affine_map<(d0)[s0] -> (d0 + s0)>
# CHECK-NEXT:  #map3 = affine_map<(d0)[s0] -> (d0 * 2 + s0)>
# CHECK-NEXT:  #map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
# CHECK-NEXT:  #map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT:  #map6 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT:  #map7 = affine_map<(d0, d1, d2, d3) -> ()>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @pad_conv2d_nhwc_mini(%arg0: tensor<1x230x230x3xf32> {llvm.noalias, memref.on_device}, %arg1: tensor<7x7x3x64xf32> {llvm.noalias, memref.on_device}, %arg2: memref<1x114x114x64xf32> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %0 = tensor.empty() : tensor<1x234x234x3xf32>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%0 : tensor<1x234x234x3xf32>) attrs =  {__xtc_id_pad_} {
# CHECK-NEXT:      ^bb0(%out: f32):
# CHECK-NEXT:        %7 = linalg.index 0 : index
# CHECK-NEXT:        %8 = linalg.index 1 : index
# CHECK-NEXT:        %9 = linalg.index 2 : index
# CHECK-NEXT:        %10 = linalg.index 3 : index
# CHECK-NEXT:        %c0 = arith.constant 0 : index
# CHECK-NEXT:        %c0_2 = arith.constant 0 : index
# CHECK-NEXT:        %11 = arith.subi %7, %c0_2 : index
# CHECK-NEXT:        %c1 = arith.constant 1 : index
# CHECK-NEXT:        %12 = arith.cmpi sge, %11, %c0 : index
# CHECK-NEXT:        %13 = arith.cmpi slt, %11, %c1 : index
# CHECK-NEXT:        %c2 = arith.constant 2 : index
# CHECK-NEXT:        %14 = arith.subi %8, %c2 : index
# CHECK-NEXT:        %c230 = arith.constant 230 : index
# CHECK-NEXT:        %15 = arith.cmpi sge, %14, %c0 : index
# CHECK-NEXT:        %16 = arith.cmpi slt, %14, %c230 : index
# CHECK-NEXT:        %c2_3 = arith.constant 2 : index
# CHECK-NEXT:        %17 = arith.subi %9, %c2_3 : index
# CHECK-NEXT:        %c230_4 = arith.constant 230 : index
# CHECK-NEXT:        %18 = arith.cmpi sge, %17, %c0 : index
# CHECK-NEXT:        %19 = arith.cmpi slt, %17, %c230_4 : index
# CHECK-NEXT:        %c0_5 = arith.constant 0 : index
# CHECK-NEXT:        %20 = arith.subi %10, %c0_5 : index
# CHECK-NEXT:        %c3 = arith.constant 3 : index
# CHECK-NEXT:        %21 = arith.cmpi sge, %20, %c0 : index
# CHECK-NEXT:        %22 = arith.cmpi slt, %20, %c3 : index
# CHECK-NEXT:        %23 = arith.andi %12, %13 : i1
# CHECK-NEXT:        %24 = arith.andi %23, %15 : i1
# CHECK-NEXT:        %25 = arith.andi %24, %16 : i1
# CHECK-NEXT:        %26 = arith.andi %25, %18 : i1
# CHECK-NEXT:        %27 = arith.andi %26, %19 : i1
# CHECK-NEXT:        %28 = arith.andi %27, %21 : i1
# CHECK-NEXT:        %29 = arith.andi %28, %22 : i1
# CHECK-NEXT:        %30 = scf.if %29 -> (f32) {
# CHECK-NEXT:          %extracted = tensor.extract %arg0[%11, %14, %17, %20] : tensor<1x230x230x3xf32>
# CHECK-NEXT:          scf.yield %extracted : f32
# CHECK-NEXT:        } else {
# CHECK-NEXT:          scf.yield %cst : f32
# CHECK-NEXT:        }
# CHECK-NEXT:        linalg.yield %30 : f32
# CHECK-NEXT:      } -> tensor<1x234x234x3xf32>
# CHECK-NEXT:      %2 = tensor.empty() : tensor<1x114x114x64xf32>
# CHECK-NEXT:      %cst_0 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %3 = linalg.fill {__xtc_id_conv_0_} ins(%cst_0 : f32) outs(%2 : tensor<1x114x114x64xf32>) -> tensor<1x114x114x64xf32>
# CHECK-NEXT:      %4 = tensor.empty() : tensor<1x114x114x64xf32>
# CHECK-NEXT:      %cst_1 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %5:2 = scf.forall (%arg3, %arg4, %arg5) in (1, 114, 114) shared_outs(%arg6 = %3, %arg7 = %4) -> (tensor<1x114x114x64xf32>, tensor<1x114x114x64xf32>) {
# CHECK-NEXT:        %7 = affine.apply #map1(%arg4)
# CHECK-NEXT:        %8 = affine.apply #map1(%arg5)
# CHECK-NEXT:        %9 = affine.apply #map1(%arg4)
# CHECK-NEXT:        %10 = affine.apply #map1(%arg5)
# CHECK-NEXT:        %extracted_slice = tensor.extract_slice %0[%arg3, %9, %10, 0] [1, 7, 7, 3] [1, 1, 1, 1] : tensor<1x234x234x3xf32> to tensor<1x7x7x3xf32>
# CHECK-NEXT:        %11 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%extracted_slice : tensor<1x7x7x3xf32>) attrs =  {__xtc_id_pad_} {
# CHECK-NEXT:        ^bb0(%out: f32):
# CHECK-NEXT:          %14 = linalg.index 0 : index
# CHECK-NEXT:          %15 = affine.apply #map2(%arg3)[%14]
# CHECK-NEXT:          %16 = linalg.index 1 : index
# CHECK-NEXT:          %17 = affine.apply #map3(%arg4)[%16]
# CHECK-NEXT:          %18 = linalg.index 2 : index
# CHECK-NEXT:          %19 = affine.apply #map3(%arg5)[%18]
# CHECK-NEXT:          %20 = linalg.index 3 : index
# CHECK-NEXT:          %c0_5 = arith.constant 0 : index
# CHECK-NEXT:          %c0_6 = arith.constant 0 : index
# CHECK-NEXT:          %21 = arith.subi %15, %c0_6 : index
# CHECK-NEXT:          %c1_7 = arith.constant 1 : index
# CHECK-NEXT:          %22 = arith.cmpi sge, %21, %c0_5 : index
# CHECK-NEXT:          %23 = arith.cmpi slt, %21, %c1_7 : index
# CHECK-NEXT:          %c2 = arith.constant 2 : index
# CHECK-NEXT:          %24 = arith.subi %17, %c2 : index
# CHECK-NEXT:          %c230 = arith.constant 230 : index
# CHECK-NEXT:          %25 = arith.cmpi sge, %24, %c0_5 : index
# CHECK-NEXT:          %26 = arith.cmpi slt, %24, %c230 : index
# CHECK-NEXT:          %c2_8 = arith.constant 2 : index
# CHECK-NEXT:          %27 = arith.subi %19, %c2_8 : index
# CHECK-NEXT:          %c230_9 = arith.constant 230 : index
# CHECK-NEXT:          %28 = arith.cmpi sge, %27, %c0_5 : index
# CHECK-NEXT:          %29 = arith.cmpi slt, %27, %c230_9 : index
# CHECK-NEXT:          %c0_10 = arith.constant 0 : index
# CHECK-NEXT:          %30 = arith.subi %20, %c0_10 : index
# CHECK-NEXT:          %c3 = arith.constant 3 : index
# CHECK-NEXT:          %31 = arith.cmpi sge, %30, %c0_5 : index
# CHECK-NEXT:          %32 = arith.cmpi slt, %30, %c3 : index
# CHECK-NEXT:          %33 = arith.andi %22, %23 : i1
# CHECK-NEXT:          %34 = arith.andi %33, %25 : i1
# CHECK-NEXT:          %35 = arith.andi %34, %26 : i1
# CHECK-NEXT:          %36 = arith.andi %35, %28 : i1
# CHECK-NEXT:          %37 = arith.andi %36, %29 : i1
# CHECK-NEXT:          %38 = arith.andi %37, %31 : i1
# CHECK-NEXT:          %39 = arith.andi %38, %32 : i1
# CHECK-NEXT:          %40 = scf.if %39 -> (f32) {
# CHECK-NEXT:            %extracted = tensor.extract %arg0[%21, %24, %27, %30] : tensor<1x230x230x3xf32>
# CHECK-NEXT:            scf.yield %extracted : f32
# CHECK-NEXT:          } else {
# CHECK-NEXT:            scf.yield %cst : f32
# CHECK-NEXT:          }
# CHECK-NEXT:          linalg.yield %40 : f32
# CHECK-NEXT:        } -> tensor<1x7x7x3xf32>
# CHECK-NEXT:        %extracted_slice_2 = tensor.extract_slice %arg1[0, 0, 0, 0] [7, 7, 3, 64] [1, 1, 1, 1] : tensor<7x7x3x64xf32> to tensor<7x7x3x64xf32>
# CHECK-NEXT:        %extracted_slice_3 = tensor.extract_slice %arg6[%arg3, %arg4, %arg5, 0] [1, 1, 1, 64] [1, 1, 1, 1] : tensor<1x114x114x64xf32> to tensor<1x1x1x64xf32>
# CHECK-NEXT:        %c0 = arith.constant 0 : index
# CHECK-NEXT:        %c7 = arith.constant 7 : index
# CHECK-NEXT:        %c1 = arith.constant 1 : index
# CHECK-NEXT:        %12 = scf.for %arg8 = %c0 to %c7 step %c1 iter_args(%arg9 = %extracted_slice_3) -> (tensor<1x1x1x64xf32>) {
# CHECK-NEXT:          %extracted_slice_5 = tensor.extract_slice %11[0, %arg8, 0, 0] [1, 1, 7, 3] [1, 1, 1, 1] : tensor<1x7x7x3xf32> to tensor<1x1x7x3xf32>
# CHECK-NEXT:          %extracted_slice_6 = tensor.extract_slice %extracted_slice_2[%arg8, 0, 0, 0] [1, 7, 3, 64] [1, 1, 1, 1] : tensor<7x7x3x64xf32> to tensor<1x7x3x64xf32>
# CHECK-NEXT:          %extracted_slice_7 = tensor.extract_slice %arg9[0, 0, 0, 0] [1, 1, 1, 64] [1, 1, 1, 1] : tensor<1x1x1x64xf32> to tensor<1x1x1x64xf32>
# CHECK-NEXT:          %c0_8 = arith.constant 0 : index
# CHECK-NEXT:          %c7_9 = arith.constant 7 : index
# CHECK-NEXT:          %c1_10 = arith.constant 1 : index
# CHECK-NEXT:          %14 = scf.for %arg10 = %c0_8 to %c7_9 step %c1_10 iter_args(%arg11 = %extracted_slice_7) -> (tensor<1x1x1x64xf32>) {
# CHECK-NEXT:            %extracted_slice_11 = tensor.extract_slice %extracted_slice_5[0, 0, %arg10, 0] [1, 1, 1, 3] [1, 1, 1, 1] : tensor<1x1x7x3xf32> to tensor<1x1x1x3xf32>
# CHECK-NEXT:            %extracted_slice_12 = tensor.extract_slice %extracted_slice_6[0, %arg10, 0, 0] [1, 1, 3, 64] [1, 1, 1, 1] : tensor<1x7x3x64xf32> to tensor<1x1x3x64xf32>
# CHECK-NEXT:            %extracted_slice_13 = tensor.extract_slice %arg11[0, 0, 0, 0] [1, 1, 1, 64] [1, 1, 1, 1] : tensor<1x1x1x64xf32> to tensor<1x1x1x64xf32>
# CHECK-NEXT:            %c0_14 = arith.constant 0 : index
# CHECK-NEXT:            %c3 = arith.constant 3 : index
# CHECK-NEXT:            %c1_15 = arith.constant 1 : index
# CHECK-NEXT:            %15 = scf.for %arg12 = %c0_14 to %c3 step %c1_15 iter_args(%arg13 = %extracted_slice_13) -> (tensor<1x1x1x64xf32>) {
# CHECK-NEXT:              %extracted_slice_17 = tensor.extract_slice %extracted_slice_11[0, 0, 0, %arg12] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x3xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:              %extracted_slice_18 = tensor.extract_slice %extracted_slice_12[0, 0, %arg12, 0] [1, 1, 1, 64] [1, 1, 1, 1] : tensor<1x1x3x64xf32> to tensor<1x1x1x64xf32>
# CHECK-NEXT:              %extracted_slice_19 = tensor.extract_slice %arg13[0, 0, 0, 0] [1, 1, 1, 64] [1, 1, 1, 1] : tensor<1x1x1x64xf32> to tensor<1x1x1x64xf32>
# CHECK-NEXT:              %c0_20 = arith.constant 0 : index
# CHECK-NEXT:              %c64 = arith.constant 64 : index
# CHECK-NEXT:              %c1_21 = arith.constant 1 : index
# CHECK-NEXT:              %16 = scf.for %arg14 = %c0_20 to %c64 step %c1_21 iter_args(%arg15 = %extracted_slice_19) -> (tensor<1x1x1x64xf32>) {
# CHECK-NEXT:                %extracted_slice_23 = tensor.extract_slice %extracted_slice_17[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                %extracted_slice_24 = tensor.extract_slice %extracted_slice_18[0, 0, 0, %arg14] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x64xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                %extracted_slice_25 = tensor.extract_slice %arg15[0, 0, 0, %arg14] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x64xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                %17 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_23, %extracted_slice_24 : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>) outs(%extracted_slice_25 : tensor<1x1x1x1xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                ^bb0(%in: f32, %in_27: f32, %out: f32):
# CHECK-NEXT:                  %18 = arith.mulf %in, %in_27 fastmath<fast> : f32
# CHECK-NEXT:                  %19 = arith.addf %out, %18 fastmath<fast> : f32
# CHECK-NEXT:                  linalg.yield %19 : f32
# CHECK-NEXT:                } -> tensor<1x1x1x1xf32>
# CHECK-NEXT:                %inserted_slice_26 = tensor.insert_slice %17 into %arg15[0, 0, 0, %arg14] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x64xf32>
# CHECK-NEXT:                scf.yield %inserted_slice_26 : tensor<1x1x1x64xf32>
# CHECK-NEXT:              } {"./f"}
# CHECK-NEXT:              %inserted_slice_22 = tensor.insert_slice %16 into %arg13[0, 0, 0, 0] [1, 1, 1, 64] [1, 1, 1, 1] : tensor<1x1x1x64xf32> into tensor<1x1x1x64xf32>
# CHECK-NEXT:              scf.yield %inserted_slice_22 : tensor<1x1x1x64xf32>
# CHECK-NEXT:            } {"./c"}
# CHECK-NEXT:            %inserted_slice_16 = tensor.insert_slice %15 into %arg11[0, 0, 0, 0] [1, 1, 1, 64] [1, 1, 1, 1] : tensor<1x1x1x64xf32> into tensor<1x1x1x64xf32>
# CHECK-NEXT:            scf.yield %inserted_slice_16 : tensor<1x1x1x64xf32>
# CHECK-NEXT:          } {"./s"}
# CHECK-NEXT:          %inserted_slice = tensor.insert_slice %14 into %arg9[0, 0, 0, 0] [1, 1, 1, 64] [1, 1, 1, 1] : tensor<1x1x1x64xf32> into tensor<1x1x1x64xf32>
# CHECK-NEXT:          scf.yield %inserted_slice : tensor<1x1x1x64xf32>
# CHECK-NEXT:        } {"./r"}
# CHECK-NEXT:        %extracted_slice_4 = tensor.extract_slice %arg6[%arg3, %arg4, %arg5, 0] [1, 1, 1, 64] [1, 1, 1, 1] : tensor<1x114x114x64xf32> to tensor<1x1x1x64xf32>
# CHECK-NEXT:        %13 = linalg.generic {indexing_maps = [#map, #map7, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%12, %cst_1 : tensor<1x1x1x64xf32>, f32) outs(%12 : tensor<1x1x1x64xf32>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:        ^bb0(%in: f32, %in_5: f32, %out: f32):
# CHECK-NEXT:          %14 = arith.maximumf %in, %in_5 : f32
# CHECK-NEXT:          linalg.yield %14 : f32
# CHECK-NEXT:        } -> tensor<1x1x1x64xf32>
# CHECK-NEXT:        scf.forall.in_parallel {
# CHECK-NEXT:          tensor.parallel_insert_slice %13 into %arg6[%arg3, %arg4, %arg5, 0] [1, 1, 1, 64] [1, 1, 1, 1] : tensor<1x1x1x64xf32> into tensor<1x114x114x64xf32>
# CHECK-NEXT:        }
# CHECK-NEXT:      } {"./w", mapping = [#gpu.block<x>, #gpu.block<y>, #gpu.block<z>]}
# CHECK-NEXT:      %6 = linalg.generic {indexing_maps = [#map, #map7, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5#0, %cst_1 : tensor<1x114x114x64xf32>, f32) outs(%4 : tensor<1x114x114x64xf32>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_2: f32, %out: f32):
# CHECK-NEXT:        %7 = arith.maximumf %in, %in_2 : f32
# CHECK-NEXT:        linalg.yield %7 : f32
# CHECK-NEXT:      } -> tensor<1x114x114x64xf32>
# CHECK-NEXT:      bufferization.materialize_in_destination %5#0 in restrict writable %arg2 : (tensor<1x114x114x64xf32>, memref<1x114x114x64xf32>) -> ()
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_post_bufferize(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match attributes {"./w", mapping = [#gpu.block<x>, #gpu.block<y>, #gpu.block<z>]} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %1 = transform.gpu.map_forall_to_blocks %0 generate_gpu_launch : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After Tensor Lowering //----- //
# CHECK-NEXT:  #map = affine_map<(d0) -> (d0 * 2)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
# CHECK-NEXT:  #map2 = affine_map<(d0)[s0] -> (d0 * 2 + s0)>
# CHECK-NEXT:  #map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
# CHECK-NEXT:  #map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT:  #map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT:  #map6 = affine_map<(d0, d1, d2, d3) -> ()>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @pad_conv2d_nhwc_mini(%arg0: memref<1x230x230x3xf32> {llvm.noalias, memref.on_device}, %arg1: memref<7x7x3x64xf32> {llvm.noalias, memref.on_device}, %arg2: memref<1x114x114x64xf32> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %c64 = arith.constant 64 : index
# CHECK-NEXT:      %c7 = arith.constant 7 : index
# CHECK-NEXT:      %c3 = arith.constant 3 : index
# CHECK-NEXT:      %c230 = arith.constant 230 : index
# CHECK-NEXT:      %c2 = arith.constant 2 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %alloc = memref.alloc() {alignment = 256 : i64} : memref<1x234x234x3xf32>
# CHECK-NEXT:      linalg.fill {__xtc_id_conv_0_} ins(%cst : f32) outs(%arg2 : memref<1x114x114x64xf32>)
# CHECK-NEXT:      %c1_0 = arith.constant 1 : index
# CHECK-NEXT:      %c1_1 = arith.constant 1 : index
# CHECK-NEXT:      %c114 = arith.constant 114 : index
# CHECK-NEXT:      %c114_2 = arith.constant 114 : index
# CHECK-NEXT:      gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c1_1, %arg10 = %c114, %arg11 = %c114_2) threads(%arg6, %arg7, %arg8) in (%arg12 = %c1_0, %arg13 = %c1_0, %arg14 = %c1_0) {
# CHECK-NEXT:        %c0_3 = arith.constant 0 : index
# CHECK-NEXT:        %block_id_x = gpu.block_id  x
# CHECK-NEXT:        %block_id_y = gpu.block_id  y
# CHECK-NEXT:        %block_id_z = gpu.block_id  z
# CHECK-NEXT:        %0 = affine.apply #map(%block_id_y)
# CHECK-NEXT:        %1 = affine.apply #map(%block_id_z)
# CHECK-NEXT:        %subview = memref.subview %alloc[0, %0, %1, 0] [1, 7, 7, 3] [1, 1, 1, 1] : memref<1x234x234x3xf32> to memref<1x7x7x3xf32, strided<[164268, 702, 3, 1], offset: ?>>
# CHECK-NEXT:        %alloca = memref.alloca() {alignment = 256 : i64} : memref<1x7x7x3xf32>
# CHECK-NEXT:        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca : memref<1x7x7x3xf32>) attrs =  {__xtc_id_pad_} {
# CHECK-NEXT:        ^bb0(%out: f32):
# CHECK-NEXT:          %3 = linalg.index 1 : index
# CHECK-NEXT:          %4 = affine.apply #map2(%block_id_y)[%3]
# CHECK-NEXT:          %5 = linalg.index 2 : index
# CHECK-NEXT:          %6 = affine.apply #map2(%block_id_z)[%5]
# CHECK-NEXT:          %7 = linalg.index 3 : index
# CHECK-NEXT:          %8 = arith.subi %4, %c2 : index
# CHECK-NEXT:          %9 = arith.cmpi sge, %8, %c0 : index
# CHECK-NEXT:          %10 = arith.cmpi slt, %8, %c230 : index
# CHECK-NEXT:          %11 = arith.subi %6, %c2 : index
# CHECK-NEXT:          %12 = arith.cmpi sge, %11, %c0 : index
# CHECK-NEXT:          %13 = arith.cmpi slt, %11, %c230 : index
# CHECK-NEXT:          %14 = arith.cmpi sge, %7, %c0 : index
# CHECK-NEXT:          %15 = arith.cmpi slt, %7, %c3 : index
# CHECK-NEXT:          %16 = arith.andi %9, %10 : i1
# CHECK-NEXT:          %17 = arith.andi %16, %12 : i1
# CHECK-NEXT:          %18 = arith.andi %17, %13 : i1
# CHECK-NEXT:          %19 = arith.andi %18, %14 : i1
# CHECK-NEXT:          %20 = arith.andi %19, %15 : i1
# CHECK-NEXT:          %21 = scf.if %20 -> (f32) {
# CHECK-NEXT:            %22 = memref.load %arg0[%c0, %8, %11, %7] : memref<1x230x230x3xf32>
# CHECK-NEXT:            scf.yield %22 : f32
# CHECK-NEXT:          } else {
# CHECK-NEXT:            scf.yield %cst : f32
# CHECK-NEXT:          }
# CHECK-NEXT:          linalg.yield %21 : f32
# CHECK-NEXT:        }
# CHECK-NEXT:        %subview_4 = memref.subview %arg2[0, %block_id_y, %block_id_z, 0] [1, 1, 1, 64] [1, 1, 1, 1] : memref<1x114x114x64xf32> to memref<1x1x1x64xf32, strided<[831744, 7296, 64, 1], offset: ?>>
# CHECK-NEXT:        %2 = scf.for %arg15 = %c0 to %c7 step %c1 iter_args(%arg16 = %subview_4) -> (memref<1x1x1x64xf32, strided<[831744, 7296, 64, 1], offset: ?>>) {
# CHECK-NEXT:          %subview_6 = memref.subview %alloca[0, %arg15, 0, 0] [1, 1, 7, 3] [1, 1, 1, 1] : memref<1x7x7x3xf32> to memref<1x1x7x3xf32, strided<[147, 21, 3, 1], offset: ?>>
# CHECK-NEXT:          %subview_7 = memref.subview %arg1[%arg15, 0, 0, 0] [1, 7, 3, 64] [1, 1, 1, 1] : memref<7x7x3x64xf32> to memref<1x7x3x64xf32, strided<[1344, 192, 64, 1], offset: ?>>
# CHECK-NEXT:          %3 = scf.for %arg17 = %c0 to %c7 step %c1 iter_args(%arg18 = %arg16) -> (memref<1x1x1x64xf32, strided<[831744, 7296, 64, 1], offset: ?>>) {
# CHECK-NEXT:            %subview_8 = memref.subview %subview_6[0, 0, %arg17, 0] [1, 1, 1, 3] [1, 1, 1, 1] : memref<1x1x7x3xf32, strided<[147, 21, 3, 1], offset: ?>> to memref<1x1x1x3xf32, strided<[147, 21, 3, 1], offset: ?>>
# CHECK-NEXT:            %subview_9 = memref.subview %subview_7[0, %arg17, 0, 0] [1, 1, 3, 64] [1, 1, 1, 1] : memref<1x7x3x64xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x1x3x64xf32, strided<[1344, 192, 64, 1], offset: ?>>
# CHECK-NEXT:            %4 = scf.for %arg19 = %c0 to %c3 step %c1 iter_args(%arg20 = %arg18) -> (memref<1x1x1x64xf32, strided<[831744, 7296, 64, 1], offset: ?>>) {
# CHECK-NEXT:              %subview_10 = memref.subview %subview_8[0, 0, 0, %arg19] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x3xf32, strided<[147, 21, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[147, 21, 3, 1], offset: ?>>
# CHECK-NEXT:              %subview_11 = memref.subview %subview_9[0, 0, %arg19, 0] [1, 1, 1, 64] [1, 1, 1, 1] : memref<1x1x3x64xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x1x1x64xf32, strided<[1344, 192, 64, 1], offset: ?>>
# CHECK-NEXT:              %5 = scf.for %arg21 = %c0 to %c64 step %c1 iter_args(%arg22 = %arg20) -> (memref<1x1x1x64xf32, strided<[831744, 7296, 64, 1], offset: ?>>) {
# CHECK-NEXT:                %subview_12 = memref.subview %subview_11[0, 0, 0, %arg21] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x64xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[1344, 192, 64, 1], offset: ?>>
# CHECK-NEXT:                %subview_13 = memref.subview %arg22[0, 0, 0, %arg21] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x64xf32, strided<[831744, 7296, 64, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[831744, 7296, 64, 1], offset: ?>>
# CHECK-NEXT:                linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_10, %subview_12 : memref<1x1x1x1xf32, strided<[147, 21, 3, 1], offset: ?>>, memref<1x1x1x1xf32, strided<[1344, 192, 64, 1], offset: ?>>) outs(%subview_13 : memref<1x1x1x1xf32, strided<[831744, 7296, 64, 1], offset: ?>>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                ^bb0(%in: f32, %in_15: f32, %out: f32):
# CHECK-NEXT:                  %6 = arith.mulf %in, %in_15 fastmath<fast> : f32
# CHECK-NEXT:                  %7 = arith.addf %out, %6 fastmath<fast> : f32
# CHECK-NEXT:                  linalg.yield %7 : f32
# CHECK-NEXT:                }
# CHECK-NEXT:                %subview_14 = memref.subview %arg22[0, 0, 0, %arg21] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x64xf32, strided<[831744, 7296, 64, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[831744, 7296, 64, 1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_13, %subview_14 : memref<1x1x1x1xf32, strided<[831744, 7296, 64, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[831744, 7296, 64, 1], offset: ?>>
# CHECK-NEXT:                scf.yield %arg22 : memref<1x1x1x64xf32, strided<[831744, 7296, 64, 1], offset: ?>>
# CHECK-NEXT:              } {"./f"}
# CHECK-NEXT:              scf.yield %5 : memref<1x1x1x64xf32, strided<[831744, 7296, 64, 1], offset: ?>>
# CHECK-NEXT:            } {"./c"}
# CHECK-NEXT:            scf.yield %4 : memref<1x1x1x64xf32, strided<[831744, 7296, 64, 1], offset: ?>>
# CHECK-NEXT:          } {"./s"}
# CHECK-NEXT:          scf.yield %3 : memref<1x1x1x64xf32, strided<[831744, 7296, 64, 1], offset: ?>>
# CHECK-NEXT:        } {"./r"}
# CHECK-NEXT:        linalg.generic {indexing_maps = [#map1, #map6, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %cst : memref<1x1x1x64xf32, strided<[831744, 7296, 64, 1], offset: ?>>, f32) outs(%2 : memref<1x1x1x64xf32, strided<[831744, 7296, 64, 1], offset: ?>>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:        ^bb0(%in: f32, %in_6: f32, %out: f32):
# CHECK-NEXT:          %3 = arith.maximumf %in, %in_6 : f32
# CHECK-NEXT:          linalg.yield %3 : f32
# CHECK-NEXT:        }
# CHECK-NEXT:        %subview_5 = memref.subview %arg2[0, %block_id_y, %block_id_z, 0] [1, 1, 1, 64] [1, 1, 1, 1] : memref<1x114x114x64xf32> to memref<1x1x1x64xf32, strided<[831744, 7296, 64, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %2, %subview_5 : memref<1x1x1x64xf32, strided<[831744, 7296, 64, 1], offset: ?>> to memref<1x1x1x64xf32, strided<[831744, 7296, 64, 1], offset: ?>>
# CHECK-NEXT:        gpu.terminator
# CHECK-NEXT:      }
# CHECK-NEXT:      memref.copy %arg2, %arg2 : memref<1x114x114x64xf32> to memref<1x114x114x64xf32>
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  graph:
# CHECK-NEXT:    name: pad_conv2d_nhwc_mini
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 1x230x230x3xfloat32
# CHECK-NEXT:    - %1 : 7x7x3x64xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %4 : 1x114x114x64xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: pad2d(%0, padding={1: (2, 2), 2: (2, 2)}, constant_value=0) {name = 'pad'} : [1x230x230x3xfloat32] -> [1x234x234x3xfloat32]
# CHECK-NEXT:    - %3: conv2d(%2, %1, stride=(2, 2)) {name = 'conv'} : [1x234x234x3xfloat32, 7x7x3x64xfloat32] -> [1x114x114x64xfloat32]
# CHECK-NEXT:    - %4: relu(%3) {name = 'relu'} : [1x114x114x64xfloat32] -> [1x114x114x64xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
