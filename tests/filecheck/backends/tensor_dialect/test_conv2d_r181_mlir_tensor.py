# RUN: python %s 2>&1 | filecheck %s

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend
from xtc.artifacts import get_operation

op = get_operation("conv2d", "ResNet18_01")
N, H, W, F, R, S, C = [op["dims"][k] for k in ["n", "h", "w", "f", "r", "s", "c"]]
SH, SW = [op["params"][k] for k in ["SH", "SW"]]
dtype = "float32"

a = O.tensor((N, H + R - 1, W + S - 1, C), dtype)
b = O.tensor((R, S, C, F), dtype)

with O.graph(name="conv2d_nhwc_r181") as gb:
    O.conv2d(a, b, stride=(SH, SW), name="O")

graph = gb.graph
print(graph)

impl = Backend(graph, use_tensor_dialect=True)

sch = impl.get_scheduler()
sch.tile("w", {"w1": 4})
sch.tile("f", {"f1": 16})
sch.interchange(["b", "h", "w", "f", "r", "s", "c", "w1", "f1"])
sch.vectorize(["f1"])
sch.unroll({"w1": 4, "c": 3})
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="conv2d_nhwc_r181_mlir_tensor",
    print_source_ir=True,
    print_transformed_ir=True,
    print_bufferization_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT:  #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @conv2d_nhwc_r181(%arg0: tensor<1x230x230x3xf32> {llvm.noalias}, %arg1: tensor<7x7x3x64xf32> {llvm.noalias}, %arg2: memref<1x112x112x64xf32> {llvm.noalias}) {
# CHECK-NEXT:      %0 = tensor.empty() : tensor<1x112x112x64xf32>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %1 = linalg.fill {__xtc_id_O_0_} ins(%cst : f32) outs(%0 : tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
# CHECK-NEXT:      %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<1x230x230x3xf32>, tensor<7x7x3x64xf32>) outs(%1 : tensor<1x112x112x64xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_0: f32, %out: f32):
# CHECK-NEXT:        %3 = arith.mulf %in, %in_0 fastmath<fast> : f32
# CHECK-NEXT:        %4 = arith.addf %out, %3 fastmath<fast> : f32
# CHECK-NEXT:        linalg.yield %4 : f32
# CHECK-NEXT:      } -> tensor<1x112x112x64xf32>
# CHECK-NEXT:      bufferization.materialize_in_destination %2 in restrict writable %arg2 : (tensor<1x112x112x64xf32>, memref<1x112x112x64xf32>) -> ()
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_post_bufferize(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match attributes {sym_name = "conv2d_nhwc_r181"} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %0 {
# CHECK-NEXT:        transform.apply_patterns.vector.lower_outerproduct
# CHECK-NEXT:        transform.apply_patterns.vector.lower_contraction
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      %1 = transform.structured.match attributes {"./c"} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.loop.unroll %1 {factor = 3 : i64} : !transform.any_op
# CHECK-NEXT:      %2 = transform.structured.match attributes {"./w1"} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.loop.unroll %2 {factor = 4 : i64} : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match attributes {__xtc_id_O_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops "./b" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_1 "./h" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %tiled_linalg_op_0 tile_sizes [0, 0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_3 "./w" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "./f" : !transform.any_op
# CHECK-NEXT:      %1 = transform.structured.match attributes {__xtc_id_O_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %1 tile_sizes [1, 0, 0, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_7 "./b" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [0, 1, 0, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_9 "./h" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %tiled_linalg_op_8 tile_sizes [0, 0, 4, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_11 "./w" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_12, %loops_13 = transform.structured.tile_using_for %tiled_linalg_op_10 tile_sizes [0, 0, 0, 16, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_13 "./f" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_14, %loops_15 = transform.structured.tile_using_for %tiled_linalg_op_12 tile_sizes [0, 0, 0, 0, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_15 "./r" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_16, %loops_17 = transform.structured.tile_using_for %tiled_linalg_op_14 tile_sizes [0, 0, 0, 0, 0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_17 "./s" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_18, %loops_19 = transform.structured.tile_using_for %tiled_linalg_op_16 tile_sizes [0, 0, 0, 0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_19 "./c" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_20, %loops_21 = transform.structured.tile_using_for %tiled_linalg_op_18 tile_sizes [0, 0, 1, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_21 "./w1" : !transform.any_op
# CHECK-NEXT:      %2 = transform.get_parent_op %tiled_linalg_op_20 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %2 {
# CHECK-NEXT:        transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      %3 = transform.structured.match interface{LinalgOp} in %2 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.include @_vecto failures(suppress) (%3) : (!transform.any_op) -> ()
# CHECK-NEXT:      %4 = transform.get_parent_op %loops_7 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %4 {
# CHECK-NEXT:        transform.apply_patterns.vector.reduction_to_contract
# CHECK-NEXT:        transform.apply_patterns.vector.transfer_permutation_patterns
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0) -> (d0 * 2)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @conv2d_nhwc_r181(%arg0: tensor<1x230x230x3xf32> {llvm.noalias}, %arg1: tensor<7x7x3x64xf32> {llvm.noalias}, %arg2: memref<1x112x112x64xf32> {llvm.noalias}) {
# CHECK-NEXT:      %0 = ub.poison : f32
# CHECK-NEXT:      %c3 = arith.constant 3 : index
# CHECK-NEXT:      %c7 = arith.constant 7 : index
# CHECK-NEXT:      %c16 = arith.constant 16 : index
# CHECK-NEXT:      %c4 = arith.constant 4 : index
# CHECK-NEXT:      %c64 = arith.constant 64 : index
# CHECK-NEXT:      %c112 = arith.constant 112 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %1 = tensor.empty() : tensor<1x112x112x64xf32>
# CHECK-NEXT:      %2 = scf.for %arg3 = %c0 to %c1 step %c1 iter_args(%arg4 = %1) -> (tensor<1x112x112x64xf32>) {
# CHECK-NEXT:        %extracted_slice = tensor.extract_slice %arg4[%arg3, 0, 0, 0] [1, 112, 112, 64] [1, 1, 1, 1] : tensor<1x112x112x64xf32> to tensor<1x112x112x64xf32>
# CHECK-NEXT:        %4 = scf.for %arg5 = %c0 to %c112 step %c1 iter_args(%arg6 = %extracted_slice) -> (tensor<1x112x112x64xf32>) {
# CHECK-NEXT:          %extracted_slice_0 = tensor.extract_slice %arg6[0, %arg5, 0, 0] [1, 1, 112, 64] [1, 1, 1, 1] : tensor<1x112x112x64xf32> to tensor<1x1x112x64xf32>
# CHECK-NEXT:          %5 = scf.for %arg7 = %c0 to %c112 step %c1 iter_args(%arg8 = %extracted_slice_0) -> (tensor<1x1x112x64xf32>) {
# CHECK-NEXT:            %extracted_slice_2 = tensor.extract_slice %arg8[0, 0, %arg7, 0] [1, 1, 1, 64] [1, 1, 1, 1] : tensor<1x1x112x64xf32> to tensor<1x1x1x64xf32>
# CHECK-NEXT:            %6 = scf.for %arg9 = %c0 to %c64 step %c1 iter_args(%arg10 = %extracted_slice_2) -> (tensor<1x1x1x64xf32>) {
# CHECK-NEXT:              %extracted_slice_4 = tensor.extract_slice %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x64xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:              %7 = linalg.fill {__xtc_id_O_0_} ins(%cst : f32) outs(%extracted_slice_4 : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
# CHECK-NEXT:              %inserted_slice_5 = tensor.insert_slice %7 into %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x64xf32>
# CHECK-NEXT:              scf.yield %inserted_slice_5 : tensor<1x1x1x64xf32>
# CHECK-NEXT:            } {"./f"}
# CHECK-NEXT:            %inserted_slice_3 = tensor.insert_slice %6 into %arg8[0, 0, %arg7, 0] [1, 1, 1, 64] [1, 1, 1, 1] : tensor<1x1x1x64xf32> into tensor<1x1x112x64xf32>
# CHECK-NEXT:            scf.yield %inserted_slice_3 : tensor<1x1x112x64xf32>
# CHECK-NEXT:          } {"./w"}
# CHECK-NEXT:          %inserted_slice_1 = tensor.insert_slice %5 into %arg6[0, %arg5, 0, 0] [1, 1, 112, 64] [1, 1, 1, 1] : tensor<1x1x112x64xf32> into tensor<1x112x112x64xf32>
# CHECK-NEXT:          scf.yield %inserted_slice_1 : tensor<1x112x112x64xf32>
# CHECK-NEXT:        } {"./h"}
# CHECK-NEXT:        %inserted_slice = tensor.insert_slice %4 into %arg4[%arg3, 0, 0, 0] [1, 112, 112, 64] [1, 1, 1, 1] : tensor<1x112x112x64xf32> into tensor<1x112x112x64xf32>
# CHECK-NEXT:        scf.yield %inserted_slice : tensor<1x112x112x64xf32>
# CHECK-NEXT:      } {"./b"}
# CHECK-NEXT:      %3 = scf.for %arg3 = %c0 to %c1 step %c1 iter_args(%arg4 = %2) -> (tensor<1x112x112x64xf32>) {
# CHECK-NEXT:        %extracted_slice = tensor.extract_slice %arg0[%arg3, 0, 0, 0] [1, 229, 229, 3] [1, 1, 1, 1] : tensor<1x230x230x3xf32> to tensor<1x229x229x3xf32>
# CHECK-NEXT:        %extracted_slice_0 = tensor.extract_slice %arg4[%arg3, 0, 0, 0] [1, 112, 112, 64] [1, 1, 1, 1] : tensor<1x112x112x64xf32> to tensor<1x112x112x64xf32>
# CHECK-NEXT:        %4 = scf.for %arg5 = %c0 to %c112 step %c1 iter_args(%arg6 = %extracted_slice_0) -> (tensor<1x112x112x64xf32>) {
# CHECK-NEXT:          %5 = affine.apply #map(%arg5)
# CHECK-NEXT:          %extracted_slice_1 = tensor.extract_slice %extracted_slice[0, %5, 0, 0] [1, 7, 229, 3] [1, 1, 1, 1] : tensor<1x229x229x3xf32> to tensor<1x7x229x3xf32>
# CHECK-NEXT:          %extracted_slice_2 = tensor.extract_slice %arg6[0, %arg5, 0, 0] [1, 1, 112, 64] [1, 1, 1, 1] : tensor<1x112x112x64xf32> to tensor<1x1x112x64xf32>
# CHECK-NEXT:          %6 = scf.for %arg7 = %c0 to %c112 step %c4 iter_args(%arg8 = %extracted_slice_2) -> (tensor<1x1x112x64xf32>) {
# CHECK-NEXT:            %7 = affine.apply #map(%arg7)
# CHECK-NEXT:            %extracted_slice_4 = tensor.extract_slice %extracted_slice_1[0, 0, %7, 0] [1, 7, 13, 3] [1, 1, 1, 1] : tensor<1x7x229x3xf32> to tensor<1x7x13x3xf32>
# CHECK-NEXT:            %extracted_slice_5 = tensor.extract_slice %arg8[0, 0, %arg7, 0] [1, 1, 4, 64] [1, 1, 1, 1] : tensor<1x1x112x64xf32> to tensor<1x1x4x64xf32>
# CHECK-NEXT:            %8 = scf.for %arg9 = %c0 to %c64 step %c16 iter_args(%arg10 = %extracted_slice_5) -> (tensor<1x1x4x64xf32>) {
# CHECK-NEXT:              %extracted_slice_7 = tensor.extract_slice %arg1[0, 0, 0, %arg9] [7, 7, 3, 16] [1, 1, 1, 1] : tensor<7x7x3x64xf32> to tensor<7x7x3x16xf32>
# CHECK-NEXT:              %extracted_slice_8 = tensor.extract_slice %arg10[0, 0, 0, %arg9] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x1x4x64xf32> to tensor<1x1x4x16xf32>
# CHECK-NEXT:              %9 = scf.for %arg11 = %c0 to %c7 step %c1 iter_args(%arg12 = %extracted_slice_8) -> (tensor<1x1x4x16xf32>) {
# CHECK-NEXT:                %extracted_slice_10 = tensor.extract_slice %extracted_slice_4[0, %arg11, 0, 0] [1, 1, 13, 3] [1, 1, 1, 1] : tensor<1x7x13x3xf32> to tensor<1x1x13x3xf32>
# CHECK-NEXT:                %extracted_slice_11 = tensor.extract_slice %extracted_slice_7[%arg11, 0, 0, 0] [1, 7, 3, 16] [1, 1, 1, 1] : tensor<7x7x3x16xf32> to tensor<1x7x3x16xf32>
# CHECK-NEXT:                %10 = scf.for %arg13 = %c0 to %c7 step %c1 iter_args(%arg14 = %arg12) -> (tensor<1x1x4x16xf32>) {
# CHECK-NEXT:                  %extracted_slice_12 = tensor.extract_slice %extracted_slice_10[0, 0, %arg13, 0] [1, 1, 7, 3] [1, 1, 1, 1] : tensor<1x1x13x3xf32> to tensor<1x1x7x3xf32>
# CHECK-NEXT:                  %extracted_slice_13 = tensor.extract_slice %extracted_slice_11[0, %arg13, 0, 0] [1, 1, 3, 16] [1, 1, 1, 1] : tensor<1x7x3x16xf32> to tensor<1x1x3x16xf32>
# CHECK-NEXT:                  %11 = scf.for %arg15 = %c0 to %c3 step %c1 iter_args(%arg16 = %arg14) -> (tensor<1x1x4x16xf32>) {
# CHECK-NEXT:                    %extracted_slice_14 = tensor.extract_slice %extracted_slice_12[0, 0, 0, %arg15] [1, 1, 7, 1] [1, 1, 1, 1] : tensor<1x1x7x3xf32> to tensor<1x1x7x1xf32>
# CHECK-NEXT:                    %extracted_slice_15 = tensor.extract_slice %extracted_slice_13[0, 0, %arg15, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x3x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                    %12 = scf.for %arg17 = %c0 to %c4 step %c1 iter_args(%arg18 = %arg16) -> (tensor<1x1x4x16xf32>) {
# CHECK-NEXT:                      %13 = affine.apply #map(%arg17)
# CHECK-NEXT:                      %extracted_slice_16 = tensor.extract_slice %extracted_slice_14[0, 0, %13, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                      %extracted_slice_17 = tensor.extract_slice %arg18[0, 0, %arg17, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                      %extracted_slice_18 = tensor.extract_slice %extracted_slice_16[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> to tensor<f32>
# CHECK-NEXT:                      %extracted_slice_19 = tensor.extract_slice %extracted_slice_15[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> to tensor<16xf32>
# CHECK-NEXT:                      %extracted_slice_20 = tensor.extract_slice %extracted_slice_17[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> to tensor<16xf32>
# CHECK-NEXT:                      %extracted = tensor.extract %extracted_slice_18[] : tensor<f32>
# CHECK-NEXT:                      %14 = vector.broadcast %extracted : f32 to vector<16xf32>
# CHECK-NEXT:                      %15 = vector.transfer_read %extracted_slice_19[%c0], %0 {in_bounds = [true]} : tensor<16xf32>, vector<16xf32>
# CHECK-NEXT:                      %16 = vector.transfer_read %extracted_slice_20[%c0], %0 {in_bounds = [true]} : tensor<16xf32>, vector<16xf32>
# CHECK-NEXT:                      %17 = arith.mulf %14, %15 fastmath<fast> : vector<16xf32>
# CHECK-NEXT:                      %18 = arith.addf %16, %17 fastmath<fast> : vector<16xf32>
# CHECK-NEXT:                      %19 = vector.transfer_write %18, %extracted_slice_20[%c0] {in_bounds = [true]} : vector<16xf32>, tensor<16xf32>
# CHECK-NEXT:                      %inserted_slice_21 = tensor.insert_slice %19 into %extracted_slice_17[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<16xf32> into tensor<1x1x1x16xf32>
# CHECK-NEXT:                      %inserted_slice_22 = tensor.insert_slice %inserted_slice_21 into %arg18[0, 0, %arg17, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                      scf.yield %inserted_slice_22 : tensor<1x1x4x16xf32>
# CHECK-NEXT:                    } {"./w1"}
# CHECK-NEXT:                    scf.yield %12 : tensor<1x1x4x16xf32>
# CHECK-NEXT:                  } {"./c"}
# CHECK-NEXT:                  scf.yield %11 : tensor<1x1x4x16xf32>
# CHECK-NEXT:                } {"./s"}
# CHECK-NEXT:                scf.yield %10 : tensor<1x1x4x16xf32>
# CHECK-NEXT:              } {"./r"}
# CHECK-NEXT:              %inserted_slice_9 = tensor.insert_slice %9 into %arg10[0, 0, 0, %arg9] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> into tensor<1x1x4x64xf32>
# CHECK-NEXT:              scf.yield %inserted_slice_9 : tensor<1x1x4x64xf32>
# CHECK-NEXT:            } {"./f"}
# CHECK-NEXT:            %inserted_slice_6 = tensor.insert_slice %8 into %arg8[0, 0, %arg7, 0] [1, 1, 4, 64] [1, 1, 1, 1] : tensor<1x1x4x64xf32> into tensor<1x1x112x64xf32>
# CHECK-NEXT:            scf.yield %inserted_slice_6 : tensor<1x1x112x64xf32>
# CHECK-NEXT:          } {"./w"}
# CHECK-NEXT:          %inserted_slice_3 = tensor.insert_slice %6 into %arg6[0, %arg5, 0, 0] [1, 1, 112, 64] [1, 1, 1, 1] : tensor<1x1x112x64xf32> into tensor<1x112x112x64xf32>
# CHECK-NEXT:          scf.yield %inserted_slice_3 : tensor<1x112x112x64xf32>
# CHECK-NEXT:        } {"./h"}
# CHECK-NEXT:        %inserted_slice = tensor.insert_slice %4 into %arg4[%arg3, 0, 0, 0] [1, 112, 112, 64] [1, 1, 1, 1] : tensor<1x112x112x64xf32> into tensor<1x112x112x64xf32>
# CHECK-NEXT:        scf.yield %inserted_slice : tensor<1x112x112x64xf32>
# CHECK-NEXT:      } {"./b"}
# CHECK-NEXT:      bufferization.materialize_in_destination %3 in restrict writable %arg2 : (tensor<1x112x112x64xf32>, memref<1x112x112x64xf32>) -> ()
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_post_bufferize(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match attributes {sym_name = "conv2d_nhwc_r181"} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %0 {
# CHECK-NEXT:        transform.apply_patterns.vector.lower_outerproduct
# CHECK-NEXT:        transform.apply_patterns.vector.lower_contraction
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      %1 = transform.structured.match attributes {"./c"} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.loop.unroll %1 {factor = 3 : i64} : !transform.any_op
# CHECK-NEXT:      %2 = transform.structured.match attributes {"./w1"} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.loop.unroll %2 {factor = 4 : i64} : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After Tensor Lowering //----- //
# CHECK-NEXT:  #map = affine_map<(d0) -> (d0 * 2)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @conv2d_nhwc_r181(%arg0: memref<1x230x230x3xf32> {llvm.noalias}, %arg1: memref<7x7x3x64xf32> {llvm.noalias}, %arg2: memref<1x112x112x64xf32> {llvm.noalias}) {
# CHECK-NEXT:      %0 = ub.poison : f32
# CHECK-NEXT:      %c3 = arith.constant 3 : index
# CHECK-NEXT:      %c7 = arith.constant 7 : index
# CHECK-NEXT:      %c16 = arith.constant 16 : index
# CHECK-NEXT:      %c4 = arith.constant 4 : index
# CHECK-NEXT:      %c64 = arith.constant 64 : index
# CHECK-NEXT:      %c112 = arith.constant 112 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %1 = scf.for %arg3 = %c0 to %c112 step %c1 iter_args(%arg4 = %arg2) -> (memref<1x112x112x64xf32>) {
# CHECK-NEXT:        %subview_0 = memref.subview %arg4[0, %arg3, 0, 0] [1, 1, 112, 64] [1, 1, 1, 1] : memref<1x112x112x64xf32> to memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:        %3 = scf.for %arg5 = %c0 to %c112 step %c1 iter_args(%arg6 = %subview_0) -> (memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>) {
# CHECK-NEXT:          %subview_2 = memref.subview %arg6[0, 0, %arg5, 0] [1, 1, 1, 64] [1, 1, 1, 1] : memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:          %4 = scf.for %arg7 = %c0 to %c64 step %c1 iter_args(%arg8 = %subview_2) -> (memref<1x1x1x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>) {
# CHECK-NEXT:            %subview_4 = memref.subview %arg8[0, 0, 0, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:            linalg.fill {__xtc_id_O_0_} ins(%cst : f32) outs(%subview_4 : memref<1x1x1x1xf32, strided<[802816, 7168, 64, 1], offset: ?>>)
# CHECK-NEXT:            %subview_5 = memref.subview %arg8[0, 0, 0, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:            memref.copy %subview_4, %subview_5 : memref<1x1x1x1xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:            scf.yield %arg8 : memref<1x1x1x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:          } {"./f"}
# CHECK-NEXT:          %subview_3 = memref.subview %arg6[0, 0, %arg5, 0] [1, 1, 1, 64] [1, 1, 1, 1] : memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:          memref.copy %4, %subview_3 : memref<1x1x1x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:          scf.yield %arg6 : memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:        } {"./w"}
# CHECK-NEXT:        %subview_1 = memref.subview %arg4[0, %arg3, 0, 0] [1, 1, 112, 64] [1, 1, 1, 1] : memref<1x112x112x64xf32> to memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %3, %subview_1 : memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:        scf.yield %arg4 : memref<1x112x112x64xf32>
# CHECK-NEXT:      } {"./h"}
# CHECK-NEXT:      %subview = memref.subview %arg0[0, 0, 0, 0] [1, 229, 229, 3] [1, 1, 1, 1] : memref<1x230x230x3xf32> to memref<1x229x229x3xf32, strided<[158700, 690, 3, 1]>>
# CHECK-NEXT:      %2 = scf.for %arg3 = %c0 to %c112 step %c1 iter_args(%arg4 = %1) -> (memref<1x112x112x64xf32>) {
# CHECK-NEXT:        %3 = affine.apply #map(%arg3)
# CHECK-NEXT:        %subview_0 = memref.subview %subview[0, %3, 0, 0] [1, 7, 229, 3] [1, 1, 1, 1] : memref<1x229x229x3xf32, strided<[158700, 690, 3, 1]>> to memref<1x7x229x3xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:        %subview_1 = memref.subview %arg4[0, %arg3, 0, 0] [1, 1, 112, 64] [1, 1, 1, 1] : memref<1x112x112x64xf32> to memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:        %4 = scf.for %arg5 = %c0 to %c112 step %c4 iter_args(%arg6 = %subview_1) -> (memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>) {
# CHECK-NEXT:          %5 = affine.apply #map(%arg5)
# CHECK-NEXT:          %subview_3 = memref.subview %subview_0[0, 0, %5, 0] [1, 7, 13, 3] [1, 1, 1, 1] : memref<1x7x229x3xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x7x13x3xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:          %subview_4 = memref.subview %arg6[0, 0, %arg5, 0] [1, 1, 4, 64] [1, 1, 1, 1] : memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x4x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:          %6 = scf.for %arg7 = %c0 to %c64 step %c16 iter_args(%arg8 = %subview_4) -> (memref<1x1x4x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>) {
# CHECK-NEXT:            %subview_6 = memref.subview %arg1[0, 0, 0, %arg7] [7, 7, 3, 16] [1, 1, 1, 1] : memref<7x7x3x64xf32> to memref<7x7x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>>
# CHECK-NEXT:            %subview_7 = memref.subview %arg8[0, 0, 0, %arg7] [1, 1, 4, 16] [1, 1, 1, 1] : memref<1x1x4x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:            %7 = scf.for %arg9 = %c0 to %c7 step %c1 iter_args(%arg10 = %subview_7) -> (memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>) {
# CHECK-NEXT:              %subview_9 = memref.subview %subview_3[0, %arg9, 0, 0] [1, 1, 13, 3] [1, 1, 1, 1] : memref<1x7x13x3xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x13x3xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:              %subview_10 = memref.subview %subview_6[%arg9, 0, 0, 0] [1, 7, 3, 16] [1, 1, 1, 1] : memref<7x7x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x7x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>>
# CHECK-NEXT:              %8 = scf.for %arg11 = %c0 to %c7 step %c1 iter_args(%arg12 = %arg10) -> (memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>) {
# CHECK-NEXT:                %subview_11 = memref.subview %subview_9[0, 0, %arg11, 0] [1, 1, 7, 3] [1, 1, 1, 1] : memref<1x1x13x3xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x7x3xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                %subview_12 = memref.subview %subview_10[0, %arg11, 0, 0] [1, 1, 3, 16] [1, 1, 1, 1] : memref<1x7x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x1x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>>
# CHECK-NEXT:                %c3_13 = arith.constant 3 : index
# CHECK-NEXT:                %subview_14 = memref.subview %subview_11[0, 0, 0, %c0] [1, 1, 7, 1] [1, 1, 1, 1] : memref<1x1x7x3xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                %subview_15 = memref.subview %subview_12[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>
# CHECK-NEXT:                %c4_16 = arith.constant 4 : index
# CHECK-NEXT:                %9 = affine.apply #map(%c0)
# CHECK-NEXT:                %subview_17 = memref.subview %subview_14[0, 0, %9, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                %subview_18 = memref.subview %arg12[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                %subview_19 = memref.subview %subview_17[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<f32, strided<[], offset: ?>>
# CHECK-NEXT:                %subview_20 = memref.subview %subview_15[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_21 = memref.subview %subview_18[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %10 = memref.load %subview_19[] : memref<f32, strided<[], offset: ?>>
# CHECK-NEXT:                %11 = vector.broadcast %10 : f32 to vector<16xf32>
# CHECK-NEXT:                %12 = vector.transfer_read %subview_20[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                %13 = vector.transfer_read %subview_21[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                %14 = arith.mulf %11, %12 fastmath<fast> : vector<16xf32>
# CHECK-NEXT:                %15 = arith.addf %13, %14 fastmath<fast> : vector<16xf32>
# CHECK-NEXT:                vector.transfer_write %15, %subview_21[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_22 = memref.subview %subview_18[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_21, %subview_22 : memref<16xf32, strided<[1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_23 = memref.subview %arg12[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_18, %subview_23 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                %c1_24 = arith.constant 1 : index
# CHECK-NEXT:                %16 = arith.muli %c1, %c1_24 : index
# CHECK-NEXT:                %17 = arith.addi %c0, %16 : index
# CHECK-NEXT:                %18 = affine.apply #map(%17)
# CHECK-NEXT:                %subview_25 = memref.subview %subview_14[0, 0, %18, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                %subview_26 = memref.subview %arg12[0, 0, %17, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                %subview_27 = memref.subview %subview_25[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<f32, strided<[], offset: ?>>
# CHECK-NEXT:                %subview_28 = memref.subview %subview_15[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_29 = memref.subview %subview_26[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %19 = memref.load %subview_27[] : memref<f32, strided<[], offset: ?>>
# CHECK-NEXT:                %20 = vector.broadcast %19 : f32 to vector<16xf32>
# CHECK-NEXT:                %21 = vector.transfer_read %subview_28[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                %22 = vector.transfer_read %subview_29[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                %23 = arith.mulf %20, %21 fastmath<fast> : vector<16xf32>
# CHECK-NEXT:                %24 = arith.addf %22, %23 fastmath<fast> : vector<16xf32>
# CHECK-NEXT:                vector.transfer_write %24, %subview_29[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_30 = memref.subview %subview_26[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_29, %subview_30 : memref<16xf32, strided<[1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_31 = memref.subview %arg12[0, 0, %17, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_26, %subview_31 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                %c2 = arith.constant 2 : index
# CHECK-NEXT:                %25 = arith.muli %c1, %c2 : index
# CHECK-NEXT:                %26 = arith.addi %c0, %25 : index
# CHECK-NEXT:                %27 = affine.apply #map(%26)
# CHECK-NEXT:                %subview_32 = memref.subview %subview_14[0, 0, %27, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                %subview_33 = memref.subview %arg12[0, 0, %26, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                %subview_34 = memref.subview %subview_32[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<f32, strided<[], offset: ?>>
# CHECK-NEXT:                %subview_35 = memref.subview %subview_15[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_36 = memref.subview %subview_33[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %28 = memref.load %subview_34[] : memref<f32, strided<[], offset: ?>>
# CHECK-NEXT:                %29 = vector.broadcast %28 : f32 to vector<16xf32>
# CHECK-NEXT:                %30 = vector.transfer_read %subview_35[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                %31 = vector.transfer_read %subview_36[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                %32 = arith.mulf %29, %30 fastmath<fast> : vector<16xf32>
# CHECK-NEXT:                %33 = arith.addf %31, %32 fastmath<fast> : vector<16xf32>
# CHECK-NEXT:                vector.transfer_write %33, %subview_36[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_37 = memref.subview %subview_33[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_36, %subview_37 : memref<16xf32, strided<[1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_38 = memref.subview %arg12[0, 0, %26, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_33, %subview_38 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                %c3_39 = arith.constant 3 : index
# CHECK-NEXT:                %34 = arith.muli %c1, %c3_39 : index
# CHECK-NEXT:                %35 = arith.addi %c0, %34 : index
# CHECK-NEXT:                %36 = affine.apply #map(%35)
# CHECK-NEXT:                %subview_40 = memref.subview %subview_14[0, 0, %36, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                %subview_41 = memref.subview %arg12[0, 0, %35, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                %subview_42 = memref.subview %subview_40[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<f32, strided<[], offset: ?>>
# CHECK-NEXT:                %subview_43 = memref.subview %subview_15[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_44 = memref.subview %subview_41[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %37 = memref.load %subview_42[] : memref<f32, strided<[], offset: ?>>
# CHECK-NEXT:                %38 = vector.broadcast %37 : f32 to vector<16xf32>
# CHECK-NEXT:                %39 = vector.transfer_read %subview_43[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                %40 = vector.transfer_read %subview_44[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                %41 = arith.mulf %38, %39 fastmath<fast> : vector<16xf32>
# CHECK-NEXT:                %42 = arith.addf %40, %41 fastmath<fast> : vector<16xf32>
# CHECK-NEXT:                vector.transfer_write %42, %subview_44[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_45 = memref.subview %subview_41[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_44, %subview_45 : memref<16xf32, strided<[1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_46 = memref.subview %arg12[0, 0, %35, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_41, %subview_46 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                %c1_47 = arith.constant 1 : index
# CHECK-NEXT:                %43 = arith.muli %c1, %c1_47 : index
# CHECK-NEXT:                %44 = arith.addi %c0, %43 : index
# CHECK-NEXT:                %subview_48 = memref.subview %subview_11[0, 0, 0, %44] [1, 1, 7, 1] [1, 1, 1, 1] : memref<1x1x7x3xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                %subview_49 = memref.subview %subview_12[0, 0, %44, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>
# CHECK-NEXT:                %c4_50 = arith.constant 4 : index
# CHECK-NEXT:                %45 = affine.apply #map(%c0)
# CHECK-NEXT:                %subview_51 = memref.subview %subview_48[0, 0, %45, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                %subview_52 = memref.subview %arg12[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                %subview_53 = memref.subview %subview_51[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<f32, strided<[], offset: ?>>
# CHECK-NEXT:                %subview_54 = memref.subview %subview_49[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_55 = memref.subview %subview_52[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %46 = memref.load %subview_53[] : memref<f32, strided<[], offset: ?>>
# CHECK-NEXT:                %47 = vector.broadcast %46 : f32 to vector<16xf32>
# CHECK-NEXT:                %48 = vector.transfer_read %subview_54[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                %49 = vector.transfer_read %subview_55[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                %50 = arith.mulf %47, %48 fastmath<fast> : vector<16xf32>
# CHECK-NEXT:                %51 = arith.addf %49, %50 fastmath<fast> : vector<16xf32>
# CHECK-NEXT:                vector.transfer_write %51, %subview_55[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_56 = memref.subview %subview_52[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_55, %subview_56 : memref<16xf32, strided<[1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_57 = memref.subview %arg12[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_52, %subview_57 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                %c1_58 = arith.constant 1 : index
# CHECK-NEXT:                %52 = arith.muli %c1, %c1_58 : index
# CHECK-NEXT:                %53 = arith.addi %c0, %52 : index
# CHECK-NEXT:                %54 = affine.apply #map(%53)
# CHECK-NEXT:                %subview_59 = memref.subview %subview_48[0, 0, %54, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                %subview_60 = memref.subview %arg12[0, 0, %53, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                %subview_61 = memref.subview %subview_59[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<f32, strided<[], offset: ?>>
# CHECK-NEXT:                %subview_62 = memref.subview %subview_49[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_63 = memref.subview %subview_60[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %55 = memref.load %subview_61[] : memref<f32, strided<[], offset: ?>>
# CHECK-NEXT:                %56 = vector.broadcast %55 : f32 to vector<16xf32>
# CHECK-NEXT:                %57 = vector.transfer_read %subview_62[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                %58 = vector.transfer_read %subview_63[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                %59 = arith.mulf %56, %57 fastmath<fast> : vector<16xf32>
# CHECK-NEXT:                %60 = arith.addf %58, %59 fastmath<fast> : vector<16xf32>
# CHECK-NEXT:                vector.transfer_write %60, %subview_63[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_64 = memref.subview %subview_60[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_63, %subview_64 : memref<16xf32, strided<[1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_65 = memref.subview %arg12[0, 0, %53, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_60, %subview_65 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                %c2_66 = arith.constant 2 : index
# CHECK-NEXT:                %61 = arith.muli %c1, %c2_66 : index
# CHECK-NEXT:                %62 = arith.addi %c0, %61 : index
# CHECK-NEXT:                %63 = affine.apply #map(%62)
# CHECK-NEXT:                %subview_67 = memref.subview %subview_48[0, 0, %63, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                %subview_68 = memref.subview %arg12[0, 0, %62, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                %subview_69 = memref.subview %subview_67[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<f32, strided<[], offset: ?>>
# CHECK-NEXT:                %subview_70 = memref.subview %subview_49[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_71 = memref.subview %subview_68[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %64 = memref.load %subview_69[] : memref<f32, strided<[], offset: ?>>
# CHECK-NEXT:                %65 = vector.broadcast %64 : f32 to vector<16xf32>
# CHECK-NEXT:                %66 = vector.transfer_read %subview_70[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                %67 = vector.transfer_read %subview_71[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                %68 = arith.mulf %65, %66 fastmath<fast> : vector<16xf32>
# CHECK-NEXT:                %69 = arith.addf %67, %68 fastmath<fast> : vector<16xf32>
# CHECK-NEXT:                vector.transfer_write %69, %subview_71[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_72 = memref.subview %subview_68[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_71, %subview_72 : memref<16xf32, strided<[1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_73 = memref.subview %arg12[0, 0, %62, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_68, %subview_73 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                %c3_74 = arith.constant 3 : index
# CHECK-NEXT:                %70 = arith.muli %c1, %c3_74 : index
# CHECK-NEXT:                %71 = arith.addi %c0, %70 : index
# CHECK-NEXT:                %72 = affine.apply #map(%71)
# CHECK-NEXT:                %subview_75 = memref.subview %subview_48[0, 0, %72, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                %subview_76 = memref.subview %arg12[0, 0, %71, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                %subview_77 = memref.subview %subview_75[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<f32, strided<[], offset: ?>>
# CHECK-NEXT:                %subview_78 = memref.subview %subview_49[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_79 = memref.subview %subview_76[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %73 = memref.load %subview_77[] : memref<f32, strided<[], offset: ?>>
# CHECK-NEXT:                %74 = vector.broadcast %73 : f32 to vector<16xf32>
# CHECK-NEXT:                %75 = vector.transfer_read %subview_78[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                %76 = vector.transfer_read %subview_79[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                %77 = arith.mulf %74, %75 fastmath<fast> : vector<16xf32>
# CHECK-NEXT:                %78 = arith.addf %76, %77 fastmath<fast> : vector<16xf32>
# CHECK-NEXT:                vector.transfer_write %78, %subview_79[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_80 = memref.subview %subview_76[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_79, %subview_80 : memref<16xf32, strided<[1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_81 = memref.subview %arg12[0, 0, %71, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_76, %subview_81 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                %c2_82 = arith.constant 2 : index
# CHECK-NEXT:                %79 = arith.muli %c1, %c2_82 : index
# CHECK-NEXT:                %80 = arith.addi %c0, %79 : index
# CHECK-NEXT:                %subview_83 = memref.subview %subview_11[0, 0, 0, %80] [1, 1, 7, 1] [1, 1, 1, 1] : memref<1x1x7x3xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                %subview_84 = memref.subview %subview_12[0, 0, %80, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>
# CHECK-NEXT:                %c4_85 = arith.constant 4 : index
# CHECK-NEXT:                %81 = affine.apply #map(%c0)
# CHECK-NEXT:                %subview_86 = memref.subview %subview_83[0, 0, %81, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                %subview_87 = memref.subview %arg12[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                %subview_88 = memref.subview %subview_86[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<f32, strided<[], offset: ?>>
# CHECK-NEXT:                %subview_89 = memref.subview %subview_84[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_90 = memref.subview %subview_87[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %82 = memref.load %subview_88[] : memref<f32, strided<[], offset: ?>>
# CHECK-NEXT:                %83 = vector.broadcast %82 : f32 to vector<16xf32>
# CHECK-NEXT:                %84 = vector.transfer_read %subview_89[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                %85 = vector.transfer_read %subview_90[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                %86 = arith.mulf %83, %84 fastmath<fast> : vector<16xf32>
# CHECK-NEXT:                %87 = arith.addf %85, %86 fastmath<fast> : vector<16xf32>
# CHECK-NEXT:                vector.transfer_write %87, %subview_90[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_91 = memref.subview %subview_87[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_90, %subview_91 : memref<16xf32, strided<[1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_92 = memref.subview %arg12[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_87, %subview_92 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                %c1_93 = arith.constant 1 : index
# CHECK-NEXT:                %88 = arith.muli %c1, %c1_93 : index
# CHECK-NEXT:                %89 = arith.addi %c0, %88 : index
# CHECK-NEXT:                %90 = affine.apply #map(%89)
# CHECK-NEXT:                %subview_94 = memref.subview %subview_83[0, 0, %90, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                %subview_95 = memref.subview %arg12[0, 0, %89, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                %subview_96 = memref.subview %subview_94[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<f32, strided<[], offset: ?>>
# CHECK-NEXT:                %subview_97 = memref.subview %subview_84[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_98 = memref.subview %subview_95[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %91 = memref.load %subview_96[] : memref<f32, strided<[], offset: ?>>
# CHECK-NEXT:                %92 = vector.broadcast %91 : f32 to vector<16xf32>
# CHECK-NEXT:                %93 = vector.transfer_read %subview_97[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                %94 = vector.transfer_read %subview_98[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                %95 = arith.mulf %92, %93 fastmath<fast> : vector<16xf32>
# CHECK-NEXT:                %96 = arith.addf %94, %95 fastmath<fast> : vector<16xf32>
# CHECK-NEXT:                vector.transfer_write %96, %subview_98[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_99 = memref.subview %subview_95[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_98, %subview_99 : memref<16xf32, strided<[1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_100 = memref.subview %arg12[0, 0, %89, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_95, %subview_100 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                %c2_101 = arith.constant 2 : index
# CHECK-NEXT:                %97 = arith.muli %c1, %c2_101 : index
# CHECK-NEXT:                %98 = arith.addi %c0, %97 : index
# CHECK-NEXT:                %99 = affine.apply #map(%98)
# CHECK-NEXT:                %subview_102 = memref.subview %subview_83[0, 0, %99, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                %subview_103 = memref.subview %arg12[0, 0, %98, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                %subview_104 = memref.subview %subview_102[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<f32, strided<[], offset: ?>>
# CHECK-NEXT:                %subview_105 = memref.subview %subview_84[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_106 = memref.subview %subview_103[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %100 = memref.load %subview_104[] : memref<f32, strided<[], offset: ?>>
# CHECK-NEXT:                %101 = vector.broadcast %100 : f32 to vector<16xf32>
# CHECK-NEXT:                %102 = vector.transfer_read %subview_105[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                %103 = vector.transfer_read %subview_106[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                %104 = arith.mulf %101, %102 fastmath<fast> : vector<16xf32>
# CHECK-NEXT:                %105 = arith.addf %103, %104 fastmath<fast> : vector<16xf32>
# CHECK-NEXT:                vector.transfer_write %105, %subview_106[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_107 = memref.subview %subview_103[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_106, %subview_107 : memref<16xf32, strided<[1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_108 = memref.subview %arg12[0, 0, %98, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_103, %subview_108 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                %c3_109 = arith.constant 3 : index
# CHECK-NEXT:                %106 = arith.muli %c1, %c3_109 : index
# CHECK-NEXT:                %107 = arith.addi %c0, %106 : index
# CHECK-NEXT:                %108 = affine.apply #map(%107)
# CHECK-NEXT:                %subview_110 = memref.subview %subview_83[0, 0, %108, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                %subview_111 = memref.subview %arg12[0, 0, %107, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                %subview_112 = memref.subview %subview_110[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<f32, strided<[], offset: ?>>
# CHECK-NEXT:                %subview_113 = memref.subview %subview_84[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_114 = memref.subview %subview_111[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %109 = memref.load %subview_112[] : memref<f32, strided<[], offset: ?>>
# CHECK-NEXT:                %110 = vector.broadcast %109 : f32 to vector<16xf32>
# CHECK-NEXT:                %111 = vector.transfer_read %subview_113[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                %112 = vector.transfer_read %subview_114[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                %113 = arith.mulf %110, %111 fastmath<fast> : vector<16xf32>
# CHECK-NEXT:                %114 = arith.addf %112, %113 fastmath<fast> : vector<16xf32>
# CHECK-NEXT:                vector.transfer_write %114, %subview_114[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_115 = memref.subview %subview_111[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_114, %subview_115 : memref<16xf32, strided<[1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_116 = memref.subview %arg12[0, 0, %107, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_111, %subview_116 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                scf.yield %arg12 : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:              } {"./s"}
# CHECK-NEXT:              scf.yield %8 : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:            } {"./r"}
# CHECK-NEXT:            %subview_8 = memref.subview %arg8[0, 0, 0, %arg7] [1, 1, 4, 16] [1, 1, 1, 1] : memref<1x1x4x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:            memref.copy %7, %subview_8 : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:            scf.yield %arg8 : memref<1x1x4x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:          } {"./f"}
# CHECK-NEXT:          %subview_5 = memref.subview %arg6[0, 0, %arg5, 0] [1, 1, 4, 64] [1, 1, 1, 1] : memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x4x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:          memref.copy %6, %subview_5 : memref<1x1x4x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x4x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:          scf.yield %arg6 : memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:        } {"./w"}
# CHECK-NEXT:        %subview_2 = memref.subview %arg4[0, %arg3, 0, 0] [1, 1, 112, 64] [1, 1, 1, 1] : memref<1x112x112x64xf32> to memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %4, %subview_2 : memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:        scf.yield %arg4 : memref<1x112x112x64xf32>
# CHECK-NEXT:      } {"./h"}
# CHECK-NEXT:      memref.copy %2, %arg2 : memref<1x112x112x64xf32> to memref<1x112x112x64xf32>
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  graph:
# CHECK-NEXT:    name: conv2d_nhwc_r181
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 1x230x230x3xfloat32
# CHECK-NEXT:    - %1 : 7x7x3x64xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2 : 1x112x112x64xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: conv2d(%0, %1, stride=(2, 2)) {name = 'O'} : [1x230x230x3xfloat32, 7x7x3x64xfloat32] -> [1x112x112x64xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
