# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm
# REQUIRES: tvm_cuda

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend
from xtc.runtimes.accelerator.gpu import GPUDevice

gpu = GPUDevice()
dtype = "float32"
N, H, W, C, R, S, F = 1, 10, 10, 3, 3, 3, 16
a = O.tensor((N, H + R - 1, W + S - 1, C), dtype, name="A", device=gpu)
b = O.tensor((R, S, C, F), dtype, name="B", device=gpu)

with O.graph(name="conv2d_nhwc_gpu") as gb:
    O.conv2d(a, b, stride=(1, 1), name="O", device=gpu)

graph = gb.graph

impl = Backend(graph)

sch = impl.get_scheduler()
sch.tile("h", {"h1": 4})
sch.tile("w", {"w1": 4})
sch.tile("f", {"f1": 16, "f2": 4})
sch.interchange(["h", "w", "b", "f", "r", "s", "c", "h1", "w1", "f1", "f2"])
sch.vectorize(["f2"])
sch.unroll({"c": 3})
sch.gpu_block(["h", "w"])
sch.gpu_thread(["h1", "w1", "f1"])
sched = sch.schedule()
print(sched)


comp = impl.get_compiler(
    shared_lib=True,
    dump_file="gpu_conv2d_tvm",
    target=gpu,
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)

evaluator = module.get_evaluator(validate=True)
results, code, error = evaluator.evaluate()
print(f"CODE: {code}")

# CHECK:       O = sch.get_sblock("O")
# CHECK-NEXT:  b, h, w, f, r, s, c, = sch.get_loops(O)
# CHECK-NEXT:  h, h1, = sch.split(h, factors=[None, 4])
# CHECK-NEXT:  w, w1, = sch.split(w, factors=[None, 4])
# CHECK-NEXT:  f, f1, f2, = sch.split(f, factors=[None, 4, 4])
# CHECK-NEXT:  c, __u_c, = sch.split(c, factors=[None, 3])
# CHECK-NEXT:  sch.reorder(h, w, b, f, r, s, c, __u_c, h1, w1, f1, f2)
# CHECK-NEXT:  sch.bind(h, 'blockIdx.x')
# CHECK-NEXT:  sch.bind(w, 'blockIdx.y')
# CHECK-NEXT:  sch.bind(h1, 'threadIdx.x')
# CHECK-NEXT:  sch.bind(w1, 'threadIdx.y')
# CHECK-NEXT:  sch.bind(f1, 'threadIdx.z')
# CHECK-NEXT:  sch.unroll(__u_c)
# CHECK-NEXT:  sch.vectorize(f2)
# CHECK-NEXT:  sch = decompose_reduction_initializers(sch)
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def conv2d_nhwc_gpu(_0: T.Buffer((1, 12, 12, 3), "float32"), _1: T.Buffer((3, 3, 3, 16), "float32"), O: T.Buffer((1, 10, 10, 16), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for b, h, w, f, r, s, c in T.grid(1, 10, 10, 16, 3, 3, 3):
# CHECK-NEXT:              with T.sblock("O"):
# CHECK-NEXT:                  v_b, v_h, v_w, v_f, v_r, v_s, v_c = T.axis.remap("SSSSRRR", [b, h, w, f, r, s, c])
# CHECK-NEXT:                  T.reads(_0[v_b, v_h + v_r, v_w + v_s, v_c], _1[v_r, v_s, v_c, v_f])
# CHECK-NEXT:                  T.writes(O[v_b, v_h, v_w, v_f])
# CHECK-NEXT:                  with T.init():
# CHECK-NEXT:                      O[v_b, v_h, v_w, v_f] = T.float32(0.0)
# CHECK-NEXT:                  O[v_b, v_h, v_w, v_f] = O[v_b, v_h, v_w, v_f] + _0[v_b, v_h + v_r, v_w + v_s, v_c] * _1[v_r, v_s, v_c, v_f]
# CHECK-NEXT:  O = sch.get_sblock("O")
# CHECK-NEXT:  b, h, w, f, r, s, c, = sch.get_loops(O)
# CHECK-NEXT:  h, h1, = sch.split(h, factors=[None, 4])
# CHECK-NEXT:  w, w1, = sch.split(w, factors=[None, 4])
# CHECK-NEXT:  f, f1, f2, = sch.split(f, factors=[None, 4, 4])
# CHECK-NEXT:  c, __u_c, = sch.split(c, factors=[None, 3])
# CHECK-NEXT:  sch.reorder(h, w, b, f, r, s, c, __u_c, h1, w1, f1, f2)
# CHECK-NEXT:  sch.bind(h, 'blockIdx.x')
# CHECK-NEXT:  sch.bind(w, 'blockIdx.y')
# CHECK-NEXT:  sch.bind(h1, 'threadIdx.x')
# CHECK-NEXT:  sch.bind(w1, 'threadIdx.y')
# CHECK-NEXT:  sch.bind(f1, 'threadIdx.z')
# CHECK-NEXT:  sch.unroll(__u_c)
# CHECK-NEXT:  sch.vectorize(f2)
# CHECK-NEXT:  sch = decompose_reduction_initializers(sch)
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def conv2d_nhwc_gpu(_0: T.Buffer((1, 12, 12, 3), "float32"), _1: T.Buffer((3, 3, 3, 16), "float32"), O: T.Buffer((1, 10, 10, 16), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for h_0 in T.thread_binding(3, thread="blockIdx.x"):
# CHECK-NEXT:              for w_0 in T.thread_binding(3, thread="blockIdx.y"):
# CHECK-NEXT:                  for b, f_0 in T.grid(1, 1):
# CHECK-NEXT:                      for h_1_init in T.thread_binding(4, thread="threadIdx.x"):
# CHECK-NEXT:                          for w_1_init in T.thread_binding(4, thread="threadIdx.y"):
# CHECK-NEXT:                              for f_1_init in T.thread_binding(4, thread="threadIdx.z"):
# CHECK-NEXT:                                  for f_2_init in T.vectorized(4):
# CHECK-NEXT:                                      with T.sblock("O_init"):
# CHECK-NEXT:                                          v_b = T.axis.spatial(1, b)
# CHECK-NEXT:                                          v_h = T.axis.spatial(10, h_0 * 4 + h_1_init)
# CHECK-NEXT:                                          v_w = T.axis.spatial(10, w_0 * 4 + w_1_init)
# CHECK-NEXT:                                          v_f = T.axis.spatial(16, f_0 * 16 + f_1_init * 4 + f_2_init)
# CHECK-NEXT:                                          T.where(w_0 * 4 + w_1_init < 10 and h_0 * 4 + h_1_init < 10)
# CHECK-NEXT:                                          T.reads()
# CHECK-NEXT:                                          T.writes(O[v_b, v_h, v_w, v_f])
# CHECK-NEXT:                                          O[v_b, v_h, v_w, v_f] = T.float32(0.0)
# CHECK-NEXT:                      for r, s, c_0 in T.grid(3, 3, 1):
# CHECK-NEXT:                          for c_1 in T.unroll(3):
# CHECK-NEXT:                              for h_1 in T.thread_binding(4, thread="threadIdx.x"):
# CHECK-NEXT:                                  for w_1 in T.thread_binding(4, thread="threadIdx.y"):
# CHECK-NEXT:                                      for f_1 in T.thread_binding(4, thread="threadIdx.z"):
# CHECK-NEXT:                                          for f_2 in T.vectorized(4):
# CHECK-NEXT:                                              with T.sblock("O_update"):
# CHECK-NEXT:                                                  v_b = T.axis.spatial(1, b)
# CHECK-NEXT:                                                  v_h = T.axis.spatial(10, h_0 * 4 + h_1)
# CHECK-NEXT:                                                  v_w = T.axis.spatial(10, w_0 * 4 + w_1)
# CHECK-NEXT:                                                  v_f = T.axis.spatial(16, f_0 * 16 + f_1 * 4 + f_2)
# CHECK-NEXT:                                                  v_r, v_s = T.axis.remap("RR", [r, s])
# CHECK-NEXT:                                                  v_c = T.axis.reduce(3, c_0 * 3 + c_1)
# CHECK-NEXT:                                                  T.where(h_0 * 4 + h_1 < 10 and w_0 * 4 + w_1 < 10)
# CHECK-NEXT:                                                  T.reads(O[v_b, v_h, v_w, v_f], _0[v_b, v_h + v_r, v_w + v_s, v_c], _1[v_r, v_s, v_c, v_f])
# CHECK-NEXT:                                                  T.writes(O[v_b, v_h, v_w, v_f])
# CHECK-NEXT:                                                  O[v_b, v_h, v_w, v_f] = O[v_b, v_h, v_w, v_f] + _0[v_b, v_h + v_r, v_w + v_s, v_c] * _1[v_r, v_s, v_c, v_f]
# CHECK-NEXT:  CODE: 0
