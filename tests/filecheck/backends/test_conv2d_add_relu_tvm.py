# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend

lv = O.tensor(shape=(1, 224, 224, 3), dtype='float32')
lv1 = O.tensor(shape=(11, 11, 3, 64), dtype='float32')
lv2 = O.tensor(shape=(1, 1, 1, 64), dtype='float32')

with O.graph(name='fused_conv2d_add_relu') as gb:
    lv_p = O.pad2d(lv, padding=(2, 2, 2, 2), name="pad")
    lv_1 = O.conv2d(lv_p, lv1, stride=(4, 4), name = "conv2d")
    lv2_1 = O.add(lv_1, lv2, name = "add")
    O.relu(lv2_1)

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler(default_node = "conv2d")
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="conv2d_add_relu_tvm",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")
# CHECK: graph:
# CHECK-NEXT:   name: fused_conv2d_add_relu
# CHECK-NEXT:   inputs:
# CHECK-NEXT:   - %0 : 1x224x224x3xfloat32
# CHECK-NEXT:   - %1 : 11x11x3x64xfloat32
# CHECK-NEXT:   - %2 : 1x1x1x64xfloat32
# CHECK-NEXT:   outputs:
# CHECK-NEXT:   - %6 : 1x55x55x64xfloat32
# CHECK-NEXT:   nodes:
# CHECK-NEXT:   - %3: pad2d(%0, padding={-3: (2, 2), -2: (2, 2)}, constant_value=0) {name = 'pad'} : [1x224x224x3xfloat32] -> [1x228x228x3xfloat32]
# CHECK-NEXT:   - %4: conv2d(%3, %1, stride=(4, 4)) {name = 'conv2d'} : [1x228x228x3xfloat32, 11x11x3x64xfloat32] -> [1x55x55x64xfloat32]
# CHECK-NEXT:   - %5: add(%4, %2) {name = 'add'} : [1x55x55x64xfloat32, 1x1x1x64xfloat32] -> [1x55x55x64xfloat32]
# CHECK-NEXT:   - %6: relu(%5) : [1x55x55x64xfloat32] -> [1x55x55x64xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT: # from tvm.script import ir as I
# CHECK-NEXT: # from tvm.script import tir as T
# CHECK-NEXT:  
# CHECK-NEXT: @I.ir_module
# CHECK-NEXT: class Module:
# CHECK-NEXT:     @T.prim_func
# CHECK-NEXT:     def main(_0: T.Buffer((1, 224, 224, 3), "float32"), _1: T.Buffer((11, 11, 3, 64), "float32"), _2: T.Buffer((1, 1, 1, 64), "float32"), T_reshape: T.Buffer((1, 55, 55, 64), "float32")):
# CHECK-NEXT:         T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
# CHECK-NEXT:         pad = T.allocate([193600], "float32", "global")
# CHECK-NEXT:         conv2d = T.allocate([193600], "float32", "global")
# CHECK-NEXT:         pad_1 = T.Buffer((154587,), data=pad)
# CHECK-NEXT:         for i1, i2, i3 in T.grid(227, 227, 3):
# CHECK-NEXT:             cse_var_1: T.int32 = i2 * 3
# CHECK-NEXT:             _0_1 = T.Buffer((150528,), data=_0.data)
# CHECK-NEXT:             pad_1[i1 * 681 + cse_var_1 + i3] = T.if_then_else(2 <= i1 and i1 < 226 and 2 <= i2 and i2 < 226, _0_1[i1 * 672 + cse_var_1 + i3 - 1350], T.float32(0.0))
# CHECK-NEXT:         for h, w, f in T.grid(55, 55, 64):
# CHECK-NEXT:             conv2d_1 = T.Buffer((193600,), data=conv2d)
# CHECK-NEXT:             conv2d_1[h * 3520 + w * 64 + f] = T.float32(0.0)
# CHECK-NEXT:             for r, s, c in T.grid(11, 11, 3):
# CHECK-NEXT:                 cse_var_2: T.int32 = h * 3520 + w * 64 + f
# CHECK-NEXT:                 _1_1 = T.Buffer((23232,), data=_1.data)
# CHECK-NEXT:                 conv2d_1[cse_var_2] = conv2d_1[cse_var_2] + pad_1[h * 2724 + r * 681 + w * 12 + s * 3 + c] * _1_1[r * 2112 + s * 192 + c * 64 + f]
# CHECK-NEXT:         for ax1, ax2, ax3 in T.grid(55, 55, 64):
# CHECK-NEXT:             pad_2 = T.Buffer((193600,), data=pad)
# CHECK-NEXT:             _2_1 = T.Buffer((64,), data=_2.data)
# CHECK-NEXT:             pad_2[ax1 * 3520 + ax2 * 64 + ax3] = _2_1[ax3]
# CHECK-NEXT:         for ax0 in range(193600):
# CHECK-NEXT:             conv2d_1 = T.Buffer((193600,), data=conv2d)
# CHECK-NEXT:             conv2d_2 = T.Buffer((193600,), data=conv2d)
# CHECK-NEXT:             pad_2 = T.Buffer((193600,), data=pad)
# CHECK-NEXT:             conv2d_1[ax0] = conv2d_2[ax0] + pad_2[ax0]
# CHECK-NEXT:         conv2d_1 = T.Buffer((193600,), data=conv2d)
# CHECK-NEXT:         for i in range(193600):
# CHECK-NEXT:             conv2d_2 = T.Buffer((193600,), data=conv2d)
# CHECK-NEXT:             conv2d_1[i] = T.max(T.float32(0.0), conv2d_2[i])
# CHECK-NEXT:         for ax1, ax2, ax3 in T.grid(55, 55, 64):
# CHECK-NEXT:             cse_var_3: T.int32 = ax1 * 3520 + ax2 * 64 + ax3
# CHECK-NEXT:             T_reshape_1 = T.Buffer((193600,), data=T_reshape.data)
# CHECK-NEXT:             T_reshape_1[cse_var_3] = conv2d_1[cse_var_3]
# CHECK-NEXT: O = obj['conv2d']
# CHECK-NEXT: b, h, w, f, = O.op.axis
# CHECK-NEXT: r, s, c, = O.op.reduce_axis
# CHECK-NEXT: sch[O].reorder(b, h, w, f, r, s, c)
# CHECK-NEXT:  
# CHECK-NEXT: # from tvm.script import ir as I
# CHECK-NEXT: # from tvm.script import tir as T
# CHECK-NEXT:  
# CHECK-NEXT: @I.ir_module
# CHECK-NEXT: class Module:
# CHECK-NEXT:     @T.prim_func
# CHECK-NEXT:     def main(_0: T.Buffer((1, 224, 224, 3), "float32"), _1: T.Buffer((11, 11, 3, 64), "float32"), _2: T.Buffer((1, 1, 1, 64), "float32"), T_reshape: T.Buffer((1, 55, 55, 64), "float32")):
# CHECK-NEXT:         T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
# CHECK-NEXT:         pad = T.allocate([193600], "float32", "global")
# CHECK-NEXT:         conv2d = T.allocate([193600], "float32", "global")
# CHECK-NEXT:         pad_1 = T.Buffer((154587,), data=pad)
# CHECK-NEXT:         for i1, i2, i3 in T.grid(227, 227, 3):
# CHECK-NEXT:             cse_var_1: T.int32 = i2 * 3
# CHECK-NEXT:             _0_1 = T.Buffer((150528,), data=_0.data)
# CHECK-NEXT:             pad_1[i1 * 681 + cse_var_1 + i3] = T.if_then_else(2 <= i1 and i1 < 226 and 2 <= i2 and i2 < 226, _0_1[i1 * 672 + cse_var_1 + i3 - 1350], T.float32(0.0))
# CHECK-NEXT:         for h, w, f in T.grid(55, 55, 64):
# CHECK-NEXT:             conv2d_1 = T.Buffer((193600,), data=conv2d)
# CHECK-NEXT:             conv2d_1[h * 3520 + w * 64 + f] = T.float32(0.0)
# CHECK-NEXT:             for r, s, c in T.grid(11, 11, 3):
# CHECK-NEXT:                 cse_var_2: T.int32 = h * 3520 + w * 64 + f
# CHECK-NEXT:                 _1_1 = T.Buffer((23232,), data=_1.data)
# CHECK-NEXT:                 conv2d_1[cse_var_2] = conv2d_1[cse_var_2] + pad_1[h * 2724 + r * 681 + w * 12 + s * 3 + c] * _1_1[r * 2112 + s * 192 + c * 64 + f]
# CHECK-NEXT:         for ax1, ax2, ax3 in T.grid(55, 55, 64):
# CHECK-NEXT:             pad_2 = T.Buffer((193600,), data=pad)
# CHECK-NEXT:             _2_1 = T.Buffer((64,), data=_2.data)
# CHECK-NEXT:             pad_2[ax1 * 3520 + ax2 * 64 + ax3] = _2_1[ax3]
# CHECK-NEXT:         for ax0 in range(193600):
# CHECK-NEXT:             conv2d_1 = T.Buffer((193600,), data=conv2d)
# CHECK-NEXT:             conv2d_2 = T.Buffer((193600,), data=conv2d)
# CHECK-NEXT:             pad_2 = T.Buffer((193600,), data=pad)
# CHECK-NEXT:             conv2d_1[ax0] = conv2d_2[ax0] + pad_2[ax0]
# CHECK-NEXT:         conv2d_1 = T.Buffer((193600,), data=conv2d)
# CHECK-NEXT:         for i in range(193600):
# CHECK-NEXT:             conv2d_2 = T.Buffer((193600,), data=conv2d)
# CHECK-NEXT:             conv2d_1[i] = T.max(T.float32(0.0), conv2d_2[i])
# CHECK-NEXT:         for ax1, ax2, ax3 in T.grid(55, 55, 64):
# CHECK-NEXT:             cse_var_3: T.int32 = ax1 * 3520 + ax2 * 64 + ax3
# CHECK-NEXT:             T_reshape_1 = T.Buffer((193600,), data=T_reshape.data)
# CHECK-NEXT:             T_reshape_1[cse_var_3] = conv2d_1[cse_var_3]
# CHECK-NEXT: CODE: 0

