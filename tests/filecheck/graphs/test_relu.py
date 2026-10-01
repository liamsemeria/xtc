# RUN: python %s 2>&1 | filecheck %s

import xtc.graphs.xtc.op as O

x = O.tensor()

with O.graph(name="relu") as gb:
    O.relu(x)

graph = gb.graph
print(graph)

import xtc.graphs.xtc.ty as T

inp_types = [
    T.TensorType((1, 5, 6), "float32"),
]
out_types = graph.forward_types(inp_types)
print(out_types)

from xtc.utils.numpy import np_init

inps = [T.Tensor(np_init(t.constant_shape, t.constant_dtype)) for t in inp_types]
print(f"Inputs: {inps}")
outs = graph.forward(inps)
print(f"Outputs: {outs}")

# CHECK:       graph:
# CHECK-NEXT:    name: relu
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %1
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %1: relu(%0)
# CHECK-NEXT:  
# CHECK-NEXT:  [1x5x6xfloat32]
# CHECK-NEXT:  Inputs: [Tensor(type=1x5x6xfloat32, data=-1 -0.875 -0.75 -0.625...0.125 0.25 0.375 0.5)]
# CHECK-NEXT:  Outputs: [Tensor(type=1x5x6xfloat32, data=0 0 0 0...0.125 0.25 0.375 0.5)]
