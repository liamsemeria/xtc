# RUN: python %s 2>&1 | filecheck %s

import xtc.graphs.xtc.op as O

x = O.tensor()
y = O.tensor()

with O.graph(name="add") as gb:
    O.add(x, y)

graph = gb.graph
print(graph)

import xtc.graphs.xtc.ty as T

inp_types = [
    T.TensorType((1, 5, 6), "float32"),
    T.TensorType((6, 5, 1), "float32"),
]
out_types = graph.forward_types(inp_types)
print(out_types)

from xtc.utils.numpy import np_init

inps = [T.Tensor(np_init(t.constant_shape, t.constant_dtype)-5) for t in inp_types]
print(f"Inputs: {inps}")
outs = graph.forward(inps)
print(f"Outputs: {outs}")

inp_types = [
    T.TensorType((1, 5, 6), "float32"),
    T.TensorType((6,), "float32"),
]
out_types = graph.forward_types(inp_types)
print(out_types)

from xtc.utils.numpy import np_init

inps = [T.Tensor(np_init(t.constant_shape, t.constant_dtype)-5) for t in inp_types]
print(f"Inputs: {inps}")
outs = graph.forward(inps)
print(f"Outputs: {outs}")

# CHECK:       graph:
# CHECK-NEXT:    name: add
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0
# CHECK-NEXT:    - %1
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: add(%0, %1)
# CHECK-NEXT:  
# CHECK-NEXT:  [6x5x6xfloat32]
# CHECK-NEXT:  Inputs: [Tensor(type=1x5x6xfloat32, data=-4 -3 -2 -1...4 -4 -3 -2), Tensor(type=6x5x1xfloat32, data=-4 -3 -2 -1...4 -4 -3 -2)]
# CHECK-NEXT:  Outputs: [Tensor(type=6x5x6xfloat32, data=-8 -7 -6 -5...2 -6 -5 -4)] 
# CHECK-NEXT:  [1x5x6xfloat32]
# CHECK-NEXT:  Inputs: [Tensor(type=1x5x6xfloat32, data=-4 -3 -2 -1...4 -4 -3 -2), Tensor(type=6xfloat32, data=-4 -3 -2 -1 0 1)]
# CHECK-NEXT:  Outputs: [Tensor(type=1x5x6xfloat32, data=-8 -6 -4 -2...2 -5 -3 -1)]

