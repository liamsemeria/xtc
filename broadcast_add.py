from xdsl.dialects import memref, arith, linalg, builtin, func
from xdsl.builder import ImplicitBuilder
from xdsl.ir import Block, Region, BlockArgument
from xdsl.dialects.builtin import (
    ModuleOp, AffineMapAttr, ArrayAttr, AffineMap, 
    IntegerAttr, i64, f32, f64, StringAttr, MemRefType
)
from typing import Sequence, cast, override
from functools import reduce
import operator

def mulall(xs):
    """Multiply all elements in a list"""
    return reduce(operator.mul, xs, 1)

class MlirOperatorBroadcastAdd:
    """
    Broadcast add operation that collapses to enable vectorization.
    
    Strategy:
    - Collapse to (outer_dims, inner_dim) preserving broadcast structure
    - Example: (1,50,50,64) + (1,1,1,64) -> collapse to (2500,64) + (1,64)
    - Vectorize over inner_dim (channels)
    """
    
    DEFAULT_NAME = "broadcast_add"
    
    def __init__(self, shape_a, shape_b, broadcast_dims_b, dtype="float32"):
        """
        Args:
            shape_a: tuple like (1, 50, 50, 64)
            shape_b: tuple like (1, 1, 1, 64)
            broadcast_dims_b: list of dims that are size 1 in B, e.g., [0,1,2]
            dtype: "float32" or "float64"
        """
        self.shape_a = shape_a
        self.shape_b = shape_b
        self.broadcast_dims_b = broadcast_dims_b
        self.dtype = dtype
        self.name = self.DEFAULT_NAME
        
        # Figure out collapsed shapes
        # Keep innermost dimension separate for vectorization
        self.ndims = len(shape_a)
        self.inner_dim = shape_a[-1]
        self.outer_size_a = mulall(shape_a[:-1])
        self.outer_size_b = mulall(shape_b[:-1])
        
        self.collapsed_shape_a = [self.outer_size_a, self.inner_dim]
        self.collapsed_shape_b = [self.outer_size_b, self.inner_dim]
    
    def generate_op(
        self, block: Block | None = None, args: Sequence[BlockArgument] = []
    ) -> tuple[Block, dict]:
        """
        Generate the broadcast add operation.
        
        Args:
            block: Optional block to add operations to
            args: [input_a, input_b, output] memrefs
        
        Returns:
            (block, attributes dict)
        """
        elt_type = {"float32": f32, "float64": f64}[self.dtype]
        
        if block is None:
            ops_types = [
                MemRefType(elt_type, list(self.shape_a)),
                MemRefType(elt_type, list(self.shape_b)),
                MemRefType(elt_type, list(self.shape_a))  # output same as A
            ]
            block = Block(arg_types=ops_types)
            args = block.args
        
        assert len(args) == 3
        assert all(isinstance(arg.type, MemRefType) for arg in args)
        
        with ImplicitBuilder(block):
            # Build reassociation arrays for collapsing
            # Collapse all dims except last: [[0,1,...,n-2], [n-1]]
            outer_indices = list(range(self.ndims - 1))
            inner_indices = [self.ndims - 1]
            
            inp_a_reassociation = builtin.ArrayAttr([
                builtin.ArrayAttr([builtin.IntegerAttr(x, i64) for x in outer_indices]),
                builtin.ArrayAttr([builtin.IntegerAttr(x, i64) for x in inner_indices])
            ])
            
            inp_b_reassociation = builtin.ArrayAttr([
                builtin.ArrayAttr([builtin.IntegerAttr(x, i64) for x in outer_indices]),
                builtin.ArrayAttr([builtin.IntegerAttr(x, i64) for x in inner_indices])
            ])
            
            out_reassociation = builtin.ArrayAttr([
                builtin.ArrayAttr([builtin.IntegerAttr(x, i64) for x in outer_indices]),
                builtin.ArrayAttr([builtin.IntegerAttr(x, i64) for x in inner_indices])
            ])
            
            # Collapse input A
            inp_a = memref.CollapseShapeOp(
                operands=[args[0]],
                properties=dict(reassociation=inp_a_reassociation),
                result_types=[MemRefType(elt_type, self.collapsed_shape_a)],
            )
            
            # Collapse input B
            inp_b = memref.CollapseShapeOp(
                operands=[args[1]],
                properties=dict(reassociation=inp_b_reassociation),
                result_types=[MemRefType(elt_type, self.collapsed_shape_b)],
            )
            
            # Collapse output
            out = memref.CollapseShapeOp(
                operands=[args[2]],
                properties=dict(reassociation=out_reassociation),
                result_types=[MemRefType(elt_type, self.collapsed_shape_a)],
            )
            
            # Create affine maps for broadcast pattern
            # A: (i, j) -> (i, j)  - normal access
            # B: (i, j) -> (0, j)  - broadcast over i dimension
            # C: (i, j) -> (i, j)  - normal access
            
            map_a = AffineMapAttr(
                AffineMap.from_callable(lambda i, j: (i, j))
            )
            
            map_b = AffineMapAttr(
                AffineMap.from_callable(lambda i, j: (0, j))
            )
            
            map_c = AffineMapAttr(
                AffineMap.from_callable(lambda i, j: (i, j))
            )
            
            # Iterator types: both parallel
            iterator_types = [
                StringAttr("parallel"),
                StringAttr("parallel")
            ]
            
            # Create the body block for the add operation
            body_block = Block(arg_types=[elt_type, elt_type, elt_type])
            
            with ImplicitBuilder(body_block):
                a_elem, b_elem, c_elem = body_block.args
                add_result = arith.AddfOp(a_elem, b_elem)
                linalg.YieldOp(add_result)
            
            # Create the linalg.generic operation
            add_op = linalg.GenericOp(
                inputs=(inp_a.results[0], inp_b.results[0]),
                outputs=(out.results[0],),
                body=Region([body_block]),
                indexing_maps=[map_a, map_b, map_c],
                iterator_types=iterator_types,
            )
        
        # Add operation ID for tracking
        from xdsl.dialects.builtin import UnitAttr
        add_node_id = f"{self.name}"
        add_op.attributes[f"__xtc_id_{add_node_id}_"] = UnitAttr()
        
        attrs = {
            "nodes_map": {
                add_node_id: add_op,
            },
            "dims_sizes": {
                "outer": self.outer_size_a,
                "inner": self.inner_dim
            },
        }
        
        return block, attrs


# Example usage
def create_broadcast_add_example():
    """Create an example broadcast add: (1,50,50,64) + (1,1,1,64)"""
    
    op = MlirOperatorBroadcastAdd(
        shape_a=(1, 50, 50, 64),
        shape_b=(1, 1, 1, 64),
        broadcast_dims_b=[0, 1, 2],
        dtype="float32"
    )
    
    block, attrs = op.generate_op()
    
    # Wrap in a function
    func_type = builtin.FunctionType.from_lists(
        [arg.type for arg in block.args],
        []
    )
    
    func_op = func.FuncOp.from_region(
        "broadcast_add_collapsed",
        [arg.type for arg in block.args],
        [],
        Region(block)
    )
    
    # Wrap in module
    module_body = Block()
    module_body.add_op(func_op)
    module = ModuleOp(Region(module_body))
    
    return module


if __name__ == "__main__":
    print("=== Broadcast Add with Collapsed Shapes ===")
    print("Strategy: (1,50,50,64) + (1,1,1,64)")
    print("Collapsed: (2500,64) + (1,64)")
    print("Vectorizes over inner dimension (64 channels)")
    print()
    
    module = create_broadcast_add_example()
    print(module)
