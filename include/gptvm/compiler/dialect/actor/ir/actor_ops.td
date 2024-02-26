include "actor_op_base.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def Actor_TaskCreateOp : Actor_Op<"task.create_model"> {
    let arguments = (ins StrAttr: $file,
                         StrAttr: $device_type,
                         OptionalAttr<StrAttr>: $backend);
    let results = (outs Actor_TaskType:$results);
}

def Actor_TaskLaunchOp : Actor_Op<"task.launch"> {
    let arguments = (ins Actor_TaskType: $task,
                         Variadic<Actor_ObjectType>: $args);
    let results = (outs Actor_TaskType:$results);
}
