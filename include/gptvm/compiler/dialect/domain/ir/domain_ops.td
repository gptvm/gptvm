include "domain_op_base.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def Domain_RegionOp : Domain_Op<"dispatch.region", [
  SingleBlockImplicitTerminator<"gptvm::domain::ReturnOp">]> {
    let summary = [{a group of ops}];
    let description = [{}];
    let arguments = (ins Variadic<AnyType>: $inputs,
                         OptionalAttr<StrAttr> :$device_type,
                         OptionalAttr<I64Attr>: $device_id);
    let results = (outs Variadic<AnyType>:$results);
    let regions = (region AnyRegion:$body);
    let hasCustomAssemblyFormat = 1;
    let builders = [
      OpBuilder<(ins
        "mlir::ArrayRef<mlir::Type>":$inType,
        "mlir::ArrayRef<mlir::Value>":$inputs,
        "std::string": $device_type,
        "int": $device_id)>
    ];
}

def Domain_ReturnOp : Domain_Op<"return", [Pure, ReturnLike, Terminator]> {
  let summary = [{}];
  let description = [{}];
  let arguments = (ins 
      Variadic<AnyType>:$operands);

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt);}]>
  ];
}
