include "mlir/Pass/PassBase.td"

def ConvertDomainToActor : Pass<"convert-domain-to-actor", "func::FuncOp"> {
  let constructor = "createConvertDomainToActorPass()";
}
