include "mlir/Pass/PassBase.td"

def DomainInit : Pass<"domain-init", "func::FuncOp"> {
  let constructor = "createDomainInitPass()";
}

def TPGroupSearchPass : Pass<"tensor-parallel-group-search-pass", "func::FuncOp"> {
  let constructor = "createTPGroupSearchPass()";
}
