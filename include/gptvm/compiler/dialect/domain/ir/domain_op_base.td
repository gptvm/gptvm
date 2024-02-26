include "mlir/IR/OpBase.td"

def Domain_Dialect : Dialect {
  let name = "domain";

  let cppNamespace = "gptvm::domain";
}

class Domain_Op<string mnemonic, list<Trait> traits = []> :
    Op<Domain_Dialect, mnemonic, traits> {
}
