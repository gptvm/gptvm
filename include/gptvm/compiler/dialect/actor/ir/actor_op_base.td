include "mlir/IR/OpBase.td"

def Actor_Dialect : Dialect {
  let name = "actor";

  let cppNamespace = "gptvm::actor";
}

class Actor_Op<string mnemonic, list<Trait> traits = []>
    : Op<Actor_Dialect, mnemonic, traits> {
}

class Actor_Type<string name, list<Trait> traits = []>
    : TypeDef<Actor_Dialect, name, traits> {
  let mnemonic = name;
}

def Actor_TaskType : Actor_Type<"Task">;

def Actor_ObjectType : Actor_Type<"Object">;
