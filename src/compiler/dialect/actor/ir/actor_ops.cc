#include "gptvm/compiler/dialect/actor/ir/actor_ops.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include "gptvm/compiler/dialect/actor/ir/actor_dialect.cc.inc"

using namespace mlir;
using namespace gptvm::actor;

void gptvm::actor::ActorDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "gptvm/compiler/dialect/actor/ir/actor_ops.cc.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "gptvm/compiler/dialect/actor/ir/actor_types.cc.inc"
      >();
}

#define GET_OP_CLASSES
#include "gptvm/compiler/dialect/actor/ir/actor_ops.cc.inc"

#define GET_TYPEDEF_CLASSES
#include "gptvm/compiler/dialect/actor/ir/actor_types.cc.inc"
