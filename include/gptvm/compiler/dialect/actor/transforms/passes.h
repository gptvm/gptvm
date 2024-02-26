#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace gptvm {
namespace actor {

#define GEN_PASS_DECL
#include "gptvm/compiler/dialect/actor/transforms/passes.h.inc"

std::unique_ptr<mlir::Pass> createConvertDomainToActorPass();

#define GEN_PASS_REGISTRATION
#include "gptvm/compiler/dialect/actor/transforms/passes.h.inc"

} // namespace actor
} // namespace gptvm
