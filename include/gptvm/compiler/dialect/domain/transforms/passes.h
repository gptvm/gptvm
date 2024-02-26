#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace gptvm {
namespace domain {

#define GEN_PASS_DECL
#include "gptvm/compiler/dialect/domain/transforms/passes.h.inc"

std::unique_ptr<mlir::Pass> createDomainInitPass();
std::unique_ptr<mlir::Pass> createTPGroupSearchPass();

void registerDomainPasses();

#define GEN_PASS_REGISTRATION
#include "gptvm/compiler/dialect/domain/transforms/passes.h.inc"

} // namespace domain
} // namespace gptvm
