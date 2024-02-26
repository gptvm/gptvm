#include "gptvm/compiler/dialect/domain/transforms/passes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace gptvm;
using namespace gptvm::domain;
using namespace mlir;

void gptvm::domain::registerDomainPasses() {
  // Generated.
  registerPasses();
}
