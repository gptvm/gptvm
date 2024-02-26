#include "gptvm/compiler/dialect/actor/ir/actor_ops.h"
#include "gptvm/compiler/dialect/actor/transforms/passes.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "convert-domain-to-actor"

using namespace gptvm::actor;
using namespace mlir;

namespace {

#define GEN_PASS_DEF_CONVERTDOMAINTOACTOR
#include "gptvm/compiler/dialect/actor/transforms/passes.h.inc"

class ConvertDomainToActor
    : public impl::ConvertDomainToActorBase<ConvertDomainToActor> {
public:
  void runOnOperation() override {}
};

} // namespace

std::unique_ptr<Pass> gptvm::actor::createConvertDomainToActorPass() {
  return std::make_unique<ConvertDomainToActor>();
}
