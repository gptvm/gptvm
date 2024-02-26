#include "llvm-project/mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm-project/mlir/include/mlir/IR/Block.h"
#include "llvm-project/mlir/include/mlir/IR/Location.h"
#include "llvm-project/mlir/include/mlir/IR/Operation.h"
#include "llvm-project/mlir/include/mlir/IR/Region.h"
#include "llvm-project/mlir/include/mlir/IR/Visitors.h"
#include "llvm-project/mlir/include/mlir/Support/LogicalResult.h"
#include "gptvm/compiler/dialect/domain/ir/domain_ops.h"
#include "gptvm/compiler/dialect/domain/transforms/passes.h"

#include "mlir/IR/Dominance.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/TopologicalSortUtils.h"

#include "onnx-mlir/src/Dialect/ONNX/ONNXDialect.hpp"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Debug.h"
#include <cstdint>

#define DEBUG_TYPE "domain-init"

using namespace gptvm::domain;
using namespace mlir;

namespace gptvm {
namespace domain {} // namespace domain
} // namespace gptvm

namespace {
static const char kRootOpAttr[] = "__root_op__";
static const char kDeviceOpAttr[] = "device_attr";
static const char kDeviceIdAttr[] = "device_id";
static const char kGPUDeviceType[] = "GPU";
static const int kDeivceID = 0;

#define GEN_PASS_DEF_DOMAININIT
#include "gptvm/compiler/dialect/domain/transforms/passes.h.inc"

static int64_t getRootNum(mlir::Operation *op) {
  return op->getAttrOfType<mlir::IntegerAttr>(kRootOpAttr).getInt();
}

static bool hasRootOpAttribute(mlir::Operation *op) {
  return static_cast<bool>(
      op->template getAttrOfType<mlir::IntegerAttr>(kRootOpAttr));
}

static void setRootAttribute(mlir::MLIRContext *context, mlir::Operation *op,
                             int64_t numRoots) {
  op->setAttr(kRootOpAttr, mlir::IntegerAttr::get(
                               mlir::IntegerType::get(context, 64), numRoots));
}

FailureOr<gptvm::domain::RegionOp>
appendDispatchRegionResult(mlir::RewriterBase &rewriter,
                           gptvm::domain::RegionOp regionOp,
                           mlir::ArrayRef<Value> results) {
  mlir::SmallVector<mlir::Type> resultTypes(regionOp.getResultTypes().begin(),
                                            regionOp.getResultTypes().end());
  auto returnOp = cast<ReturnOp>(regionOp.getBody().front().getTerminator());
  mlir::SmallVector<Value> returnValues(returnOp.getOperands().begin(),
                                        returnOp.getOperands().end());
  for (auto [index, result] : llvm::enumerate(results)) {
    resultTypes.push_back(result.getType());
    returnValues.push_back(result);
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(regionOp);
  SmallVector<mlir::Value> refRegionValues;
  for (auto input : regionOp.getInputs()) {
    refRegionValues.push_back(input);
  }
  auto newRegionOp = rewriter.create<gptvm::domain::RegionOp>(
      regionOp->getLoc(), resultTypes, refRegionValues,
      std::string(kGPUDeviceType), kDeivceID);
  rewriter.inlineRegionBefore(regionOp.getBody(), newRegionOp.getBody(),
                              newRegionOp.getBody().begin());
  rewriter.replaceOp(
      regionOp, newRegionOp.getResults().take_front(regionOp->getNumResults()));
  auto newRegionReturnOp =
      cast<ReturnOp>(newRegionOp.getBody().front().getTerminator());
  rewriter.setInsertionPoint(newRegionReturnOp);
  rewriter.replaceOpWithNewOp<ReturnOp>(newRegionReturnOp, returnValues);
  return newRegionOp;
}

mlir::FailureOr<gptvm::domain::RegionOp>
movePrecedingOpsIntoRegion(mlir::RewriterBase &rewriter,
                           mlir::ArrayRef<Operation *> producers,
                           gptvm::domain::RegionOp regionOp) {
  SmallVector<Value> replacedValues;
  SmallVector<Value> yieldedResults;
  Block &body = regionOp.getBody().front();
  for (mlir::Operation *produce : producers) {
    // Clone op into dispatch region
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&body);
    mlir::Operation *clonedOp = rewriter.clone(*produce);

    bool hasUsesOutsideOfRegion = false;
    for (auto [index, result] : llvm::enumerate(produce->getResults())) {
      hasUsesOutsideOfRegion =
          llvm::any_of(result.getUses(), [&](mlir::OpOperand &use) {
            mlir::Operation *user = use.getOwner();
            return !regionOp->isProperAncestor(user);
          });
      if (hasUsesOutsideOfRegion) {
        replacedValues.push_back(result);
        yieldedResults.push_back(clonedOp->getResult(index));
      }
    }
    rewriter.replaceOpWithinBlock(produce, clonedOp->getResults(), &body);
  }

  mlir::FailureOr<gptvm::domain::RegionOp> newRegionOp =
      appendDispatchRegionResult(rewriter, regionOp, yieldedResults);
  mlir::ValueRange replacements = newRegionOp->getResults().take_back();
  for (auto [index, replaceVal] : llvm::enumerate(replacedValues)) {
    rewriter.replaceAllUsesWith(replaceVal, replacements[index]);
  }

  for (auto produce : llvm::reverse(producers)) {
    rewriter.eraseOp(produce);
  }

  return newRegionOp;
}

static FailureOr<gptvm::domain::RegionOp>
wrapOpInDispatchRegion(RewriterBase &rewriter, Operation *op) {
  OpBuilder::InsertionGuard g(rewriter);

  rewriter.setInsertionPointAfter(op);
  mlir::MLIRContext *context = rewriter.getContext();
  gptvm::domain::RegionOp regionOp = rewriter.create<gptvm::domain::RegionOp>(
      op->getLoc(), TypeRange(), ValueRange(),
      mlir::StringAttr::get(context, kGPUDeviceType),
      mlir::IntegerAttr::get(rewriter.getI64Type(), kDeivceID));
  Block &body = regionOp.getBody().emplaceBlock();
  rewriter.setInsertionPointToStart(&body);
  rewriter.create<gptvm::domain::ReturnOp>(op->getLoc(), ValueRange());

  // Move the op into the dispatch region.
  auto newRegionOp = movePrecedingOpsIntoRegion(rewriter, op, regionOp);
  return newRegionOp;
}

static unsigned decideRegionOps(mlir::func::FuncOp &funcOp,
                                DominanceInfo const &dominaceInfo) {
  unsigned numRootOps = 0;
  mlir::MLIRContext *context = funcOp.getContext();
  mlir::Block &entryBlock = funcOp.getBody().front();
  SmallVector<Operation *> worklist;
  funcOp->walk<WalkOrder::PostOrder>([&](Operation *op) -> void {
    op->dump();
    if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
      for (auto item : returnOp->getOperands()) {
        auto refOp = item.getDefiningOp();
        refOp->dump();
        worklist.push_back(item.getDefiningOp());
      }
    }
  });

  funcOp->walk<WalkOrder::PreOrder>([&](Operation *op) -> void {
    op->dump();
    if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
      for (auto item : returnOp->getOperands()) {
        auto refOp = item.getDefiningOp();
        refOp->dump();
        worklist.push_back(item.getDefiningOp());
      }
    }
  });

  for (mlir::Value operand : entryBlock.getArguments()) {
    unsigned newGroup = numRootOps++;
    for (mlir::OpOperand &use : operand.getUses()) {
      mlir::Operation *userOp = use.getOwner();
      if (!hasRootOpAttribute(userOp)) {
        setRootAttribute(context, userOp, newGroup);
      }
    }
  }
  return numRootOps;
}

static gptvm::domain::RegionOp makeEmptyRegionOp(RewriterBase &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);
  mlir::MLIRContext *context = rewriter.getContext();
  Location loc = UnknownLoc::get(context);
  gptvm::domain::RegionOp regionOp = rewriter.create<gptvm::domain::RegionOp>(
      loc, TypeRange(), ValueRange(),
      mlir::StringAttr::get(context, kGPUDeviceType),
      mlir::IntegerAttr::get(rewriter.getI64Type(), kDeivceID));
  Block &body = regionOp.getBody().emplaceBlock();
  rewriter.setInsertionPointToStart(&body);
  rewriter.create<gptvm::domain::ReturnOp>(loc, ValueRange());

  return regionOp;
}

static FailureOr<gptvm::domain::RegionOp>
addOpIntoRegionOp(mlir::RewriterBase &rewriter,
                  gptvm::domain::RegionOp regionOp, Operation *op) {
  Block &body = regionOp.getBody().front();
  mlir::FailureOr<gptvm::domain::RegionOp> newRegionOp;
  if (body.getOperations().empty()) {
    newRegionOp = wrapOpInDispatchRegion(rewriter, op);
  } else {
    newRegionOp = movePrecedingOpsIntoRegion(rewriter, op, regionOp);
  }
  return newRegionOp;
}

static LogicalResult createDomainOps(mlir::func::FuncOp funcOp,
                                     DominanceInfo const &dominaceInfo) {
  // unsigned numRoots = decideRegionOps(funcOp, dominaceInfo);
  // mlir::SmallVector<mlir::Operation *> roots(numRoots, nullptr);
  // DenseMap<unsigned, mlir::SmallVector<mlir::Operation *>> producers;
  // mlir::MLIRContext *context = funcOp.getContext();
  // mlir::IRRewriter rewriter(context);
  // funcOp.walk([&](mlir::Operation *op) -> void {
  //   if (hasRootOpAttribute(op)) {
  //     roots[getRootNum(op)] = op;
  //   }
  // });

  // mlir::OpBuilder::InsertionGuard guard(rewriter);
  // SmallVector<gptvm::domain::RegionOp> regionOps;
  // for (const auto &it : llvm::enumerate(roots)) {
  //   gptvm::domain::RegionOp regionOp;
  //   auto refRegionOp = wrapOpInDispatchRegion(rewriter, it.value());
  //   if (mlir::failed(refRegionOp)) {
  //     return mlir::failure();
  //   }

  //   for (mlir::Operation *producer : llvm::reverse(producers[it.index()])) {
  //     auto newRegionOp =
  //         movePrecedingOpsIntoRegion(rewriter, producer, regionOp);
  //     if (mlir::failed(newRegionOp)) {
  //       return mlir::failure();
  //     }
  //     regionOp = *newRegionOp;
  //     regionOps.push_back(regionOp);
  //   }
  // }
  SmallVector<mlir::Operation *> worklist;
  for (mlir::Block &block : funcOp.getRegion().getBlocks()) {
    for (mlir::Operation &refOp : block.getOperations()) {
      worklist.push_back(&refOp);
    }
  }

  mlir::MLIRContext *context = funcOp.getContext();
  mlir::IRRewriter rewriter(context);
  OpBuilder::InsertionGuard g(rewriter);

  mlir::Region &funcBody = funcOp.getRegion();
  rewriter.setInsertionPointToStart(&funcBody.front());
  gptvm::domain::RegionOp regionOp = makeEmptyRegionOp(rewriter);

  for (mlir::Operation *it : llvm::reverse(worklist)) {
    if (!it->hasTrait<OpTrait::IsTerminator>()) {
      auto newRegionOp = addOpIntoRegionOp(rewriter, regionOp, it);
      if (mlir::failed(newRegionOp)) {
        return mlir::failure();
      }
      regionOp = *newRegionOp;
    }
  }

  return mlir::success();
}

class DomainInit : public impl::DomainInitBase<DomainInit> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<mlir::tosa::TosaDialect, tensor::TensorDialect, scf::SCFDialect,
                gptvm::domain::DomainDialect, mlir::ONNXDialect>();
  }
  void runOnOperation() override {
    mlir::func::FuncOp funcOp = getOperation();
    DominanceInfo const &dominanceInfo = getAnalysis<DominanceInfo>();
    if (failed(createDomainOps(funcOp, dominanceInfo))) {
      funcOp->emitOpError("failed to create domain op");
    }
  }
};

} // namespace

std::unique_ptr<Pass> gptvm::domain::createDomainInitPass() {
  return std::make_unique<DomainInit>();
}
