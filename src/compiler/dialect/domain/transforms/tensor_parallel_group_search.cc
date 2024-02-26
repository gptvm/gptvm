// #include "gptvm/compiler/dialect/domain/transforms/axis_map.h"
#include "llvm-project/mlir/include/mlir/Transforms/DialectConversion.h"
#include "gptvm/compiler/dialect/domain/transforms/axis_map.h"
#include "gptvm/compiler/dialect/domain/transforms/onnx_op_info.h"

#include "llvm-project/mlir/include/mlir/IR/Dominance.h"
#include "llvm-project/mlir/include/mlir/IR/Iterators.h"
#include "llvm-project/mlir/include/mlir/IR/Operation.h"
#include "llvm-project/mlir/include/mlir/IR/Value.h"
#include "llvm-project/mlir/include/mlir/IR/Visitors.h"
#include "llvm-project/mlir/include/mlir/Support/LLVM.h"
#include "onnx-mlir/src/Dialect/ONNX/ONNXOps.hpp"
#include "gptvm/compiler/dialect/domain/transforms/passes.h"

#include "llvm-project/mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm-project/mlir/include/mlir/IR/BuiltinOps.h"
#include "llvm-project/mlir/include/mlir/IR/MLIRContext.h"
#include "llvm-project/mlir/include/mlir/IR/PatternMatch.h"
#include "llvm-project/mlir/include/mlir/Pass/Pass.h"
#include "llvm-project/mlir/include/mlir/Pass/PassRegistry.h"
#include "llvm-project/mlir/include/mlir/Support/LogicalResult.h"
#include "llvm-project/mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <queue>
using namespace mlir;
using namespace gptvm::domain;

namespace gptvm {
namespace domain {} // namespace domain
} // namespace gptvm

namespace {
static const char kRootOpAttr[] = "__root_op__";

#define GEN_PASS_DEF_TPGROUPSEARCHPASS
#include "gptvm/compiler/dialect/domain/transforms/passes.h.inc"

static void setRootAttribute(mlir::MLIRContext *context, mlir::Operation *op,
                             int64_t numRoots) {
  op->setAttr(kRootOpAttr, mlir::IntegerAttr::get(
                               mlir::IntegerType::get(context, 64), numRoots));
}

static LogicalResult getReshapeOpType(mlir::Operation *op) {
  if (auto reshapeOp = mlir::dyn_cast<mlir::ONNXReshapeOp>(op)) {
    reshapeOp.getReshaped();
  }

  return mlir::success();
}

struct QKVMatmulPattern : public OpRewritePattern<mlir::ONNXMatMulOp> {
  QKVMatmulPattern(MLIRContext *ctx)
      : OpRewritePattern<mlir::ONNXMatMulOp>(ctx) {}
  LogicalResult matchAndRewrite(mlir::ONNXMatMulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    bool isMatchedPattern = false;
    for (auto item : matmulOp.getOperands()) {
      if (item.getDefiningOp<mlir::ONNXConstantOp>() != nullptr) {
        isMatchedPattern = true;
        break;
      }
    }
    if (!isMatchedPattern) {
      return mlir::success();
    }

    gptvm::domain::AxisMap axisMap = gptvm::domain::getMatmulAxisMap(matmulOp);
    auto inputAxisDef = axisMap.getInsAxisDef();
    for (size_t idx = 0; idx < inputAxisDef.size(); idx++) {
      for (size_t defIdx = 0; defIdx < inputAxisDef[idx].size(); defIdx++) {
        gptvm::domain::AxisDefRef refAxisDef = inputAxisDef[idx][defIdx];
        llvm::outs() << refAxisDef->index
                     << " || parallel type is:" << refAxisDef->type << "\n";
      }
    }

    mlir::DominanceInfo domInfo(matmulOp.getOperation());
    mlir::PostDominanceInfo postDomInfo(matmulOp.getOperation());
    mlir::SmallVector<mlir::Operation *> users;
    for (auto res : matmulOp.getOperation()->getResults()) {
      for (mlir::OpOperand &user : res.getUses()) {
        users.push_back(user.getOwner());
      }
    }

    matmulOp.getOperation()->dump();
    std::queue<mlir::Operation *> workList;
    std::set<mlir::Operation *> visitedSet;
    std::queue<mlir::Value> opList;

    workList.push(matmulOp.getOperation());
    opList.push(matmulOp);
    visitedSet.insert(matmulOp.getOperation());

    gptvm::domain::AxisMap matmulOpAxisMap =
        gptvm::domain::getMatmulAxisMap(matmulOp);
    std::vector<gptvm::domain::AxisDefRef> matmulRhsAxisDef =
        matmulOpAxisMap.getInsAxisDef()[1];

    // TODO<ysheng>:
    // matmul: tensor<512x4096xf32> ---> select 4096 as parallel axis
    gptvm::domain::AxisDefRef transferAxis = matmulRhsAxisDef[1];

    while (!workList.empty()) {
      mlir::Operation *rootNode = workList.front();
      llvm::outs() << "****Root Node is****"
                   << "\n";
      rootNode->dump();
      workList.pop();

      gptvm::domain::AxisMap rootOpAxisMap =
          gptvm::domain::getAxisMap(rootNode);

      mlir::DominanceInfo domInfo(rootNode);
      for (auto res : rootNode->getResults()) {
        for (mlir::OpOperand &user : res.getUses()) {
          if (user.getOwner()->hasTrait<OpTrait::IsTerminator>()) {
            return mlir::success();
          }

          gptvm::domain::AxisMap axisMap =
              gptvm::domain::getAxisMap(user.getOwner());
          auto operandItr =
              std::find(user.getOwner()->getOperands().begin(),
                        user.getOwner()->getOperands().end(), user.get());
          assert(operandItr != user.getOwner()->getOperands().end() &&
                 "Invalid operand");
          int32_t indx =
              std::distance(user.getOwner()->getOperands().begin(), operandItr);
          assert(indx != -1 && "Can't find the operand");
          std::vector<gptvm::domain::AxisDefRef> userAxisDefRef =
              axisMap.getInsAxisDef()[indx];
          auto iter =
              std::find_if(userAxisDefRef.begin(), userAxisDefRef.end(),
                           [&](gptvm::domain::AxisDefRef item) {
                             return (item->type == transferAxis->type) &&
                                    (item->index == transferAxis->index);
                           });

          if (iter != userAxisDefRef.end()) {
            workList.push(user.getOwner());
          }
        }
      }
    }

    for (auto user : users) {
      if (domInfo.dominates(matmulOp.getOperation(), user)) {

        user->dump();
        llvm::outs() << "*************"
                     << "\n";
      } else if (postDomInfo.postDominates(user, matmulOp.getOperation())) {
        user->dump();
        llvm::outs() << "*************"
                     << "\n";
      }
    }

    setRootAttribute(rewriter.getContext(), matmulOp.getOperation(), numRoots_);
    // numRoots_ = numRoots_ + 1;
    return mlir::success();
  }

  int numRoots_ = 0;
};

class TensorParallelGroupSearch
    : public impl::TPGroupSearchPassBase<TensorParallelGroupSearch> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::ONNXDialect>();
  }

  void runOnOperation() override {
    mlir::func::FuncOp funcOp = getOperation();
    mlir::MLIRContext *context = funcOp.getContext();
    mlir::RewritePatternSet groupPattern(context);
    groupPattern.insert<QKVMatmulPattern>(context);

    if (mlir::failed(
            applyPatternsAndFoldGreedily(funcOp, std::move(groupPattern)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> gptvm::domain::createTPGroupSearchPass() {
  return std::make_unique<TensorParallelGroupSearch>();
}
