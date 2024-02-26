#pragma once

#include "llvm-project/mlir/include/mlir/IR/Operation.h"
#include "llvm-project/mlir/include/mlir/Support/LLVM.h"
#include "gptvm/compiler/dialect/domain/transforms/axis_map.h"

#include "llvm-project/mlir/include/mlir/IR/BuiltinTypeInterfaces.h"
#include "llvm-project/mlir/include/mlir/IR/BuiltinTypes.h"
#include "onnx-mlir/src/Dialect/ONNX/ONNXOps.hpp"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>

using namespace mlir;

#define DEBUG_TYPE "onnx-op-axis-map"

namespace gptvm {
namespace domain {
enum class OpType {
  OPTYPE_BINARY_ELEMENTWISE = 0,
  OPTYPE_UNARY_ELEMENTWISE = 1,
  OPTYPE_REDUCE = 2
};

AxisMap getMatmulAxisMap(mlir::ONNXMatMulOp matmulOp) {
  mlir::Value lhs = matmulOp.getA();
  mlir::Value rhs = matmulOp.getB();

  mlir::ArrayRef<int64_t> lhsShape =
      lhs.getType().cast<mlir::TensorType>().getShape();
  mlir::ArrayRef<int64_t> rhsShape =
      rhs.getType().cast<mlir::TensorType>().getShape();

  int idx = 0;
  std::vector<AxisDefRef> lhsInput, rhsInput, outs;
  if (lhsShape.size() == rhsShape.size()) {
    for (size_t i = 0; i < lhsShape.size() - 2; i++) {
      auto batch = AxisDef::getParallel(idx++);
      lhsInput.push_back(batch);
      rhsInput.push_back(batch);
      outs.push_back(batch);
    }
  } else {
    for (size_t i = 0; i < lhsShape.size() - 2; i++) {
      auto batch = AxisDef::getParallel(idx++);
      lhsInput.push_back(batch);
      outs.push_back(batch);
    }
  }

  auto m = AxisDef::getParallel(idx++);
  auto n = AxisDef::getParallel(idx++);
  auto k = AxisDef::getReduceSum(idx++);

  lhsInput.push_back(m);
  lhsInput.push_back(k);

  rhsInput.push_back(k);
  rhsInput.push_back(n);
  outs.push_back(m);
  outs.push_back(n);

  // LLVM_DEBUG(llvm::dbgs() << "********Matmul op axis map info********"
  //                         << "\n");
  // LLVM_DEBUG(matmulOp.getOperation()->dump());
  // LLVM_DEBUG(llvm::dbgs() << "******Input lhs axis map is:"
  //                         << "\n");
  // for (auto inputAxisDef : lhsInput) {
  //   LLVM_DEBUG(llvm::dbgs() << inputAxisDef->index
  //                           << " axis type is:" << inputAxisDef->type <<
  //                           "\n");
  // }

  // LLVM_DEBUG(llvm::dbgs() << "******Input rhs axis map is:"
  //                         << "\n");
  // for (auto inputAxisDef : rhsInput) {
  //   LLVM_DEBUG(llvm::dbgs() << inputAxisDef->index
  //                           << " axis type is: " << inputAxisDef->type <<
  //                           "\n");
  // }

  // LLVM_DEBUG(llvm::dbgs() << "******Output  axis map is:"
  //                         << "\n");
  // for (auto outputAxisDef : outs) {
  //   LLVM_DEBUG(llvm::dbgs() << outputAxisDef->index
  //                           << " axis type is:" << outputAxisDef->type <<
  //                           "\n");
  // }

  std::vector<std::vector<AxisDefRef>> ins({lhsInput, rhsInput});
  return AxisMap(ins, outs);
}

AxisMap getElementwiseAxisMap(Operation *op) {
  auto outShape = op->getResult(0).getType().cast<ShapedType>().getShape();
  std::vector<AxisDefRef> outs = AxisDef::getAllParallel(outShape.size());
  std::vector<std::vector<AxisDefRef>> ins;
  for (auto value : op->getOperands()) {
    auto argShape = value.getType().cast<ShapedType>().getShape();
    std::vector<AxisDefRef> input = AxisDef::getAllNone(argShape.size());
    if (argShape.size() != 0) {
      if (argShape.size() == outShape.size()) {
        for (size_t i = 0; i < argShape.size(); i++) {
          if (argShape[i] == outShape[i]) {
            input[i] = outs[i];
          }
        }
      } else {
        if (argShape.size() != 1) {
          op->dump();
        }
        assert(argShape.size() == 1);
        input[0] = outs[outs.size() - 1];
      }
    }
    ins.emplace_back(input);
  }
  return AxisMap(ins, outs);
}

AxisMap getSliceAxisMap(mlir::ONNXSliceOp sliceOp) {
  mlir::ArrayRef<int64_t> dataShape =
      sliceOp.getData().getType().cast<ShapedType>().getShape();

  mlir::ArrayRef<int64_t> resShape =
      sliceOp.getOutput().getType().cast<ShapedType>().getShape();

  std::vector<AxisDefRef> inputs = AxisDef::getAllNone(dataShape.size());
  std::vector<AxisDefRef> outs = AxisDef::getAllParallel(resShape.size());

  for (size_t i = 0; i < dataShape.size(); i++) {
    if (dataShape[i] == resShape[i]) {
      inputs[i] = AxisDef::getParallel(i);
    } else {
      outs[i] = AxisDef::getNone();
    }
  }

  std::vector<std::vector<AxisDefRef>> ins;
  ins.emplace_back(inputs);
  return AxisMap(ins, outs);
}

AxisMap getConcateAxisMap(mlir::ONNXConcatOp concatOp) {
  mlir::ArrayRef<int64_t> resShape =
      concatOp.getConcatResult().getType().cast<ShapedType>().getShape();

  std::vector<AxisDefRef> input = AxisDef::getAllParallel(resShape.size());
  input[concatOp.getAxis()] = AxisDef::getNone();

  std::vector<std::vector<AxisDefRef>> ins;
  std::vector<AxisDefRef> outs;
  for (size_t inputIdx = 0; inputIdx < concatOp.getInputs().size();
       inputIdx++) {
    ins.emplace_back(input);
  }

  outs = input;
  return AxisMap(ins, outs);
}

AxisMap getGatherAxisMap(mlir::ONNXGatherOp gatherOp) {
  mlir::ArrayRef<int64_t> resShape =
      gatherOp.getOutput().getType().cast<ShapedType>().getShape();

  mlir::ArrayRef<int64_t> inputShape =
      gatherOp.getData().getType().cast<mlir::ShapedType>().getShape();

  mlir::ArrayRef<int64_t> indicesShape =
      gatherOp.getIndices().getType().cast<mlir::ShapedType>().getShape();

  std::vector<std::vector<AxisDefRef>> ins;
  std::vector<AxisDefRef> input = AxisDef::getAllParallel(inputShape.size());
  input[gatherOp.getAxis()] = AxisDef::getNone();
  ins.emplace_back(input);

  std::vector<AxisDefRef> indices = AxisDef::getAllNone(indicesShape.size());
  ins.emplace_back(indices);

  std::vector<int32_t> resDims(resShape.size(), 0);
  for (size_t idx = 0; idx < inputShape.size(); idx++) {
    if ((int64_t)idx == gatherOp.getAxis()) {
      for (size_t indicesIdx = 0; indicesIdx < indicesShape.size();
           indicesIdx++) {
        int32_t resIdx = idx + indicesIdx;
        resDims[resIdx] = 1;
      }
    }
  }

  std::vector<AxisDefRef> output = AxisDef::getAllParallel(resShape.size());
  for (size_t idx = 0; idx < resShape.size(); idx++) {
    if (resDims[idx] == 1) {
      output[idx] = AxisDef::getNone();
    }
  }

  // LLVM_DEBUG(llvm::dbgs() << "********Gather op axis map info********"
  //                         << "\n");
  // LLVM_DEBUG(gatherOp.getOperation()->dump());
  // LLVM_DEBUG(llvm::dbgs() << "******Input axis map is:"
  //                         << "\n");
  // for (auto inputAxisDef : input) {
  //   LLVM_DEBUG(llvm::dbgs() << inputAxisDef->index
  //                           << " axis type is:" << inputAxisDef->type <<
  //                           "\n");
  // }

  // LLVM_DEBUG(llvm::dbgs() << "******Output  axis map is:"
  //                         << "\n");
  // for (auto outputAxisDef : output) {
  //   LLVM_DEBUG(llvm::dbgs() << outputAxisDef->index
  //                           << " axis type is:" << outputAxisDef->type <<
  //                           "\n");
  // }

  return AxisMap(ins, output);
}

AxisMap getReshapeAxisMap(mlir::ONNXReshapeOp reshapeOp) {
  mlir::ArrayRef<int64_t> inputShape =
      reshapeOp.getData().getType().cast<mlir::ShapedType>().getShape();
  mlir::ArrayRef<int64_t> reshapeShape =
      reshapeOp.getShape().getType().cast<mlir::ShapedType>().getShape();
  mlir::ArrayRef<int64_t> resShape =
      reshapeOp.getReshaped().getType().cast<mlir::ShapedType>().getShape();
  assert(reshapeShape.size() == 1 && "Only support to dim is 1 for reshape op");

  std::vector<std::vector<AxisDefRef>> ins;
  std::vector<AxisDefRef> input = AxisDef::getAllParallel(inputShape.size());

  std::vector<AxisDefRef> output = AxisDef::getAllParallel(resShape.size());
  if (reshapeShape[0] == 3 && inputShape.size() == 4) {
    // tensor<?x?x32x128xf32> --> tensor<1x1x4096xf32>
    input[3] = AxisDef::getParallel(2);
  }

  // LLVM_DEBUG(llvm::dbgs() << "********Gather op axis map info********"
  //                         << "\n");
  // LLVM_DEBUG(reshapeOp.getOperation()->dump());
  // LLVM_DEBUG(llvm::dbgs() << "******Input axis map is:"
  //                         << "\n");
  // for (auto inputAxisDef : input) {
  //   LLVM_DEBUG(llvm::dbgs() << inputAxisDef->index
  //                           << " axis type is:" << inputAxisDef->type <<
  //                           "\n");
  // }

  // LLVM_DEBUG(llvm::dbgs() << "******Output  axis map is:"
  //                         << "\n");
  // for (auto outputAxisDef : output) {
  //   LLVM_DEBUG(llvm::dbgs() << outputAxisDef->index
  //                           << " axis type is:" << outputAxisDef->type <<
  //                           "\n");
  // }

  ins.emplace_back(input);
  return AxisMap(ins, output);
}

AxisMap getTransposeAxisMap(mlir::ONNXTransposeOp transposeOp) {
  mlir::ArrayRef<int64_t> inputShape =
      transposeOp.getData().getType().cast<mlir::ShapedType>().getShape();

  mlir::ArrayRef<int64_t> resShape =
      transposeOp.getTransposed().getType().cast<mlir::ShapedType>().getShape();
  std::vector<AxisDefRef> output = AxisDef::getAllParallel(resShape.size());

  std::vector<AxisDefRef> input;
  for (size_t idx = 0; idx < inputShape.size(); idx++) {
    input[transposeOp.getPermAttr().cast<mlir::IntegerAttr>().getInt()] =
        output[idx];
  }

  std::vector<std::vector<AxisDefRef>> transIns({input});
  return AxisMap(transIns, output);
}

static AxisMap getAxisMap(mlir::Operation *inputOp) {
  AxisMap axisMap;
  mlir::TypeSwitch<mlir::Operation *, void>(inputOp)
      .Case([&](mlir::ONNXSliceOp sliceOp) {
        axisMap = getSliceAxisMap(sliceOp);
      })
      .Case([&](mlir::ONNXMatMulOp matmulOp) {
        axisMap = getMatmulAxisMap(matmulOp);
      })
      .Case([&](mlir::ONNXConcatOp concatOp) {
        axisMap = getConcateAxisMap(concatOp);
      })
      .Case([&](mlir::ONNXGatherOp gatherOp) {
        axisMap = getGatherAxisMap(gatherOp);
      })
      .Case([&](mlir::ONNXReshapeOp reshapeOp) {
        axisMap = getReshapeAxisMap(reshapeOp);
      })
      .Case([&](mlir::ONNXTransposeOp transposeOp) {
        axisMap = getTransposeAxisMap(transposeOp);
      })
      .Default([&](mlir::Operation *elementwiseOp) {
        axisMap = getElementwiseAxisMap(elementwiseOp);
      });

  return axisMap;
}

} // namespace domain
} // namespace gptvm