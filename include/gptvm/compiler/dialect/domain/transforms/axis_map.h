
#pragma once
#include "llvm-project/mlir/include/mlir/IR/Operation.h"
#include "llvm-project/mlir/include/mlir/Support/LLVM.h"

#include <vector>

namespace gptvm {
namespace domain {
enum class AxisType {
  AxisType_Parallel = 0,
  AxisType_Reduce_Sum = 1,
  AxisType_Reduce_Min = 2,
  AxisType_Reduce_Max = 3,
  AxisType_Reduce_Prod = 4
};

class AxisDef;

using AxisDefRef = std::shared_ptr<AxisDef>;

class AxisDef {
public:
  AxisDef(AxisType c, int idx) : type(c), index(idx){};

  AxisType type;

  int index;

  static AxisDefRef getParallel(int index) {
    AxisDefRef axis(
        std::make_shared<AxisDef>(AxisType::AxisType_Parallel, index));
    return axis;
  };

  static AxisDefRef getReduceSum(int index) {
    AxisDefRef axis(
        std::make_shared<AxisDef>(AxisType::AxisType_Reduce_Sum, index));
    return axis;
  };

  static AxisDefRef getReduceMin(int index) {
    AxisDefRef axis(
        std::make_shared<AxisDef>(AxisType::AxisType_Reduce_Min, index));
    return axis;
  };

  static AxisDefRef getReduceMax(int index) {
    AxisDefRef axis(
        std::make_shared<AxisDef>(AxisType::AxisType_Reduce_Max, index));
    return axis;
  };

  static AxisDefRef getReduceProd(int index) {
    AxisDefRef axis(
        std::make_shared<AxisDef>(AxisType::AxisType_Reduce_Prod, index));
    return axis;
  };

  static AxisDefRef getNone() { return nullptr; };

  static std::vector<AxisDefRef> getAllParallel(int size) {
    std::vector<AxisDefRef> allParallel;
    for (int i = 0; i < size; i++) {
      allParallel.emplace_back(getParallel(i));
    }
    return allParallel;
  }

  static std::vector<AxisDefRef> getAllNone(int size) {
    std::vector<AxisDefRef> allNone;
    for (int i = 0; i < size; i++) {
      allNone.emplace_back(getNone());
    }
    return allNone;
  }
};

class AxisMap {
public:
  AxisMap() = default;
  AxisMap(const std::vector<std::vector<AxisDefRef>> &a,
          const std::vector<AxisDefRef> &b)
      : ins(a), outs({b}) {}
  AxisMap(const std::vector<std::vector<AxisDefRef>> &a,
          const std::vector<std::vector<AxisDefRef>> &b)
      : ins(a), outs(b) {}

  std::vector<std::vector<AxisDefRef>> getInsAxisDef() const { return ins; }

  std::vector<std::vector<AxisDefRef>> getOutsAxisDef() const { return outs; }

  std::vector<AxisDefRef> getAxes() const;

private:
  std::vector<std::vector<AxisDefRef>> ins;

  std::vector<std::vector<AxisDefRef>> outs;
};

void print(AxisType axisType, llvm::raw_ostream &os) {
  switch (axisType) {
  case AxisType::AxisType_Parallel:
    os << "Parallel";
    break;
  case AxisType::AxisType_Reduce_Sum:
    os << "Reduce Sum";
    break;
  case AxisType::AxisType_Reduce_Min:
    os << "Reduce Min";
    break;
  case AxisType::AxisType_Reduce_Max:
    os << "Reduce Max";
    break;
  case AxisType::AxisType_Reduce_Prod:
    os << "Reduce Prod";
    break;
  default:
    os << "Unknown AxisType";
  }
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, AxisType axisType) {
  print(axisType, os);
  return os;
}
} // namespace domain
} // namespace gptvm