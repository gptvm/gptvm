#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DebugCounter.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "onnx-mlir/src/Dialect/ONNX/ONNXDialect.hpp"

#include "gptvm/compiler/dialect/domain/ir/domain_ops.h"
#include "gptvm/compiler/dialect/domain/transforms/passes.h"

using namespace mlir;
using namespace llvm;
int main(int argc, char *argv[]) {
  mlir::registerAllPasses();
  gptvm::domain::registerDomainPasses();

  DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  registry.insert<mlir::ONNXDialect>();
  registry.insert<gptvm::domain::DomainDialect>();

  return failed(mlir::MlirOptMain(argc, argv, "GPTVM pass driver\n", registry));
}
