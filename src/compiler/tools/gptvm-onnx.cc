#include "gptvm/compiler/dialect/domain/ir/domain_ops.h"
#include "gptvm/compiler/dialect/domain/transforms/passes.h"
#include "gptvm/compiler/frontend.h"

#include "onnx-mlir/src/Dialect/ONNX/ONNXDialect.hpp"
#include "onnx-mlir/src/Pass/Passes.hpp"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;

static cl::OptionCategory gv_onnx_opt("GPTVM ONNX Tool Options");

static cl::opt<std::string> input_file_name(cl::Positional,
                                            cl::desc("<input onnx file>"),
                                            cl::cat(gv_onnx_opt));

static cl::opt<std::string> output_file_name("o", cl::desc("Output filename"),
                                             cl::value_desc("filename"),
                                             cl::init("-"),
                                             cl::cat(gv_onnx_opt));

static cl::opt<bool>
    export_onnx("export-onnx",
                cl::desc("Export functions back to onnx model files"),
                cl::init(false), cl::cat(gv_onnx_opt));

static void addONNXPasses(PassManager &pm) {
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createDecomposeONNXToONNXPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createConstPropONNXToONNXPass());
  // pm.addPass(onnx_mlir::createONNXOpTransformPass());
  pm.addPass(onnx_mlir::createSimplifyShapeRelatedOpsPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createStandardFuncReturnPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(onnx_mlir::createScrubDisposablePass());
}

int main(int argc, char *argv[]) {
  registerAsmPrinterCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "GPTVM ONNX Tool");

  std::string error_msg;
  auto input_file = openInputFile(input_file_name, &error_msg);
  if (!input_file) {
    errs() << error_msg << "\n";
    return 1;
  }

  auto output_file = openOutputFile(output_file_name, &error_msg);
  if (!output_file) {
    errs() << error_msg << "\n";
    return 1;
  }

  MLIRContext context;
  DialectRegistry registry;
  registry.insert<mlir::ONNXDialect>();
  registry.insert<gptvm::domain::DomainDialect>();

  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
  context.allowUnregisteredDialects(true);

  auto module = gptvm::importFromOnnx(context, *input_file);

  PassManager pm(&context);
  addONNXPasses(pm);
  if (failed(pm.run(*module)))
    return 1;

  module->print(output_file->os());
  output_file->keep();

  if (export_onnx) {
    for (auto func : module->getOps<func::FuncOp>()) {
      auto pb = gptvm::exportToOnnx(func);
      std::string func_name = func.getName().str();
      auto onnx_file = openOutputFile(func_name + ".onnx", &error_msg);
      onnx_file->os().write(pb->getBufferStart(), pb->getBufferSize());
      onnx_file->keep();
    }
  }
  return 0;
}
