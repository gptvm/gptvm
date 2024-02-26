#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MemoryBufferRef.h"

namespace gptvm {

mlir::OwningOpRef<mlir::ModuleOp> importFromOnnx(mlir::MLIRContext &context,
                                                 llvm::MemoryBufferRef onnx);

std::unique_ptr<llvm::MemoryBuffer> exportToOnnx(mlir::func::FuncOp func);

} // namespace gptvm
