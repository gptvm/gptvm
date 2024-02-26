#pragma once

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "gptvm/compiler/dialect/domain/ir/domain_dialect.h.inc"

#define GET_OP_CLASSES
#include "gptvm/compiler/dialect/domain/ir/domain_ops.h.inc"
