#include "gptvm/compiler/dialect/domain/ir/domain_ops.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/SmallVector.h"

#include "gptvm/compiler/dialect/domain/ir/domain_dialect.cc.inc"

using namespace mlir;
using namespace gptvm::domain;

struct DomainInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *op, Region *region, bool wouldBeCloned,
                       IRMapping &map) const final {
    return true;
  }

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &map) const final {
    return true;
  }
}; // struct DomainInlinerInterface

void gptvm::domain::DomainDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "gptvm/compiler/dialect/domain/ir/domain_ops.cc.inc"
      >();
  addInterfaces<DomainInlinerInterface>();
}

namespace gptvm {
namespace domain {
void RegionOp::print(OpAsmPrinter &printer) {
  SmallVector<StringRef, 1> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  printer.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elidedAttrs);

  if (!getResults().empty()) {
    printer << " -> (" << getResults().getTypes() << ")";
  }
  bool printTerminator = true;
  if (!getBody().empty()) {
    auto *item = getBody().begin()->getTerminator();
    printTerminator = item->getAttrDictionary().empty() ||
                      item->getNumOperands() != 0 || item->getNumResults() != 0;
    auto &region = getBody();
    printer.printRegion(region, true, printTerminator);
  }
}

ParseResult RegionOp::parse(OpAsmParser &parser, OperationState &result) {
  //  auto &builder = parser.getBuilder();
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  SmallVector<Type> resultTypes;
  if (succeeded(parser.parseOptionalArrow())) {
    ParseResult typeListResult =
        parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, [&]() {
          if (parser.parseType(resultTypes.emplace_back())) {
            return failure();
          }

          auto shapeType = resultTypes.back().dyn_cast<ShapedType>();
          if (shapeType.hasStaticShape()) {
            return success();
          }
          return success();
        });
    if (typeListResult) {
      return failure();
    }
  }

  auto region = std::make_unique<Region>();
  if (parser.parseRegion(*region)) {
    return failure();
  }
  RegionOp::ensureTerminator(*region, parser.getBuilder(), result.location);
  result.addRegion(std::move(region));

  result.addTypes(resultTypes);
  return success();
}

void RegionOp::build(mlir::OpBuilder &builder, ::mlir::OperationState &state,
                     ArrayRef<mlir::Type> inTypes, ArrayRef<mlir::Value> inputs,
                     std::string device_type, int device_id) {
  mlir::MLIRContext *context = builder.getContext();
  auto deviceType = mlir::StringAttr::get(context, device_type);
  auto deviceId = mlir::IntegerAttr::get(builder.getI64Type(), device_id);
  build(builder, state, inTypes, inputs, deviceType, deviceId);
}

} // namespace domain
} // namespace gptvm

#define GET_OP_CLASSES
#include "gptvm/compiler/dialect/domain/ir/domain_ops.cc.inc"
