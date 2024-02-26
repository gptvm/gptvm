#include "gptvm/compiler/frontend.h"

#include "onnx-mlir/src/Builder/FrontendDialectTransformer.hpp"
#include "onnx-mlir/src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"
#include "onnx-mlir/src/Dialect/ONNX/ElementsAttr/ElementsAttrBuilder.hpp"
#include "onnx-mlir/src/Dialect/ONNX/ONNXOps.hpp"

#include "onnx/version_converter/convert.h"

using namespace llvm;
using namespace mlir;

namespace gptvm {

OwningOpRef<ModuleOp> importFromOnnx(MLIRContext &context,
                                     MemoryBufferRef onnx) {
  OwningOpRef<ModuleOp> module;
  std::string errorMsg;
  onnx_mlir::ImportFrontendModelArray(
      onnx.getBufferStart(), onnx.getBufferSize(), context, module, &errorMsg);
  return std::move(module);
}

namespace {

class GraphConverter {
public:
  GraphConverter(mlir::func::FuncOp func) {
    for (auto attr : func->getAttrOfType<ArrayAttr>("input_names"))
      input_names.emplace_back(attr.cast<StringAttr>().str());
    for (auto attr : func->getAttrOfType<ArrayAttr>("output_names"))
      output_names.emplace_back(attr.cast<StringAttr>().str());
  }

  void convert(onnx::GraphProto *proto, mlir::Region &region);

private:
  std::vector<std::string> input_names;

  std::vector<std::string> output_names;

  llvm::DenseMap<mlir::Value, std::string> value_map;

  size_t temp_id{0};

  void convert(onnx::AttributeProto *proto, mlir::DenseElementsAttr dense) {
    auto *t = proto->mutable_t();
    auto type = dense.getType().cast<ShapedType>();
    for (auto i : type.getShape()) {
      t->add_dims(i);
    }
    t->set_data_type(convert(type.getElementType()));
    auto raw_data = dense.getRawData();
    t->set_raw_data(raw_data.data(), raw_data.size());
    proto->set_type(onnx::AttributeProto_AttributeType_TENSOR);
  }

  void convert(onnx::AttributeProto *proto, mlir::Attribute attr) {
    if (auto int_attr = attr.dyn_cast<IntegerAttr>()) {
      proto->set_i(int_attr.getValue().getZExtValue());
      proto->set_type(onnx::AttributeProto_AttributeType_INT);
    } else if (auto float_attr = attr.dyn_cast<FloatAttr>()) {
      proto->set_f(float_attr.getValueAsDouble());
      proto->set_type(onnx::AttributeProto_AttributeType_FLOAT);
    } else if (auto str_attr = attr.dyn_cast<StringAttr>()) {
      proto->set_s(str_attr.str());
      proto->set_type(onnx::AttributeProto_AttributeType_STRING);
    } else if (auto type_attr = attr.dyn_cast<TypeAttr>()) {
      proto->set_i(convert(type_attr.getValue()));
      proto->set_type(onnx::AttributeProto_AttributeType_INT);
    } else if (auto dense = attr.dyn_cast<DenseElementsAttr>()) {
      convert(proto, dense);
    } else if (auto disposable = attr.dyn_cast<DisposableElementsAttr>()) {
      auto dense =
          onnx_mlir::ElementsAttrBuilder::toDenseElementsAttr(disposable);
      convert(proto, dense);
    } else if (auto array_attr = attr.dyn_cast<ArrayAttr>()) {
      if (llvm::all_of(array_attr,
                       [](auto attr) { return isa<IntegerAttr>(attr); })) {
        for (auto attr : array_attr) {
          proto->add_ints(attr.cast<IntegerAttr>().getValue().getZExtValue());
        }
        proto->set_type(onnx::AttributeProto_AttributeType_INTS);
      } else if (llvm::all_of(array_attr,
                              [](auto attr) { return isa<FloatAttr>(attr); })) {
        for (auto attr : array_attr) {
          proto->add_floats(attr.cast<FloatAttr>().getValueAsDouble());
        }
        proto->set_type(onnx::AttributeProto_AttributeType_STRINGS);
      } else if (llvm::all_of(array_attr, [](auto attr) {
                   return isa<StringAttr>(attr);
                 })) {
        for (auto attr : array_attr) {
          proto->add_strings(attr.cast<StringAttr>().str());
        }
        proto->set_type(onnx::AttributeProto_AttributeType_STRINGS);
      }
    } else {
      assert(false && "Unsupported attribute type");
    }
  }

  onnx::TensorProto_DataType convert(mlir::Type type) {
    if (type.isF16())
      return onnx::TensorProto_DataType_FLOAT16;
    if (type.isF32())
      return onnx::TensorProto_DataType_FLOAT;
    if (type.isF64())
      return onnx::TensorProto_DataType_DOUBLE;
    if (type.isInteger(8))
      return onnx::TensorProto_DataType_INT8;
    if (type.isInteger(16))
      return onnx::TensorProto_DataType_INT16;
    if (type.isInteger(32))
      return onnx::TensorProto_DataType_INT32;
    if (type.isInteger(64))
      return onnx::TensorProto_DataType_INT64;
    if (type.isUnsignedInteger(8))
      return onnx::TensorProto_DataType_UINT8;
    if (type.isUnsignedInteger(16))
      return onnx::TensorProto_DataType_UINT16;
    if (type.isUnsignedInteger(32))
      return onnx::TensorProto_DataType_UINT32;
    if (type.isUnsignedInteger(64))
      return onnx::TensorProto_DataType_UINT64;
    llvm_unreachable("Unsupported data type");
  }

  void convert(onnx::TypeProto *proto, mlir::Type type) {
    auto ttype = type.cast<RankedTensorType>();
    auto *tensor = proto->mutable_tensor_type();
    tensor->set_elem_type(convert(ttype.getElementType()));
    auto *shape = tensor->mutable_shape();
    for (auto i : ttype.getShape()) {
      auto *dim = shape->add_dim();
      dim->set_dim_value(i);
    }
  }

  void convert(onnx::NodeProto *proto, Operation *op) {
    for (auto opr : op->getOperands()) {
      if (!opr.getDefiningOp<ONNXNoneOp>())
        proto->add_input(value_map.at(opr));
    }

    std::string node_name;
    if (op->hasAttr("onnx_node_name")) {
      node_name = op->getAttrOfType<StringAttr>("onnx_node_name").str();
    } else {
      node_name = "anonymous_" + std::to_string(temp_id++);
    }

    for (auto res : llvm::enumerate(op->getResults())) {
      // Check if this is the output node whose name is already defined as graph
      // output
      auto it = value_map.find(res.value());
      if (it != value_map.end()) {
        proto->add_output(it->second);
      } else {
        std::string name = node_name + "_output_" + std::to_string(res.index());
        proto->add_output(name);
        value_map[res.value()] = name;
      }
    }

    proto->set_name(node_name);

    std::string op_name = op->getName().stripDialect().str();
    if (op_name == "MaxPoolSingleOut")
      op_name = "MaxPool";
    proto->set_op_type(op_name);

    for (auto attr : op->getAttrs()) {
      auto name = attr.getName().str();
      if (name == "onnx_node_name")
        continue;

      auto attr_proto = proto->add_attribute();
      attr_proto->set_name(name);
      convert(attr_proto, attr.getValue());
    }
  }
};

void GraphConverter::convert(onnx::GraphProto *proto, mlir::Region &region) {
  for (auto [name, arg] : llvm::zip(input_names, region.getArguments())) {
    auto *value_info = proto->add_input();
    value_info->set_name(name);
    convert(value_info->mutable_type(), arg.getType());
    value_map[arg] = name;
  }

  auto *terminator = &region.front().back();
  for (auto [name, out] : llvm::zip(output_names, terminator->getOperands())) {
    auto *value_info = proto->add_output();
    value_info->set_name(name);
    convert(value_info->mutable_type(), out.getType());
    value_map[out] = name;
  }

  for (auto &op : region.front()) {
    if (!isa<ONNXNoneOp, ONNXReturnOp>(&op))
      convert(proto->add_node(), &op);
  }
}

} // namespace

std::unique_ptr<MemoryBuffer> exportToOnnx(func::FuncOp func) {
  onnx::ModelProto model_proto;
  model_proto.set_ir_version(onnx::IR_VERSION);
  model_proto.set_producer_name("gptvm_compiler");
  model_proto.add_opset_import()->set_version(18);

  auto *graph_proto = model_proto.mutable_graph();
  GraphConverter(func).convert(graph_proto, func.getBody());

  // model_proto = onnx::version_conversion::ConvertVersion(model_proto, 13);

  size_t proto_size = model_proto.ByteSizeLong();
  auto buffer = WritableMemoryBuffer::getNewMemBuffer(proto_size);
  model_proto.SerializeToArray(buffer->getBufferStart(), proto_size);
  return std::move(buffer);
}

} // namespace gptvm
