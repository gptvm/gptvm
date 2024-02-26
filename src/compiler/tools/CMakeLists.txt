add_gptvm_executable(gptvm-onnx gptvm-onnx.cc
  LINK_LIBS
  GVCompilerActorDialect
  GVCompilerActorTransforms
  GVCompilerFrontend
  GVCompilerDomainDialect
  GVCompilerDomainTransforms
  #OMOpTransform
  OMONNXSimplifyShapeRelatedOps
  OMONNXStandardFuncReturnPass
  )

  get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
  get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
  get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
  set(LIBS
  ${dialect_libs}
  ${extension_libs}
  ${conversion_libs}
  GVCompilerActorDialect
  GVCompilerActorTransforms
  GVCompilerDomainDialect
  GVCompilerDomainTransforms

  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRCastInterfaces
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  OMONNXSimplifyShapeRelatedOps
  )

add_gptvm_executable(gptvm-opt gptvm-opt.cc
  LINK_LIBS
  ${LIBS}
  )

install(TARGETS gptvm-opt gptvm-onnx DESTINATION gptvm/bin)
