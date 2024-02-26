function(add_gptvm_compiler_dialect_library target_prefix)
  add_mlir_dialect_library(${target_prefix}Dialect
    ${ARGN}
    DISABLE_INSTALL
    DEPENDS 
    ${target_prefix}IncGen
    LINK_LIBS PUBLIC
    MLIRIR
    MLIRDialect
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTosaDialect
    )
  target_include_directories(${target_prefix}Dialect PRIVATE
    ${GV_BINARY_DIR}/compiler/include)
endfunction()

function(add_gptvm_compiler_dialect_transforms_library target_prefix)
  add_mlir_dialect_library(${target_prefix}Transforms
    ${ARGN}
    DISABLE_INSTALL
    DEPENDS ${target_prefix}PassesGen
    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransforms
    OMONNXSimplifyShapeRelatedOps
    ${target_prefix}Dialect
    )
endfunction()

add_subdirectory(actor)
add_subdirectory(domain)
