file(GLOB GV_COMPILER_FE_SRCS CONFIGURE_DEPENDS *.cc)

add_gptvm_library(GVCompilerFrontend ${GV_COMPILER_FE_SRCS}
  LINK_LIBS OMBuilder
  )
