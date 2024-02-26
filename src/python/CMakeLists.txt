file(GLOB GV_PYTHON_SRCS CONFIGURE_DEPENDS *.cc)

add_gptvm_library(gptvm SHARED ${GV_PYTHON_SRCS} LINK_LIBS gptvmrt)
set_target_properties(gptvm
                      PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
add_custom_command(
        TARGET gptvm POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${GV_SOURCE_DIR}/python/gptvm/bin/gptvm
                ${GV_BINARY_DIR}/bin/gptvm)

install(TARGETS gptvm DESTINATION gptvm/lib)
install(DIRECTORY ${GV_SOURCE_DIR}/python/gptvm/ DESTINATION gptvm
        USE_SOURCE_PERMISSIONS PATTERN "__pycache__" EXCLUDE)
