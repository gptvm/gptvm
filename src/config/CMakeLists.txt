add_custom_target(gv_config)
add_custom_command(
        TARGET gv_config POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/nv_spec.json
                ${CMAKE_CURRENT_BINARY_DIR})

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/nv_spec.json DESTINATION gptvm/config)
