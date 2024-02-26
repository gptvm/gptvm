function(add_gptvm_library name)
  cmake_parse_arguments(ARG
    ""
    ""
    "DEPENDS;INCLUDE_DIRS;LINK_LIBS;LINK_COMPONENTS"
    ${ARGN}
    )

  add_library(${name} ${ARG_UNPARSED_ARGUMENTS})

  if (ARG_DEPENDS)
    add_dependencies(${name} ${ARG_DEPENDS})
  endif()

  if (ARG_INCLUDE_DIRS)
    target_include_directories(${name} PUBLIC ${ARG_INCLUDE_DIRS})
  endif()
  target_include_directories(${name} PUBLIC ${GV_INCLUDE_DIRS})

  if (ARG_LINK_LIBS)
    target_link_libraries(${name} PUBLIC ${ARG_LINK_LIBS})
  endif()
  target_link_libraries(${name} PUBLIC ${GV_LINK_LIBS})
  set_property(TARGET ${name} PROPERTY POSITION_INDEPENDENT_CODE ON)
endfunction(add_gptvm_library)

function(add_gptvm_backend name)
  cmake_parse_arguments(ARG
    ""
    ""
    "FIND_HEADERS;FIND_LIBS;INCLUDE_DIRS;LINK_LIBS;PATH"
    ${ARGN}
    )

  list(APPEND GV_BACKENDS ${name})
  set(GV_BACKENDS ${GV_BACKENDS} PARENT_SCOPE)

  if (ARG_FIND_HEADERS)
    unset(FOUND_INC_DIR CACHE)
    find_path(FOUND_INC_DIR NAMES ${ARG_FIND_HEADERS} HINTS ${ARG_PATH}
      PATH_SUFFIXES include
      )
    if (FOUND_INC_DIR STREQUAL "FOUND_INC_DIR-NOTFOUND")
      message(STATUS "Could NOT find ${ARG_FIND_HEADERS} for backend ${name}")
      return()
    endif()
  endif()

  if (ARG_FIND_LIBS)
    unset(FOUND_LIBS CACHE)
    find_library(FOUND_LIBS NAMES ${ARG_FIND_LIBS} HINTS ${ARG_PATH}
      PATH_SUFFIXES lib
      )
    if (FOUND_LIBS STREQUAL "FOUND_LINK_LIBS-NOTFOUND")
      message(STATUS "Could NOT find ${ARG_FIND_LIBS} for backend ${name}")
      return()
    endif()
  endif()

  add_gptvm_library(gv_${name} SHARED ${ARG_UNPARSED_ARGUMENTS}
    INCLUDE_DIRS PUBLIC ${FOUND_INC_DIR} ${ARG_INCLUDE_DIRS}
    LINK_LIBS PUBLIC ${FOUND_LIBS} ${ARG_LINK_LIBS}
    )

  install(TARGETS gv_${name} DESTINATION gptvm/backend)
endfunction(add_gptvm_backend)

function(add_gptvm_executable name)
  cmake_parse_arguments(ARG
    ""
    ""
    "DEPENDS;INCLUDE_DIRS;LINK_LIBS;DEFINE"
    ${ARGN}
    )

  if (EXCLUDE_FROM_ALL)
    add_executable(${name} EXCLUDE_FROM_ALL ${ARG_UNPARSED_ARGUMENTS})
  else()
    add_executable(${name} ${ARG_UNPARSED_ARGUMENTS})
  endif()

  if (ARG_DEPENDS)
    add_dependencies(${name} ${ARG_DEPENDS})
  endif()

  if (ARG_INCLUDE_DIRS)
    target_include_directories(${name} PUBLIC ${ARG_INCLUDE_DIRS})
  endif()
  target_include_directories(${name} PUBLIC ${GV_INCLUDE_DIRS})

  if (ARG_LINK_LIBS)
    target_link_libraries(${name} PUBLIC ${ARG_LINK_LIBS})
  endif()
  target_link_libraries(${name} PUBLIC ${GV_LINK_LIBS})

  if (ARG_DEFINE)
    target_compile_definitions(${name} ${ARG_DEFINE})
  endif()
endfunction(add_gptvm_executable)
