# Local changes to the ggml submodule, kept as patches until they're upstream (patches/ggml/).
# Each one is applied once and skipped when it's already in the tree, so re-configuring is safe.
#
#   0001  Metal: load a metallib compiled at build time (see MetalPrecompiled.cmake)
#   0002  Metal: im2col with one thread per output element (2x faster DA3 on Apple GPUs)
#
# -DDA_GGML_PATCHES=OFF builds ggml exactly as checked out.

option(DA_GGML_PATCHES "Apply patches/ggml/*.patch to third_party/ggml" ON)
if(NOT DA_GGML_PATCHES)
    return()
endif()

find_package(Git QUIET)
if(NOT GIT_FOUND)
    message(WARNING "depth-anywhere: git not found; ggml patches not applied")
    return()
endif()

file(GLOB _da_patches "${DA_ROOT}/patches/ggml/*.patch")
list(SORT _da_patches)
foreach(_p IN LISTS _da_patches)
    get_filename_component(_name "${_p}" NAME)
    execute_process(COMMAND "${GIT_EXECUTABLE}" apply --reverse --check "${_p}"
                    WORKING_DIRECTORY "${DA_ROOT}/third_party/ggml" RESULT_VARIABLE _rev OUTPUT_QUIET ERROR_QUIET)
    if(_rev EQUAL 0)
        continue()  # already applied
    endif()
    execute_process(COMMAND "${GIT_EXECUTABLE}" apply "${_p}"
                    WORKING_DIRECTORY "${DA_ROOT}/third_party/ggml" RESULT_VARIABLE _fail)
    if(_fail)
        message(FATAL_ERROR "depth-anywhere: couldn't apply ${_name} to third_party/ggml "
                            "(was the submodule updated? refresh the patch or set DA_GGML_PATCHES=OFF)")
    endif()
    message(STATUS "depth-anywhere: applied ${_name} to ggml")
endforeach()
