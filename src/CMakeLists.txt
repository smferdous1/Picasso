set(HEADER_LIST
  "${PROJECT_SOURCE_DIR}/include/ClqPart/Types.h"
  "${PROJECT_SOURCE_DIR}/include/ClqPart/JsonGraph.h"
  "${PROJECT_SOURCE_DIR}/include/ClqPart/graphColor.h"
  "${PROJECT_SOURCE_DIR}/include/ClqPart/graph.h"
  "${PROJECT_SOURCE_DIR}/include/ClqPart/input.h"
  "${PROJECT_SOURCE_DIR}/include/ClqPart/utility.h"
  "${PROJECT_SOURCE_DIR}/include/ClqPart/greedyColor.h"
  "${PROJECT_SOURCE_DIR}/include/ClqPart/paletteCol.h"
    )

# Make an automatic library - will be static or dynamic based on user setting
add_library(ClqPart
  JsonGraph.cc
  graph.cc
  input.cc
  greedyColor.cc
  paletteColor.cc
  utility.cc
  ${HEADER_LIST}
  )

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
  target_link_libraries(ClqPart PUBLIC OpenMP::OpenMP_CXX)
endif()

target_include_directories(ClqPart PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
  )

target_compile_features(ClqPart PUBLIC cxx_std_17)
target_compile_options(ClqPart PRIVATE -O3)

if(ENABLE_CUDA)
  set(CU_HEADER_LIST
    "${PROJECT_SOURCE_DIR}/include/ClqPart/Types.h"
    "${PROJECT_SOURCE_DIR}/include/ClqPart/JsonGraph.h"
    "${PROJECT_SOURCE_DIR}/include/ClqPart/graphColor.h"
    "${PROJECT_SOURCE_DIR}/include/ClqPart/graph.h"
    "${PROJECT_SOURCE_DIR}/include/ClqPart/input.h"
    "${PROJECT_SOURCE_DIR}/include/ClqPart/utility.h"
    "${PROJECT_SOURCE_DIR}/include/ClqPart/greedyColor.h"
    "${PROJECT_SOURCE_DIR}/include/ClqPart/paletteCol.h"
    "${PROJECT_SOURCE_DIR}/include/ClqPart/cuPaletteCol.cuh"
      )

  add_library(ClqPartGpu
    JsonGraph.cc
    graph.cc
    input.cc
    greedyColor.cc
    paletteColor.cc
    utility.cc
    cuPaletteCol.cu
    ${CU_HEADER_LIST}
    )

  if(OpenMP_CXX_FOUND)
    target_link_libraries(ClqPartGpu PUBLIC OpenMP::OpenMP_CXX)
  endif()

  target_include_directories(ClqPartGpu PUBLIC
    "${PROJECT_SOURCE_DIR}/include"
    )
  target_link_libraries(ClqPartGpu PUBLIC ${CUDA_LIBRARIES})
  target_include_directories(ClqPartGpu PUBLIC ${CUDA_INCLUDE_DIRS})
  target_compile_features(ClqPartGpu PUBLIC cxx_std_17)
  target_compile_options(ClqPartGpu PRIVATE -O3)
  target_compile_definitions(ClqPartGpu PUBLIC ENABLE_GPU)
endif()
