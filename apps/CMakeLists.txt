# Make an automatic library - will be static or dynamic based on user setting
#add_executable(cliqePart cliqePart.cpp)
#target_link_libraries(cliqePart PUBLIC ${ColPack_LIBS})
#target_link_directories(cliqePart PUBLIC ${ColPack_LIBDIR})
#target_include_directories(cliqePart PUBLIC ${ColPack_INCDIR})
#target_include_directories(cliqePart PUBLIC "${PROJECT_SOURCE_DIR}/include") 

#if(OpenMP_CXX_FOUND)
#  target_link_libraries(cliqePart PUBLIC OpenMP::OpenMP_CXX)
#endif()


add_executable(testjson testJsonGraph.cpp)
target_include_directories(testjson PUBLIC "${PROJECT_SOURCE_DIR}/include") 
target_link_libraries(testjson PUBLIC ClqPart)
target_compile_options(testjson PRIVATE -O3 )

add_executable(jsonmtx jsonToMtx.cpp)
target_include_directories(jsonmtx PUBLIC "${PROJECT_SOURCE_DIR}/include") 
target_link_libraries(jsonmtx PUBLIC ClqPart)

#add_executable(gcol appGraphColor.cc)
#target_include_directories(gcol PUBLIC "${PROJECT_SOURCE_DIR}/include") 
#target_link_libraries(gcol PUBLIC ClqPart)

add_executable(grdcol appGreedyColor.cc)
target_include_directories(grdcol PUBLIC "${PROJECT_SOURCE_DIR}/include") 
target_link_libraries(grdcol PUBLIC ClqPart)

add_executable(palcol appPaletteColor.cc)
target_include_directories(palcol PUBLIC "${PROJECT_SOURCE_DIR}/include") 
target_link_libraries(palcol PUBLIC ClqPart)
target_compile_options(palcol PRIVATE -O3 )

add_executable(palcolEcl appPaletteColorECL.cc)
target_include_directories(palcolEcl PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(palcolEcl PUBLIC ClqPart)
target_compile_options(palcolEcl PRIVATE -O3 )

add_executable(palcolS appPalColDir.cc)
target_include_directories(palcolS PUBLIC "${PROJECT_SOURCE_DIR}/include") 
target_link_libraries(palcolS PUBLIC ClqPart)
target_compile_options(palcolS PRIVATE -O3 )

add_executable(palcolE appPalColEnc.cc)
target_include_directories(palcolE PUBLIC "${PROJECT_SOURCE_DIR}/include") 
target_link_libraries(palcolE PUBLIC ClqPart)
target_compile_options(palcolE PRIVATE -O3 )

add_executable(palcolR appPalColRecDir.cc)
target_include_directories(palcolR PUBLIC "${PROJECT_SOURCE_DIR}/include") 
target_link_libraries(palcolR PUBLIC ClqPart)
target_compile_options(palcolR PRIVATE -O3 )

add_executable(palcolEr appPalColEncRecDir.cc)
target_include_directories(palcolEr PUBLIC "${PROJECT_SOURCE_DIR}/include") 
target_link_libraries(palcolEr PUBLIC ClqPart)
target_compile_options(palcolEr PRIVATE -O3 )

if(ENABLE_CUDA)
  add_executable(palcolG appPalColGpu.cc)
  target_include_directories(palcolG PUBLIC "${PROJECT_SOURCE_DIR}/include")
  target_link_libraries(palcolG PUBLIC ClqPartGpu)
  target_compile_definitions(palcolG PUBLIC ENABLE_GPU)
  target_compile_options(palcolG PRIVATE -O3 )
  set_target_properties(palcolG PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  add_executable(palcolGm appPalColGpuMem.cc)
  target_include_directories(palcolGm PUBLIC "${PROJECT_SOURCE_DIR}/include")
  target_link_libraries(palcolGm PUBLIC ClqPartGpu)
  target_compile_definitions(palcolGm PUBLIC ENABLE_GPU)
  target_compile_options(palcolGm PRIVATE -O3 )
  set_target_properties(palcolGm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  add_executable(palcolGr appPalColGpuMemRec.cc)
  target_include_directories(palcolGr PUBLIC "${PROJECT_SOURCE_DIR}/include")
  target_link_libraries(palcolGr PUBLIC ClqPartGpu)
  target_compile_definitions(palcolGr PUBLIC ENABLE_GPU)
  set_target_properties(palcolGr PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_options(palcolGr PRIVATE -O3 )
endif()

add_executable(naiveCol appNaiveColDir.cc)
target_include_directories(naiveCol PUBLIC "${PROJECT_SOURCE_DIR}/include") 
target_link_libraries(naiveCol PUBLIC ClqPart)

add_executable(tGraph testGraphIn.cc)
target_include_directories(tGraph PUBLIC "${PROJECT_SOURCE_DIR}/include") 
target_link_libraries(tGraph PUBLIC ClqPart)

#message(STATUS "ColPack: found=${ColPack_FOUND}, include_dirs=${ColPack_INCDIR}, "
#  "lib=${ColPack_LIBDIR}, version=${ColPack_VERSION}")
  #target_include_directories(gstream PUBLIC
  #"${PROJECT_SOURCE_DIR}/include"
  #)

# All users of this library will need at least C++11
#target_compile_features(gstream PUBLIC cxx_std_17)
#target_compile_options(ra PRIVATE -O2 -fopenmp)
#target_compile_options(gstream PRIVATE -O3)
