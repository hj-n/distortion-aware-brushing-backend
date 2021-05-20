cmake_minimum_required(VERSION 3.9)

project(brushing_mdp_backend)

find_package(OpenMP REQUIRED)

# SET ( CMAKE_VERBOSE_MAKEFILE true )

SET ( BACK_METHODS kde.cpp msq.cpp )
SET ( B_LIB backend )  ## target for shared backend library (will be called in python)
SET ( B_LIB_DR ${CMAKE_CURRENT_LIST_DIR}/../lib )

ADD_EXECUTABLE ( test.out test.cpp ${BACK_METHODS} )
ADD_LIBRARY ( ${B_LIB} MODULE ${BACK_METHODS} )

target_link_libraries(test.out PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(${B_LIB} PRIVATE OpenMP::OpenMP_CXX)

set_target_properties(${B_LIB} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${B_LIB_DR} )


