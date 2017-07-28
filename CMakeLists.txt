cmake_minimum_required(VERSION 3.0)

set(PROJECT_NAME SolR)
project(${PROJECT_NAME})

set(${PROJECT_NAME}_VERSION_MAJOR 0)
set(${PROJECT_NAME}_VERSION_MINOR 1)
set(${PROJECT_NAME}_VERSION_PATCH 0)

set(CPACK_PACKAGE_VERSION_MAJOR "${${PROJECT_NAME}_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${${PROJECT_NAME}_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${${PROJECT_NAME}_VERSION_PATCH}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# ================================================================================
# Project
# ================================================================================

option(SOLR_KINECT_ENABLED "Activate Kinect" OFF)
option(SOLR_OCULUS_ENABLED "Activate Oculus" OFF)
option(SOLR_SIXENSE_ENABLED "Activate Sixense Controller" OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif(NOT CMAKE_BUILD_TYPE)

set(SOLR_ENGINE "OPENCL" CACHE STRING "engine use (CUDA or OPENCL)")
set_property(CACHE SOLR_ENGINE PROPERTY STRINGS CUDA OPENCL)

# Windows' math include does not define constants by default.
# Set this definition so it does.
# Also set NOMINMAX so the min and max functions are not overwritten with macros.
IF(MSVC)
    add_definitions(-D_USE_MATH_DEFINES)
    add_definitions(-DNOMINMAX)
else(MSVC)
  if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing -Wno-deprecated-declarations -Wno-macro-redefined -Wno-shift-negative-value -Wno-comment")
  endif(APPLE)
ENDIF(MSVC)

# ================================================================================
# GL
# ================================================================================
find_package(OpenGL REQUIRED SYSTEM)
if (OpenGL_FOUND)
    message(STATUS "OpenGL found and selected for build")
	include_directories(${OPENGL_INCLUDE_DIR})
else()
	message(ERROR " OpenGL not found!")
endif()

# ================================================================================
# GLUT
# ================================================================================
find_package(FreeGLUT REQUIRED SYSTEM)
if (FREEGLUT_FOUND)
    message(STATUS "FreeGlut found and selected for build")
	include_directories(${FREEGLUT_INCLUDE_DIR})
else()
	message(ERROR " FreeGlut not found!")
endif()
	
# ================================================================================
# OpenCL 1.2
# ================================================================================
if (${SOLR_ENGINE} STREQUAL "OPENCL")
  set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE)
  find_package(OpenCL REQUIRED SYSTEM)
  if (OPENCL_FOUND)
    list(APPEND FIND_PACKAGES_DEFINES USE_OPENCL)
    message(STATUS "OpenCL found and selected for build")
    include_directories(${OPENCL_INCLUDE_DIRS})
  else()
    message(ERROR " OpenCL not found. Project will not be built with that technology")
  endif()
endif()

# ================================================================================
# CUDA 8.0
# ================================================================================
if (${SOLR_ENGINE} STREQUAL "CUDA")
  find_package(CUDA REQUIRED SYSTEM)
  if (CUDA_FOUND)
    list(APPEND FIND_PACKAGES_DEFINES USE_CUDA)
    message(STATUS "CUDA found and selected for build")
    include_directories(${CUDA_TOOLKIT_INCLUDE})
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 3.0 3.5 5.0 5.2)
	LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
    if (WIN32)
      set(EXTRA_CXX_FLAGS "/DVERBOSE /D_CRT_SECURE_NO_WARNINGS")
    endif()
    if (UNIX)
      if (APPLE)
        set(EXTRA_CXX_FLAGS "-DVERBOSE -msse2")
      else()
        set(EXTRA_CXX_FLAGS "-DVERBOSE -msse2 -std=c++0x ")
      endif()
    endif()
  else()
    message(ERROR " CUDA not found. Project will not be built with that technology")
  endif()
endif()

# ================================================================================
# KINECT 1.8
# ================================================================================
if(SOLR_KINECT_ENABLED)
	find_package(KINECT REQUIRED SYSTEM)
	if (KINECT_FOUND)
		message(STATUS "Kinect found and selected for build")
		include_directories(${KINECT_INCLUDE_DIR})
		list(APPEND FIND_PACKAGES_DEFINES USE_KINECT)
	else()
		message(ERROR " KINECT not found!")
	endif()
endif()

# ================================================================================
# OCULUS 0.2.5
# ================================================================================
if(SOLR_OCULUS_ENABLED)
	find_package(OculusSDK REQUIRED SYSTEM)
	if (OCULUS_SDK_FOUND)
		message(STATUS "Oculus found and selected for build")
		include_directories(${OCULUS_SDK_INCLUDE_DIR})
		list(APPEND FIND_PACKAGES_DEFINES USE_OCULUS)
	else()
		message(ERROR " OCULUS not found!")
	endif()
endif()

# ================================================================================
# Sixense SDK
# ================================================================================
if(SOLR_SIXENSE_ENABLED)
	find_package(SixenseSdk REQUIRED SYSTEM)
	if (SIXENSESDK_FOUND)
		message(STATUS "Sixense found and selected for build")
		include_directories(${SIXENSESDK_INCLUDE_DIR})
		list(APPEND FIND_PACKAGES_DEFINES USE_SIXENSE)
	else()
		message(ERROR " Sixense not found!")
	endif()
endif()

# ================================================================================
# Project
# ================================================================================
add_subdirectory(solr)
add_subdirectory(apps)

# ================================================================================
# Write defines.h and options.cmake
# ================================================================================
if(NOT OPTIONS_CMAKE)
	set(OPTIONS_CMAKE ${CMAKE_CURRENT_BINARY_DIR}/options.cmake)
endif()
set(DEFINES_FILE "${PROJECT_SOURCE_DIR}/${PROJECT_INCLUDE_NAME}/generated/defines${SYSTEM}.h")
file(WRITE ${DEFINES_FILE}
	"// generated by CMake, do not edit.\n\n"
	"#ifndef ${PROJECT_NAME}_DEFINES_${SYSTEM}_H\n"
	"#define ${PROJECT_NAME}_DEFINES_${SYSTEM}_H\n\n")
file(WRITE ${OPTIONS_CMAKE} "# Optional modules enabled during build\n")
foreach(DEF ${FIND_PACKAGES_DEFINES})
	add_definitions(-D${DEF}=1)
	file(APPEND ${DEFINES_FILE}
	"#ifndef ${DEF}\n"
	"#  define ${DEF} 1\n"
	"#endif\n")
	if(NOT DEF STREQUAL SYSTEM)
		file(APPEND ${OPTIONS_CMAKE} "set(${DEF} ON)\n")
	endif()
endforeach()
file(APPEND ${DEFINES_FILE} "\nstatic const char* DEFAULT_KERNEL_FILENAME=\"${PROJECT_SOURCE_DIR}/solr/engines/opencl/RayTracer.cl\";\n")
file(APPEND ${DEFINES_FILE} "\nstatic const char* DEFAULT_MEDIA_FOLDER=\"${PROJECT_SOURCE_DIR}/medias\";\n")
file(APPEND ${DEFINES_FILE} "\n#endif\n")
