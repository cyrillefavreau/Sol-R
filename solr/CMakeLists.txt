include_directories(${PROJECT_SOURCE_DIR}/solr ${PROJECT_SOURCE_DIR}/generated)

SET(SOLR_SOURCES
    DLL_API.h
    Logging.h
    types.h
    Consts.h
    RayTracingEngineStub.cpp
    RayTracingEngineStub.h
    engines/GPUKernel.cpp
    engines/GPUKernel.h
    #engines/cpu/CPUKernel.cpp
    #engines/cpu/CPUKernel.h
    io/PDBReader.cpp
    io/PDBReader.h
    io/OBJReader.cpp
    io/OBJReader.h
    io/FileMarshaller.cpp
    io/SWCReader.cpp
    io/SWCReader.h
    io/FileMarshaller.cpp
    io/FileMarshaller.h
    images/ImageLoader.cpp
    images/ImageLoader.h
    images/jpge.cpp
    images/jpge.h
    images/jpgd.cpp
    images/jpgd.h
    images/tgad.cpp
    images/tgad.h
    opengl/rtgl.cpp
    opengl/rtgl.h
    games/MapReader.cpp
    games/MapReader.h
)

if (CUDA_FOUND)
	set(CUDA_INCLUDE_USER_DIRECTORIES -I${PROJECT_SOURCE_DIR}/solr -I${PROJECT_SOURCE_DIR}/generated)
	set(CUDA_NVCC_USER_ARGS --maxrregcount=32 --use_fast_math)
	set(CUDA_64_BIT_DEVICE_CODE ON)
	set(CUDA_VERBOSE_BUILD ON)
	if (KINECT_FOUND)
		list(APPEND CUDA_NVCC_USER_ARGS -DUSE_KINECT)
	endif()
	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${CUDA_INCLUDE_USER_DIRECTORIES} ${CUDA_NVCC_USER_ARGS})
    CUDA_COMPILE(
		CUDARAYTRACER_O engines/cuda/CudaRayTracer.cu)
    CUDA_ADD_LIBRARY(
		solr
		engines/cuda/CudaKernel.cpp
		engines/cuda/CudaKernel.h
		engines/cuda/CudaRayTracer.h
		engines/cuda/TextureMapping.cuh
		engines/cuda/VectorUtils.cuh
		engines/cuda/GeometryIntersections.cuh
		engines/cuda/GeometryShaders.cuh
		${CUDARAYTRACER_O}
		${SOLR_SOURCES}
		)
    TARGET_LINK_LIBRARIES(
		solr
		${FREEGLUT_LIBRARIES}
		${OPENGL_gl_LIBRARY}
		${KINECT_LIBRARIES}
		${OCULUS_SDK_LIBRARIES}
		${SIXENSESDK_LIBRARIES}
		${LEAP_LIBRARY}
		)
endif()

if(OPENCL_FOUND)
    ADD_LIBRARY(
		solr STATIC 
		engines/opencl/OpenCLKernel.cpp 
                engines/opencl/RayTracer.cl
                ${SOLR_SOURCES})

    TARGET_LINK_LIBRARIES(
		solr
        ${OPENCL_LIBRARIES}
        ${OpenCL_LIBRARIES}
        ${FREEGLUT_LIBRARIES}
        ${OPENGL_gl_LIBRARY}
		${KINECT_LIBRARIES}
		${OCULUS_SDK_LIBRARIES}
		${SIXENSESDK_LIBRARIES}
		${LEAP_LIBRARY}
        )
endif()

INSTALL(TARGETS solr DESTINATION lib)
