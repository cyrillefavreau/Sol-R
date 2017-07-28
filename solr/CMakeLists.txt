include_directories(${PROJECT_SOURCE_DIR}/solr)

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
	if (KINECT_FOUND)
		set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -D_FORCE_INLINES -DUSE_KINECT --maxrregcount=32 --use_fast_math)
	else()
		set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -D_FORCE_INLINES --maxrregcount=32 --use_fast_math)
	endif()
    CUDA_COMPILE(CUDARAYTRACER_O engines/cuda/CudaRayTracer.cu)
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
		)
endif()

if(OPENCL_FOUND)
    ADD_LIBRARY(
		solr 
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
        )
endif()

INSTALL(TARGETS solr DESTINATION lib)
