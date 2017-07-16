INCLUDE(${PROJECT_SOURCE_DIR}/cmake/glut.cmake)

SET(SOLR_SOURCES
  GPUKernel.cpp
  io/PDBReader.cpp
  io/OBJReader.cpp
  io/FileMarshaller.cpp
  io/SWCReader.cpp
  images/jpge.cpp
  images/jpgd.cpp
  images/ImageLoader.cpp
  images/tgad.cpp
  games/MapReader.cpp
  RayTracingEngineStub.cpp
  opengl/rtgl.cpp
  cpu/CPUKernel.cpp
)

SET(SOLR_HEADERS
  RayTracingEngineStub.h
  GPUKernel.h
  Logging.h
  cuda/VectorUtils.cuh
  cuda/GeometryIntersections.cuh
  cuda/GeometryShaders.cuh
  cuda/CudaKernel.h
  cuda/CudaRayTracer.h
  cuda/CudaDataTypes.h
  cuda/TextureMapping.cuh
  RayTracingEngineStub.h
  io/FileMarshaller.cpp
  io/FileMarshaller.h
  io/PDBReader.h
  io/OBJReader.h
  io/SWCReader.h
  DLL_API.h
  images/jpge.h
  images/jpgd.h
  images/tgad.h
  images/ImageLoader.h
  games/MapReader.h
  opengl/rtgl.h
  Consts.h
  cpu/CPUKernel.h
  resource.h
)

if (CUDA_FOUND)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -D_FORCE_INLINES --maxrregcount=32)
    CUDA_COMPILE(CUDARAYTRACER_O cuda/CudaRayTracer.cu)
    CUDA_ADD_LIBRARY(
      solr
      cuda/CudaKernel.cpp
      ${CUDARAYTRACER_O}
      ${SOLR_SOURCES}
      )
    TARGET_LINK_LIBRARIES(
      solr
      ${CUDARAYTRACER}
      ${GLUT_LIBRARIES}
      )
endif()

if(OPENCL_FOUND)
    ADD_LIBRARY(
      solr
      opencl/OpenCLKernel.cpp
      ${SOLR_SOURCES}
    )

    TARGET_LINK_LIBRARIES(
        solr
        ${OpenCL_LIBRARIES}
        ${GLUT_LIBRARIES}
    )
endif()

INSTALL(TARGETS solr DESTINATION lib)
