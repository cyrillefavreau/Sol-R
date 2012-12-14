################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/Cuda/CudaKernel.cpp 

CU_SRCS += \
../src/Cuda/CudaRayTracer.cu 

CU_DEPS += \
./src/Cuda/CudaRayTracer.d 

OBJS += \
./src/Cuda/CudaKernel.o \
./src/Cuda/CudaRayTracer.o 

CPP_DEPS += \
./src/Cuda/CudaKernel.d 


# Each subdirectory must supply rules for building sources it contributes
src/Cuda/%.o: ../src/Cuda/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -DRAYTRACINGENGINE_EXPORTS -DUSE_CUDA -I/usr/local/cuda/include -I/home/geek/NVIDIA_GPU_Computing_SDK/C/common/inc -O3 -gencode arch=compute_10,code=sm_10 -gencode arch=compute_13,code=sm_13 -odir "src/Cuda" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -DRAYTRACINGENGINE_EXPORTS -DUSE_CUDA -I/usr/local/cuda/include -I/home/geek/NVIDIA_GPU_Computing_SDK/C/common/inc -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/Cuda/%.o: ../src/Cuda/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -DRAYTRACINGENGINE_EXPORTS -DUSE_CUDA -I/usr/local/cuda/include -I/home/geek/NVIDIA_GPU_Computing_SDK/C/common/inc -O3 -gencode arch=compute_10,code=sm_10 -gencode arch=compute_13,code=sm_13 -odir "src/Cuda" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -DRAYTRACINGENGINE_EXPORTS -DUSE_CUDA -I/usr/local/cuda/include -I/home/geek/NVIDIA_GPU_Computing_SDK/C/common/inc -O3 -gencode arch=compute_10,code=compute_10 -gencode arch=compute_10,code=sm_10 -gencode arch=compute_13,code=compute_13 -gencode arch=compute_13,code=sm_13  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


